# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/19

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int, 
                 commitment_cost : float=0.25):
        """
        Vector quantizer initialization.

        :param int num_embeddings: Codebook size.
        :param int embedding_dim: Dimension of each embedding.
        :param float commitment_cost: Commitment loss multiplier, defaults to 0.25
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # TODO REVISIT
        # check if: 1) using weight.data instead of init makes sense
        # 2) if the code for initialization makes sense (especially dividing by num_embeddings)
        # for example, LATENT_PLANNING does this: module.weight.data.normal_(mean=0.0, std=0.02)
        # haven't found any mathematically elaborate explanation of why this initialization makes sense
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs [(batch_size, sequence_length) OR (batch_size x sequence_length), embedding_dim]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim) # [ (batch x seq_len), emb_dim ]

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)           # [ (batch x seq_len), emb_num ]    
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.T))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1) # [ (batch x seq_len) ]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,  # [ (batch x seq_len), emb_num ]
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        # Perplexity: diversity of latent code usage, keep it mid (high=uniform, no learning, low=not used fully)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1])

class GPT2VQVAE(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 num_thoughts=32, n_positions=1024):
        """
        GPT2-based VQ-VAE model that uses GPT2 as both encoder and decoder.
        
        Args:
            vocab_size (int): Vocabulary size
            d_model (int): Model dimension (default: 768 for GPT2)
            num_embeddings (int): VQ codebook size
            commitment_cost (float): VQ commitment cost
            aggregation_hidden_dim (int): Aggregation MLP hidden dimension
            num_thoughts (int): Number of parallel sequences
            n_positions (int): Maximum sequence length for GPT2 (default: 1024)
        """
        super(GPT2VQVAE, self).__init__()

        # TODO ADD INITIALIZATION FOR ENCODER, DECODER AND MLP
        # THOUGHT: COULD ADD output_attentions=True FOR DEBUGGING (E.G. FOR HAND-MADE CROSS-ATTENTION MASK OF DECODER)
        
        # Load GPT2 model and config
        self.encoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_positions=n_positions,
        )
        
        # Create separate config for decoder with cross-attention
        self.decoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_positions=n_positions,
            add_cross_attention=True,  # Enable cross-attention for decoder
            is_decoder=True,  # Mark as decoder
        )
        
        # Initialize encoder and decoder with GPT2
        self.encoder = GPT2Model(self.encoder_config)
        self.decoder = GPT2LMHeadModel(self.decoder_config)
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        
        # Aggregation MLP
        # TODO: Consider different options - might cause posterior collapse if too strong
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(num_thoughts * d_model, aggregation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # GPT2's default dropout
            nn.Linear(aggregation_hidden_dim, d_model)
        )
        
        # Chain-positional embeddings to differentiate M sequences
        self.chain_embeddings = nn.Embedding(num_thoughts, d_model)
        # Initialize with small values
        nn.init.normal_(self.chain_embeddings.weight, mean=0.0, std=0.02)
        
        self.d_model = d_model
        self.num_thoughts = num_thoughts
        
    def aggregate(self, memory, mode="linear"):
        """
        Aggregates same-position embeddings from chains sharing the same prompt.
        
        Args:
            memory (torch.Tensor): Shape [batch_size*L, num_thoughts, d_model]
            mode (str): Aggregation mode, defaults to "linear"
            
        Returns:
            torch.Tensor: Aggregated embeddings [batch_size*L, d_model]
        """
        _, M, d_model = memory.shape
        if mode == "linear":
            # Reshape to pass through MLP
            memory = memory.view(-1, M*d_model)         # [batch_size*L, M*d_model]
            # Pass through MLP for non-linear transformation
            aggregated = self.aggregation_mlp(memory)   # [batch_size*L, d_model]
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
            
        return aggregated
        
    def encode(self, combined_sequences, K, L, combined_mask=None, aggregate_mode="linear", quantize_cot_only=True):
        """
        !!! TODO: MAKE EFFICIENT USING KV CACHING !!!
        Encodes BATCH * M sequences of length K+L into BATCH sequences of latent tokens.
        Is done causally with GPT2 (ith token only sees up to ith tokens from M sequences).

        Args:
            combined_sequences (torch.Tensor): Combined prompt + COT sequences [batch_size, M, K+L]
            K (int): Length of prompt sequences
            L (int): Length of COT sequences
            combined_mask (torch.Tensor, optional): Attention mask for padding
            aggregate_mode (str): Mode of aggregation
            quantize_cot_only (bool): If True, only quantize COT positions (K to K+L-1). 
                                    If False, quantize all positions (0 to K+L-1).
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, indices) 
        """
        batch_size, M, total_len = combined_sequences.shape
        
        # Reshape to process all sequences together
        combined_sequences = combined_sequences.view(batch_size * M, total_len)
        if combined_mask is not None:
            combined_mask = combined_mask.view(batch_size * M, total_len)
        
        # Get GPT2 encoder outputs
        encoder_outputs = self.encoder(
            input_ids=combined_sequences,
            attention_mask=combined_mask,
            use_cache=False,
            return_dict=True
        )
        
        # Get the last hidden state
        memory = encoder_outputs.last_hidden_state
        
        # Reshape to group tokens at same positions across sequences
        memory = memory.view(batch_size, M, total_len, -1)
        memory = memory.transpose(1, 2)  # [batch_size, total_len, M, d_model]
        
        if quantize_cot_only:
            # Only keep memory positions corresponding to COT sequences (K to K+L-1)
            memory = memory[:, K:K+L, :, :]  # [batch_size, L, M, d_model]

        memory = memory.reshape(-1, M, memory.size(-1))  # [batch_size*L or batch_size*total_len, M, d_model]

        # aggregate the memory content per-prompt into single chains 
        aggregated = self.aggregate(memory, mode=aggregate_mode) # [batch_size*L or batch_size*total_len, d_model]
        
        # Apply VQ
        quantized, vq_loss, perplexity, indices = self.vector_quantizer(aggregated) # [batch_size*L or batch_size*total_len, d_model]
        
        # Tile back to obtain the same shape and amount of info
        # Expand vs repeat: more memory efficient + no in-place change
        quantized = quantized.unsqueeze(2).expand(-1, -1, M, -1)  # [batch_size*(L or total_len), M, d_model]
        
        # Reshape back using the appropriate length
        quantized = quantized.view(batch_size, -1, M, quantized.size(-1))
        indices = indices.view(batch_size, -1)
        
        return quantized, vq_loss, perplexity, indices
        
    def decode(self, memory, combined_sequences, K, L, combined_mask=None):
        """
        !!! TODO: MAKE EFFICIENT USING KV CACHING !!!
        !!! TODO: MAKE MORE EFFICIENT USE OF CACHING WHEN INTRODUCING NEW MEMORY CELLS
                 AS IT IS MEANT TO DYNAMICALLY GROW !!!
        Decodes using GPT2 decoder with chain-positional embeddings.
        
        Args:
            memory (torch.Tensor): Encoded memory [batch_size, L, M, d_model]
            combined_sequences (torch.Tensor): Combined prompt + COT sequences [batch_size, M, K+L]
            K (int): Length of prompt sequences
            L (int): Length of COT sequences (memory length)
            combined_mask (torch.Tensor, optional): Attention mask for padding
            
        Returns:
            torch.Tensor: Decoded output logits [batch_size, M, K+L, vocab_size]
        """
        batch_size, _, M, _ = memory.shape
        
        # Prepare input sequences
        input_sequences = combined_sequences.view(batch_size * M, -1) # [batch_size * M, K+L]
        if combined_mask is not None:
            combined_mask = combined_mask.view(batch_size * M, -1)              # [batch_size * M, K+L]
        
        # Reshape memory for decoder
        memory = memory.transpose(1, 2)  # [batch_size, M, L, d_model]
        memory = memory.reshape(batch_size * M, L, -1)
        
        # Add chain-positional embeddings to memory
        chain_indices = torch.arange(M, device=memory.device).repeat(batch_size)
        chain_emb = self.chain_embeddings(chain_indices).unsqueeze(1)  # [batch_size*M, 1, d_model]
        memory = memory + chain_emb  # Add to all positions in the sequence
        
        # Create cross-attention mask for memory attention
        # By passing a mask with the right shape, we can directly influence computation within the cross-attention module
        # The attention mask will ultimately be of the same shape as the causal mask generated by 
        # GPT2Model._prepare_4d_causal_attention_mask_with_cache_position because it will be passed into the same 
        # function as part of GPT2Attention's forward
        # The final form is of shape: 1x1x(query_len)x(key_len), or a broadcasted version of this

        # Create cross-attention mask: (K+L) x L
        cross_attention_mask = create_cross_attention_mask(K + L, L, memory.device)
        
        # Expand to match batch and sequence dimensions for GPT2
        # Shape: [batch_size*M, 1, K+L, L]
        cross_attention_mask = cross_attention_mask.unsqueeze(0).unsqueeze(0)
        cross_attention_mask = cross_attention_mask.expand(batch_size * M, -1, K + L, L)
        
        # Get GPT2 decoder outputs with language modeling head
        decoder_outputs = self.decoder(
            input_ids=input_sequences,
            attention_mask=combined_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=cross_attention_mask,  # Apply cross-attention mask to memory attention
            use_cache=False,
            return_dict=True
        )

        # Reshape back to [batch_size, M, K+L, vocab_size]
        return decoder_outputs.logits.view(batch_size, M, K+L, -1)
        
    def forward(self, prompt, cot_sequences, cot_mask=None, prompt_mask=None, inference=False, quantize_cot_only=True):
        """
        Forward pass through the model.
        
        Args:
            prompt (torch.Tensor): Prompt sequences [batch_size, M, K] where K is prompt length
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            cot_mask (torch.Tensor, optional): Chain-of-thought attention mask for padding
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            inference (bool): If True, performs inference without teacher forcing
            quantize_cot_only (bool): If True, only quantize the COT portion of sequences
            
        Returns:
            tuple: (output_sequences, output_logits, vq_loss, perplexity)
                - output_sequences: Generated token sequences [batch_size, M, L]
                - output_logits: Token logits for each position [batch_size, M, L, vocab_size]
                - vq_loss: Vector quantization loss
                - perplexity: Codebook usage perplexity
        """
        batch_size, M, K = prompt.shape
        _, _, L = cot_sequences.shape
        
        # Concatenate prompts with COT sequences for encoding
        combined_sequences = torch.cat([prompt, cot_sequences], dim=2) # [batch_size, M, K+L]
        
        if cot_mask is None and prompt_mask is None:
            combined_mask = None
        else: # Create combined mask if one or both masks are provided
            if prompt_mask is None: # Assume prompts are complete
                prompt_mask = torch.ones((batch_size, M, K), device=cot_mask.device)
            elif cot_mask is None: # Assume COTs are complete
                cot_mask = torch.ones((batch_size, M, L), device=prompt_mask.device)
            
            combined_mask = torch.cat([prompt_mask, cot_mask], dim=2)
        
        # Encode the combined sequences (prompt + COT)
        quantized, vq_loss, perplexity, _ = self.encode(combined_sequences, K, L, combined_mask, 
                                                        quantize_cot_only=quantize_cot_only)
        
        # quantized shape depends on quantize_cot_only:
        if quantize_cot_only:          # if True: [batch_size, L, M, d_model] (only COT positions)
            cot_quantized = quantized
        else:                          # if False: [batch_size, K+L, M, d_model] (all positions)
            cot_quantized = quantized[:, K:, :, :]
        
        
        # Initialize tensor to store generation results and logits
        output_sequences = torch.empty((batch_size, M, L), dtype=torch.long, device=cot_sequences.device)
        output_logits = torch.empty((batch_size, M, L, self.decoder_config.vocab_size), device=cot_sequences.device)
        
        if not inference:
            # During training, use teacher forcing with single forward pass to get all logits
            full_logits = self.decode(cot_quantized, combined_sequences, K, L, combined_mask) # [batch_size, M, K+L, vocab_size]
            
            # Extract only the logits corresponding to COT positions : K to K+L-1
            output_logits = full_logits[:, :, K:, :]  # [batch_size, M, L, vocab_size]
            
            # Get the predicted tokens from logits
            output_sequences = torch.argmax(output_logits, dim=-1)
        else:
            # During inference, generate sequence auto-regressively
            for t in range(L):
                # Use prompt + generated sequence for decoding
                current_input = torch.cat([prompt, output_sequences[:, :, :t]], dim=2)
                current_output = self.decode(cot_quantized, current_input, K, t, combined_mask) # [batch_size, M, K+t, vocab_size]
                
                # Get next token predictions
                next_token_logits = current_output[:, :, -1, :]  # [batch_size, M, vocab_size]
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # [batch_size, M]
                
                # Store logits + new token for current position
                output_logits[:, :, t, :] = next_token_logits
                output_sequences[:, :, t] = next_tokens
        
        return output_sequences, output_logits, vq_loss, perplexity

def create_cross_attention_mask(query_length, key_length, device):
    """
    Create a cross-attention mask for memory attention.
    
    Args:
        query_length (int): Length of the query sequence (K+L)
        key_length (int): Length of the key sequence (L)
        device (torch.device): Device to create the mask on
        
    Returns:
        torch.Tensor: Cross-attention mask of shape (query_length, key_length)
                     The ith row can attend to key positions 0 to (i-(query_length-key_length))
    """
    # Calculate the offset: query_length - key_length = K
    offset = query_length - key_length
    
    # Vectorized implementation
    i = torch.arange(query_length, device=device).unsqueeze(1)  # [query_length, 1]
    j = torch.arange(key_length, device=device).unsqueeze(0)    # [1, key_length]
    mask = (j <= i - offset).float()
    
    return mask