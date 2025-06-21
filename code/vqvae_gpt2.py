# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/21

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

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
                 num_thoughts=32, n_positions=1024, 
                 use_pretrained_encoder=True, use_pretrained_decoder=True,
                 pretrained_model_name="gpt2"):
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
            use_pretrained_encoder (bool): Whether to load pretrained weights for encoder
            use_pretrained_decoder (bool): Whether to load pretrained weights for decoder
            pretrained_model_name (str): Name of pretrained model to load (default: "gpt2")
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
        
        # Initialize encoder with or without pretrained weights
        if use_pretrained_encoder:
            print(f"Loading pretrained {pretrained_model_name} weights for encoder...")
            self.encoder = GPT2Model.from_pretrained(pretrained_model_name, config=self.encoder_config)
            # Ensure the encoder uses our config (in case vocab_size differs)
            if self.encoder.config.vocab_size != vocab_size:
                print(f"Warning: Pretrained model vocab_size ({self.encoder.config.vocab_size}) "
                      f"differs from specified vocab_size ({vocab_size}). "
                      f"Using specified vocab_size.")
                self.encoder.resize_token_embeddings(vocab_size)
        else:
            print("Initializing encoder with random weights...")
            self.encoder = GPT2Model(self.encoder_config)
        
        # Initialize decoder with or without pretrained weights
        if use_pretrained_decoder:
            print(f"Loading pretrained {pretrained_model_name} weights for decoder...")
            self.decoder = GPT2LMHeadModel.from_pretrained(pretrained_model_name, config=self.decoder_config)
            # Ensure the decoder uses our config
            if self.decoder.config.vocab_size != vocab_size:
                print(f"Warning: Pretrained model vocab_size ({self.decoder.config.vocab_size}) "
                      f"differs from specified vocab_size ({vocab_size}). "
                      f"Using specified vocab_size.")
                self.decoder.resize_token_embeddings(vocab_size)
        else:
            print("Initializing decoder with random weights...")
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
            torch.Tensor: Aggregated embeddings [batch_size*L or batch_size*(K+L), d_model]
        """
        _, M, d_model = memory.shape
        if mode == "linear":
            # Reshape to pass through MLP
            memory = memory.reshape(-1, M*d_model)         # [batch_size*L, M*d_model]
            # Pass through MLP for non-linear transformation
            aggregated = self.aggregation_mlp(memory)   # [batch_size*L, d_model]
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
            
        return aggregated
        
    def _create_combined_mask(self, prompt_mask, cot_mask_flat, batch_size, M, K, L, device):
        """
        Create combined attention mask from prompt and COT masks.
        
        Args:
            prompt_mask (torch.Tensor, optional): Prompt attention mask [batch_size, K]
            cot_mask_flat (torch.Tensor, optional): Flattened COT attention mask [batch_size * M, L]
            batch_size (int): Batch size
            M (int): Number of COT sequences
            K (int): Prompt length
            L (int): COT length
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Combined attention mask [batch_size * M, K + L]
        """
        if prompt_mask is not None and cot_mask_flat is not None:
            prompt_mask_expanded = prompt_mask.unsqueeze(1).expand(-1, M, -1).reshape(batch_size * M, K)
            combined_mask = torch.cat([prompt_mask_expanded, cot_mask_flat], dim=1)
        elif prompt_mask is not None:
            prompt_mask_expanded = prompt_mask.unsqueeze(1).expand(-1, M, -1).reshape(batch_size * M, K)
            cot_mask_ones = torch.ones(batch_size * M, L, device=device)
            combined_mask = torch.cat([prompt_mask_expanded, cot_mask_ones], dim=1)
        elif cot_mask_flat is not None:
            prompt_mask_ones = torch.ones(batch_size * M, K, device=device)
            combined_mask = torch.cat([prompt_mask_ones, cot_mask_flat], dim=1)
        else:
            combined_mask = None
            
        return combined_mask
    
    def _pad_kv_cache(self, prompt_cache, batch_size, M, K):
        """
        Pad KV cache M times to match COT batch size.
        
        Args:
            prompt_cache: KV cache from prompt processing (Cache object or legacy tuple format)
            batch_size (int): Batch size
            M (int): Number of COT sequences
            K (int): Prompt length
            
        Returns:
            Padded KV cache in the same format as input (Cache object or legacy tuple)
        """
        # Check if it's a Cache object (like DynamicCache)
        if hasattr(prompt_cache, 'key_cache') and hasattr(prompt_cache, 'value_cache'):
            # It's a Cache object (e.g., DynamicCache)
            padded_cache = type(prompt_cache)()  # Create new instance of same type
            
            # Pad each layer's key and value tensors
            for layer_idx in range(len(prompt_cache.key_cache)):
                key_tensor = prompt_cache.key_cache[layer_idx]  # [batch_size, num_heads, K, head_dim]
                value_tensor = prompt_cache.value_cache[layer_idx]  # [batch_size, num_heads, K, head_dim]
                
                # Expand from [batch_size, num_heads, K, head_dim] to [batch_size * M, num_heads, K, head_dim]
                padded_key = key_tensor.unsqueeze(1).expand(-1, M, -1, -1, -1)
                padded_key = padded_key.reshape(batch_size * M, key_tensor.size(1), K, key_tensor.size(-1))
                
                padded_value = value_tensor.unsqueeze(1).expand(-1, M, -1, -1, -1)
                padded_value = padded_value.reshape(batch_size * M, value_tensor.size(1), K, value_tensor.size(-1))
                
                # Update the cache with padded tensors
                padded_cache.update(padded_key, padded_value, layer_idx)
                
        else:
            # It's a legacy tuple format: Tuple[Tuple[torch.Tensor, torch.Tensor]]
            padded_cache = []
            for layer_cache in prompt_cache:
                # Each layer cache is a tuple of (key, value) for each layer
                # key/value shape: [batch_size, num_heads, seq_len, head_dim]
                padded_layer_cache = []
                for kv in layer_cache:
                    # Expand from [batch_size, num_heads, K, head_dim] to [batch_size * M, num_heads, K, head_dim]
                    padded_kv = kv.unsqueeze(1).expand(-1, M, -1, -1, -1)
                    padded_kv = padded_kv.reshape(batch_size * M, kv.size(1), K, kv.size(-1))
                    padded_layer_cache.append(padded_kv)
                padded_cache.append(tuple(padded_layer_cache))
            
            # Convert back to tuple format
            padded_cache = tuple(padded_cache)
            
        return padded_cache
        
    def encode(self, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None, 
               aggregate_mode="linear", quantize_cot_only=True):
        """
        Encodes prompt sequences first with caching, then processes COT sequences with padded prompt activations.
        
        Args:
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            aggregate_mode (str): Mode of aggregation
            quantize_cot_only (bool): If True, only quantize COT positions (K to K+L-1). 
                                    If False, quantize all positions (0 to K+L-1).
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, indices) 
        """
        batch_size, K = prompt_sequences.shape
        _, M, L = cot_sequences.shape
        
        # Step 1: Encode prompt sequences with caching
        
        # Get GPT2 encoder outputs for prompt with caching
        prompt_outputs = self.encoder(
            input_ids=prompt_sequences,
            attention_mask=prompt_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True
        )
        
        # Extract prompt activations and cache
        prompt_activations = prompt_outputs.last_hidden_state  # [batch_size, K, d_model]
        prompt_cache = prompt_outputs.past_key_values
        
        # Step 2: Process COT sequences with padded prompt cache
        # Reshape COT sequences to [batch_size * M, L]
        cot_flat = cot_sequences.view(batch_size * M, L)  # [batch_size * M, L]
        if cot_mask is not None:
            cot_mask_flat = cot_mask.view(batch_size * M, L)  # [batch_size * M, L]
        else:
            cot_mask_flat = None
        
        # Pad prompt cache M times using helper method
        padded_cache = self._pad_kv_cache(prompt_cache, batch_size, M, K)
        
        # Step 3: Continue encoding with COT sequences using cached prompt activations
        # Create combined mask using helper method
        combined_mask = self._create_combined_mask(prompt_mask, cot_mask_flat, batch_size, M, K, L, cot_flat.device)
        
        # Continue encoding with cached activations - only pass COT sequences as input_ids
        cot_outputs = self.encoder(
            input_ids=cot_flat,  # Only COT sequences, not combined input
            attention_mask=combined_mask,
            past_key_values=padded_cache,  # Use padded prompt cache
            use_cache=False,
            return_dict=True
        )
        
        # Get COT hidden states (only the newly computed activations)
        cot_memory = cot_outputs.last_hidden_state  # [batch_size * M, L, d_model]
        
        if quantize_cot_only:
            # Only use COT activations, no need to concatenate with prompt
            # Reshape to group tokens at same positions across sequences
            cot_memory = cot_memory.view(batch_size, M, L, -1)
            memory = cot_memory.transpose(1, 2)  # [batch_size, L, M, d_model]
        else:
            # Pad prompt activations M times to match COT batch size
            # [batch_size, K, d_model] -> [batch_size * M, K, d_model]
            padded_prompt_activations = prompt_activations.unsqueeze(1).expand(-1, M, -1, -1)
            padded_prompt_activations = padded_prompt_activations.reshape(batch_size * M, K, -1)
            
            # Concatenate prompt activations with COT activations to get full sequence
            # padded_prompt_activations: [batch_size * M, K, d_model]
            # cot_memory: [batch_size * M, L, d_model]
            full_memory = torch.cat([padded_prompt_activations, cot_memory], dim=1)  # [batch_size * M, K + L, d_model]
            
            # Reshape to group tokens at same positions across sequences
            full_memory = full_memory.view(batch_size, M, K + L, -1)
            memory = full_memory.transpose(1, 2)  # [batch_size, K + L, M, d_model]

        # Reshape for aggregation
        memory = memory.reshape(-1, M, memory.size(-1))  # [batch_size*L or batch_size*(K+L), M, d_model]

        # Aggregate the memory content per-prompt into single chains 
        aggregated = self.aggregate(memory, mode=aggregate_mode) # [batch_size*L or batch_size*(K+L), d_model]
        
        # Apply VQ
        quantized, vq_loss, perplexity, indices = self.vector_quantizer(aggregated) # [batch_size*L or batch_size*(K+L), d_model]
        
        # Tile back to obtain the same shape and amount of info
        # Expand vs repeat: more memory efficient + no in-place change
        quantized = quantized.unsqueeze(1).expand(-1, M, -1)  # [batch_size*(L or K+L), M, d_model]
        
        # Reshape back using the appropriate length
        quantized = quantized.view(batch_size, -1, M, quantized.size(-1))
        indices = indices.view(batch_size, -1)
        
        return quantized, vq_loss, perplexity, indices
        
    def decode(self, memory, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None):
        """
        Decodes using GPT2 decoder with chain-positional embeddings and KV caching optimization.
        
        Args:
            memory (torch.Tensor): Encoded memory [batch_size, L, M, d_model]
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            
        Returns:
            torch.Tensor: Decoded output logits [batch_size, M, L, vocab_size] (only COT positions)
        """
        batch_size, K = prompt_sequences.shape
        _, M, L = cot_sequences.shape
        
        # Step 1: Decode prompt sequences with caching
        # Get GPT2 decoder outputs for prompt with caching
        prompt_outputs = self.decoder(
            input_ids=prompt_sequences,
            attention_mask=prompt_mask,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True
        )
        
        # Extract prompt cache
        prompt_cache = prompt_outputs.past_key_values
        
        # Step 2: Process COT sequences with padded prompt cache
        # Reshape COT sequences to [batch_size * M, L]
        cot_flat = cot_sequences.view(batch_size * M, L)  # [batch_size * M, L]
        if cot_mask is not None:
            cot_mask_flat = cot_mask.view(batch_size * M, L)  # [batch_size * M, L]
        else:
            cot_mask_flat = None
        
        # Pad prompt cache M times using helper method
        padded_cache = self._pad_kv_cache(prompt_cache, batch_size, M, K)
        
        # Step 3: Continue decoding with COT sequences using cached prompt activations
        # Create combined mask using helper method
        combined_mask = self._create_combined_mask(prompt_mask, cot_mask_flat, batch_size, M, K, L, cot_flat.device)
        
        # Reshape memory for decoder
        memory = memory.transpose(1, 2)  # [batch_size, M, L, d_model]
        memory = memory.reshape(batch_size * M, L, -1)
        
        # Add chain-positional embeddings to memory
        chain_indices = torch.arange(M, device=memory.device).repeat(batch_size)
        chain_emb = self.chain_embeddings(chain_indices).unsqueeze(1)  # [batch_size*M, 1, d_model]
        memory = memory + chain_emb  # Add to all positions in the sequence
        
        # Create cross-attention mask for memory attention: (K+L) x L
        cross_attention_mask = create_cross_attention_mask(K + L, L, memory.device)
        
        # Expand to match batch and sequence dimensions for GPT2
        # Shape: [batch_size*M, 1, K+L, L]
        cross_attention_mask = cross_attention_mask.unsqueeze(0).unsqueeze(0).expand(
            batch_size * M, -1, K + L, L)
        
        # Wrap padded_cache in EncoderDecoderCache for cross-attention
        # GPT2 expects EncoderDecoderCache when encoder_hidden_states are provided
        encoder_decoder_cache = EncoderDecoderCache(padded_cache, DynamicCache())
        
        # Get GPT2 decoder outputs with language modeling head using cached prompt activations
        # Only pass COT sequences as input_ids, not the combined sequence
        decoder_outputs = self.decoder(
            input_ids=cot_flat,  # Only COT sequences, not combined input
            attention_mask=combined_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=cross_attention_mask,  # Apply cross-attention mask to memory attention
            past_key_values=encoder_decoder_cache,  # Use EncoderDecoderCache for cross-attention
            use_cache=False,
            return_dict=True
        )

        # Get COT logits (only the newly computed logits for COT positions)
        cot_logits = decoder_outputs.logits  # [batch_size * M, L, vocab_size]
        
        # Reshape back to [batch_size, M, L, vocab_size] (only COT positions)
        return cot_logits.view(batch_size, M, L, -1)
        
    def forward(self, prompt, cot_sequences, cot_mask=None, prompt_mask=None, inference=False, quantize_cot_only=True):
        """
        Forward pass through the model.
        
        Args:
            prompt (torch.Tensor): Prompt sequences [batch_size, K] where K is prompt length
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
        batch_size, K = prompt.shape
        _, M, L = cot_sequences.shape
        
        # Encode using the new separate prompt and COT approach
        quantized, vq_loss, perplexity, _ = self.encode(
            prompt, cot_sequences, 
            prompt_mask, cot_mask, 
            quantize_cot_only=quantize_cot_only
        )
        
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
            output_logits = self.decode(cot_quantized, prompt, cot_sequences, prompt_mask, cot_mask) # [batch_size, M, L, vocab_size]
            
            # Get the predicted tokens from logits
            output_sequences = torch.argmax(output_logits, dim=-1)
        else:
            # TODO IF I FIND THE TIME : USE KV-CACHING TO SPEED UP AUTO-REGRESSIVE GENERATION
            # During inference, generate sequence auto-regressively
            for t in range(L):
                current_cot = output_sequences[:, :, :t]  # [batch_size, M, t]
                current_output = self.decode(cot_quantized, prompt, current_cot, prompt_mask, cot_mask[:, :, :t] if cot_mask is not None else None)
                
                # Get next token predictions
                next_token_logits = current_output[:, :, -1, :]  # [batch_size, M, vocab_size]
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # [batch_size, M]
                
                # Store logits + new token for current position
                output_logits[:, :, t, :] = next_token_logits
                output_sequences[:, :, t] = next_tokens
        
        return output_sequences, output_logits, vq_loss, perplexity

def create_cross_attention_mask(query_length, key_length, device, dtype=torch.float32):
    """
    Create a cross-attention mask for memory attention.
    
    Args:
        query_length (int): Length of the query sequence (K+L)
        key_length (int): Length of the key sequence (L)
        device (torch.device): Device to create the mask on
        dtype (torch.dtype): Data type for the mask
        
    Returns:
        torch.Tensor: Cross-attention mask of shape (query_length, key_length)
                     The ith row can attend to key positions 0 to (i-(query_length-key_length))
                     Uses 0 for attended positions and torch.finfo(dtype).min for masked positions
    """
    # Calculate the offset: query_length - key_length = K
    offset = query_length - key_length
    
    # Use the same pattern as _prepare_4d_causal_attention_mask_with_cache_position
    min_dtype = torch.finfo(dtype).min
    mask = torch.full(
        (query_length, key_length), 
        fill_value=min_dtype, 
        dtype=dtype, 
        device=device
    )
    
    # Vectorized implementation: set positions that can be attended to as 0
    i = torch.arange(query_length, device=device).unsqueeze(1)  # [query_length, 1]
    j = torch.arange(key_length, device=device).unsqueeze(0)    # [1, key_length]
    attend_mask = (j <= i - offset)  # [query_length, key_length]
    
    # Set attended positions to 0 (can attend) and keep masked positions as min_dtype (cannot attend)
    mask = torch.where(attend_mask, torch.tensor(0.0, dtype=dtype, device=device), mask)
    
    return mask