# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/17

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int, 
                 commitment_cost : float=0.25):
        """
        Initializes the vector quantizer.

        :param int num_embeddings: The size of the codebook.
        :param int embedding_dim: The dimension size of each embedding / code.
        :param float commitment_cost: Multiplier for commitment loss, defaults to 0.25
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create the embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # TODO REVISIT
        # check if: 1) using weight.data instead of init makes sense
        # 2) if the code for initialization makes sense (especially dividing by num_embeddings)
        # for example, LATENT_PLANNING does this: module.weight.data.normal_(mean=0.0, std=0.02)
        # haven't found any mathematically elaborate explanation of why this initialization makes sense
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        # TODO REVISIT END
        
    def forward(self, inputs):
        # Convert inputs [batch_size, sequence_length, embedding_dim]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim) # [ (batch x seq_len), emb_dim ]

        # Calculate distances - uses broadcasting elegantly!
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
        # perplexity correlates with the diversity of latent code usage given the input
        # keep it not too high (indicating lack of learning), not too low (codebook isn't used to its full)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1])

class GPT2VQVAE(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 num_thoughts=32):
        """
        Initialize GPT2-based VQ-VAE model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the model (default: 768 for GPT2)
            num_embeddings (int): Size of the VQ codebook
            commitment_cost (float): Commitment cost for VQ
            aggregation_hidden_dim (int): Hidden dimension for aggregation MLP
            num_thoughts (int): Number of parallel sequences
        """
        super(GPT2VQVAE, self).__init__()
        
        # Load GPT2 model and configuration
        self.gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_positions=1024, # Adjust based on needs
        )
        
        # Initialize encoder and decoder with GPT2
        self.encoder = GPT2Model(self.gpt2_config)
        self.decoder = GPT2LMHeadModel(self.gpt2_config)
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        
        # Aggregation MLP
        # TODO CONSIDER DIFFERENT OPTIONS! THIS MIGHT BE A VERY STRONG
        # BACK-BONE AND MIGHT CAUSE POSTERIOR COLLAPSE EVEN WITH A VQ-VAE
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(num_thoughts * d_model, aggregation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Using GPT2's default dropout
            nn.Linear(aggregation_hidden_dim, d_model)
        )
        
        # Chain-positional embeddings to differentiate between M sequences
        self.chain_embeddings = nn.Embedding(num_thoughts, d_model)
        # Initialize chain embeddings with small values
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
            torch.Tensor: Aggregated embeddings of shape [batch_size*L, d_model]
        """
        _, M, d_model = memory.shape
        if mode == "linear":
            # reshape to pass through mlp
            memory = memory.view(-1, M*d_model)
            # Pass through MLP for non-linear transformation
            aggregated = self.aggregation_mlp(memory)  # [batch_size*L, num_thoughts*d_model]
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
            
        return aggregated
        
    def encode(self, src, src_mask=None, aggregate_mode="linear"):
        """
        Encodes BATCH * M sequences of length L into BATCH sequences of L latent tokens.
        The encoding can be done causally, in which case the ith latent token is produced only
        looking at up to the ith tokens from the M sequences.

        Args:
            src (torch.Tensor): Input sequences [batch_size, M, L]
            src_mask (torch.Tensor, optional): Attention mask
            aggregate_mode (str): Mode of aggregation
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, indices) 
        """
        # src shape: [batch_size, M, L] where M is num sequences and L is sequence length
        batch_size, M, L = src.shape
        
        # Reshape to process all sequences together
        src = src.view(batch_size * M, L)
        
        # Get GPT2 encoder outputs
        encoder_outputs = self.encoder(
            input_ids=src,
            attention_mask=src_mask,
            use_cache=False,
            return_dict=True
        )
        
        # Get the last hidden state
        memory = encoder_outputs.last_hidden_state
        
        # Reshape to group tokens at same positions across sequences
        memory = memory.view(batch_size, M, L, -1)
        memory = memory.transpose(1, 2)  # [batch_size, L, M, d_model]
        memory = memory.reshape(batch_size * L, M, -1)  # [batch_size*L, M, d_model]
        
        # aggregate the memory content per-prompt into single chains 
        aggregated = self.aggregate(memory, mode=aggregate_mode) # [batch_size*L, d_model]
        
        # Apply VQ
        quantized, vq_loss, perplexity, indices = self.vector_quantizer(aggregated)
        
        # Tile back to obtain the same shape and amount of info
        quantized = quantized.unsqueeze(1).repeat(1, M, 1)  # [batch_size*L, M, d_model]
        
        # Reshape back
        quantized = quantized.view(batch_size, L, M, -1)
        return quantized, vq_loss, perplexity, indices
        
    def decode(self, memory, tgt, tgt_mask=None):
        """
        Decodes using GPT2 decoder with chain-positional embeddings.
        
        Args:
            memory (torch.Tensor): Encoded memory [batch_size, L, M, d_model]
            tgt (torch.Tensor): Target sequences
            tgt_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Decoded output logits
        """
        batch_size, L, M, _ = memory.shape
        
        # Prepare target
        tgt = tgt.view(batch_size * M, -1)
        
        # Reshape memory for decoder
        memory = memory.transpose(1, 2)  # [batch_size, M, L, d_model]
        memory = memory.reshape(batch_size * M, L, -1)
        
        # Add chain-positional embeddings to memory
        # Create chain indices: [0, 1, ..., M-1] repeated batch_size times
        chain_indices = torch.arange(M, device=memory.device).repeat(batch_size)
        # Get chain embeddings and add them to the memory
        chain_emb = self.chain_embeddings(chain_indices).unsqueeze(1)  # [batch_size*M, 1, d_model]
        memory = memory + chain_emb  # Add to all positions in the sequence
        
        # Get GPT2 decoder outputs with language modeling head
        decoder_outputs = self.decoder(
            input_ids=tgt,
            attention_mask=tgt_mask,
            encoder_hidden_states=memory,
            use_cache=False,
            return_dict=True
        )
        
        return decoder_outputs.logits
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the model.
        
        Args:
            src (torch.Tensor): Source sequences [batch_size, M, L]
            tgt (torch.Tensor): Target sequences [batch_size, M, L]
            src_mask (torch.Tensor, optional): Source attention mask
            tgt_mask (torch.Tensor, optional): Target attention mask
            
        Returns:
            tuple: (output, vq_loss, perplexity)
        """
        quantized, vq_loss, perplexity, _ = self.encode(src, src_mask)
        output = self.decode(quantized, tgt, tgt_mask)
        return output, vq_loss, perplexity

def create_mask(size):
    """Create a causal mask for the decoder."""
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

def train_step(model, optimizer, src, tgt, criterion):
    """Single training step."""
    optimizer.zero_grad()
    
    # Create causal mask for the decoder
    tgt_mask = create_mask(tgt.size(-1)).to(tgt.device)
    
    # Forward pass
    output, vq_loss, perplexity = model(src, tgt[:, :-1], tgt_mask=tgt_mask)
    
    # Calculate loss
    output = output.view(-1, output.size(-1))
    tgt = tgt[:, 1:].reshape(-1)
    reconstruction_loss = criterion(output, tgt)
    
    # Total loss
    loss = reconstruction_loss + vq_loss
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item(), reconstruction_loss.item(), vq_loss.item(), perplexity.item()

if __name__ == "__main__":
    # Example usage
    VOCAB_SIZE = 50257  # GPT2's vocabulary size
    BATCH_SIZE = 4
    NUM_SEQUENCES = 3
    SEQ_LENGTH = 32
    
    # Initialize model
    model = GPT2VQVAE(
        vocab_size=VOCAB_SIZE,
        d_model=768,  # GPT2's default dimension
        num_embeddings=512,
        num_thoughts=NUM_SEQUENCES
    )
    
    # Create example data
    src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NUM_SEQUENCES, SEQ_LENGTH))
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NUM_SEQUENCES, SEQ_LENGTH))
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training step
    loss, rec_loss, vq_loss, perplexity = train_step(model, optimizer, src, tgt, criterion)
    print(f"Loss: {loss:.4f}, Reconstruction Loss: {rec_loss:.4f}, VQ Loss: {vq_loss:.4f}, Perplexity: {perplexity:.4f}")