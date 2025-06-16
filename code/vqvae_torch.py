# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/13

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerVQVAE(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 num_embeddings=512, commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 num_thoughts=32):
        super(TransformerVQVAE, self).__init__()
        
        # Token embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        
        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Aggregation MLP
        # TODO CONSIDER DIFFERENT OPTIONS! THIS MIGHT BE A VERY STRONG
        # BACK-BONE AND MIGHT CAUSE POSTERIOR COLLAPSE EVEN WITH A VQ-VAE
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(num_thoughts * d_model, aggregation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aggregation_hidden_dim, d_model)
        )
        
        self.d_model = d_model
        self.num_thoughts = num_thoughts

    def aggregate(self, memory, mode="linear"):
        """
        Aggregates same-position embeddings from chains sharing the same prompt, compressing
        it into one embedding of the same dimensions.
        e.g. three same-position embeddings of dimension 512 will be compressed into
             one 512-dimensional latent embedding

        :param torch.Tensor memory: The embeddings obtained from the encoder, of shape:
        [batch_size*L, num_thoughts, d_model] where L is the length of sequences.
        :param str mode: Mode of aggregation across same position & input tokens, 
        defaults to "linear"
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
        
    def encode(self, src, src_mask=None, is_causal=True, aggregate_mode="linear"):
        """
        Encodes BATCH * M sequences of length L into BATCH sequences of L latent tokens.
        The encoding can be done causally, in which case the ith latent token is produced only
        looking at up to the ith tokens from the M sequences.

        :param torch.Tensor src: The source data to be encoded. 
        [batch_size, M, L] where M is num sequences and L is sequence length.
        :param torch.Tensor src_mask: Optional mask applied to src, defaults to None
        :param bool is_causal: Whether to apply the causal attention mask for the encoders, 
        defaults to True
        :param str aggregate_mode: Mode of aggregation of tokens from different chains.
        Defaults to linear, using an MLP for mapping
        """
        # src shape: [batch_size, M, L] where M is num sequences and L is sequence length
        batch_size, M, L = src.shape
        
        # Reshape to process all sequences together
        src = src.view(batch_size * M, L)
        
        # Embed tokens and add positional encoding
        src = self.token_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        # Transformer encoding
        memory = self.transformer_encoder(src, src_mask, is_causal)
        
        # Reshape to group tokens at same positions across sequences
        memory = memory.view(batch_size, M, L, -1)
        memory = memory.transpose(1, 2)  # [batch_size, L, M, d_model]
        memory = memory.reshape(batch_size * L, M, -1)  # [batch_size*L, M, d_model]
        
        # aggregate the memory content per-prompt into single chains 
        aggregated = self.aggregate(memory, mode=aggregate_mode) # [batch_size*L, d_model]
        
        # Apply VQ to each position's tokens across sequences
        quantized, vq_loss, perplexity, indices = self.vector_quantizer(aggregated)

        # then tile back to obtain the same shape and amount of info for the quantized result
        quantized = quantized.unsqueeze(1).repeat(1, M, 1)  # [batch_size*L, M, d_model]
        
        # Reshape back
        quantized = quantized.view(batch_size, L, M, -1)
        return quantized, vq_loss, perplexity, indices
        
    def decode(self, memory, tgt, tgt_mask=None):
        # memory shape: [batch_size, L, M, d_model]
        batch_size, L, M, _ = memory.shape
        
        # Prepare target
        tgt = tgt.view(batch_size * M, -1)
        tgt = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Reshape memory for decoder
        memory = memory.transpose(1, 2)  # [batch_size, M, L, d_model]
        memory = memory.reshape(batch_size * M, L, -1)

        # add a "chain-positional encoding" indicating which chain we want the transformer to decode into
        # also condition the decoder on the prompt?
        
        
        # Decode
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        output = self.output_layer(output)
        return output
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        quantized, vq_loss, perplexity, indices = self.encode(src, src_mask)
        output = self.decode(quantized, tgt, tgt_mask)
        return output, vq_loss, perplexity

def create_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

def train_step(model, optimizer, src, tgt, criterion):
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

    
class SequenceAwareTransformerVQVAE(TransformerVQVAE):
    # A model with sequence-aware components
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learnable embeddings for sequence positions
        self.sequence_embeddings = nn.Embedding(kwargs.get('num_sequences', 3), self.d_model)
        
    def decode(self, memory, tgt, tgt_mask=None):
        # memory shape: [batch_size, L, M, d_model]
        batch_size, L, M, _ = memory.shape
        
        # Prepare target
        tgt = tgt.view(batch_size * M, -1)
        tgt = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Add sequence embeddings
        sequence_indices = torch.arange(M, device=tgt.device).repeat(batch_size)
        sequence_emb = self.sequence_embeddings(sequence_indices).unsqueeze(1)
        tgt = tgt + sequence_emb
        
        # Reshape memory for decoder
        memory = memory.transpose(1, 2)  # [batch_size, M, L, d_model]
        memory = memory.reshape(batch_size * M, L, -1)
        
        # Decode
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        output = self.output_layer(output)
        return output

if __name__ == "__main__":
    VOCAB_SIZE = 1024
    NUM_EPOCHS = 1000

    dataloader = None

    # Initialize model
    model = TransformerVQVAE(
        vocab_size=VOCAB_SIZE,
        d_model=512,  # embedding dimension
        nhead=8,      # number of attention heads
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_embeddings=512  # size of the VQ codebook
    )

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            src = batch['input']  # shape: [batch_size, M, L]
            tgt = batch['target']  # shape: [batch_size, M, L]
            
            loss, rec_loss, vq_loss, perplexity = train_step(
                model, optimizer, src, tgt, criterion
            )