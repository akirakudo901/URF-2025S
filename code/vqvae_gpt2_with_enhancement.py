# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import functions and classes from the base file
from vqvae_gpt2 import (
    compute_perplexity,
    GPT2VQVAE
)

logger = logging.getLogger(__name__)

class EnhancedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 diversity_gamma: float = 0.1, reset_threshold: float = 0.1,
                 reset_frequency: int = 1000, use_ema: bool = True):
        """
        Enhanced Vector quantizer initialization with EMA updates and diversity mechanisms.

        :param int num_embeddings: Codebook size.
        :param int embedding_dim: Dimension of each embedding.
        :param float commitment_cost: Commitment loss multiplier, defaults to 0.25
        :param float ema_decay: EMA decay rate for codebook updates, defaults to 0.99
        :param float diversity_gamma: Weight for diversity-promoting loss, defaults to 0.1
        :param float reset_threshold: Threshold for codebook reset (usage ratio), defaults to 0.1
        :param int reset_frequency: Frequency of codebook reset checks, defaults to 1000
        :param bool use_ema: Whether to use EMA updates, defaults to True
        """
        super(EnhancedVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.diversity_gamma = diversity_gamma
        self.reset_threshold = reset_threshold
        self.reset_frequency = reset_frequency
        self.use_ema = use_ema
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Improved initialization
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        # EMA tracking variables
        if self.use_ema:
            self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
            self.register_buffer('_ema_decay', torch.tensor(ema_decay))
        
        # Codebook reset tracking
        self.register_buffer('_reset_counter', torch.tensor(0))
        self.register_buffer('_usage_counts', torch.zeros(num_embeddings))
        
    def _update_ema(self, flat_input, encoding_indices):
        """
        Update EMA statistics for codebook learning.
        
        Args:
            flat_input (torch.Tensor): Flattened input embeddings
            encoding_indices (torch.Tensor): Indices of nearest embeddings
        """
        if not self.use_ema:
            return
            
        # Create one-hot encoding for EMA updates
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, 
                               device=flat_input.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Update EMA cluster sizes
        ema_decay_val = self._ema_decay.item()
        self._ema_cluster_size.mul_(ema_decay_val).add_( # [num_embeddings]
            (1 - ema_decay_val) * torch.sum(encodings, 0)
        )
        
        # Update EMA weights
        n = torch.sum(self._ema_cluster_size).item()
        # below, matmul is the sum of all latents mapped to each code
        self._ema_w.mul_(ema_decay_val).add_(                           # [num_embeddings x emb_dim]
            (1 - ema_decay_val) * torch.matmul(encodings.T, flat_input) 
        )
        
        # Normalize EMA weights
        if n > 0:
            # Add small epsilon to prevent division by zero
            cluster_sizes_safe = self._ema_cluster_size + 1e-8
            self.embedding.weight.data.copy_(self._ema_w / cluster_sizes_safe.unsqueeze(1))
    
    def _compute_diversity_loss(self, encoding_indices):
        """
        Compute diversity-promoting loss to encourage uniform codebook usage.
        
        Args:
            encoding_indices (torch.Tensor): Indices of nearest embeddings
            
        Returns:
            torch.Tensor: Diversity loss value
        """
        # Count usage of each embedding
        usage_counts = torch.bincount(encoding_indices, minlength=self.num_embeddings)
        total_usage = usage_counts.sum()
        
        # Handle case where no embeddings are used
        if total_usage == 0:
            return torch.tensor(0.0, device=encoding_indices.device)
        
        usage_probs = usage_counts.float() / total_usage
        
        # Target uniform distribution
        target_probs = torch.ones_like(usage_probs) / self.num_embeddings
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        usage_probs_safe = usage_probs + epsilon
        target_probs_safe = target_probs + epsilon
        
        # KL divergence from uniform distribution
        kl_div = torch.sum(usage_probs_safe * torch.log(usage_probs_safe / target_probs_safe))
        
        # Clip the loss to prevent explosion
        kl_div = torch.clamp(kl_div, max=100.0)
        
        return kl_div
    
    def _compute_regularization_loss(self, embedding_weight):
        """
        Compute regularization loss to prevent embedding weights from growing too large.
        
        Args:
            embedding_weight (torch.Tensor): Current embedding weights
            
        Returns:
            torch.Tensor: Regularization loss value
        """
        # L2 regularization on embedding weights
        l2_reg = torch.norm(embedding_weight, p=2, dim=1).mean()
        
        # Orthogonality regularization to encourage diverse embeddings
        normalized_embeddings = F.normalize(embedding_weight, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(self.num_embeddings, device=embedding_weight.device)
        off_diagonal_similarities = similarity_matrix * (1 - mask)
        
        # Penalize high similarities (encourage orthogonality)
        orthogonality_reg = torch.mean(torch.abs(off_diagonal_similarities))
        
        # Clip both components to prevent explosion
        l2_reg = torch.clamp(l2_reg, max=10.0)
        orthogonality_reg = torch.clamp(orthogonality_reg, max=10.0)
        
        return l2_reg + orthogonality_reg
    
    def _check_codebook_reset(self, encoding_indices):
        """
        Check if codebook reset is needed based on usage statistics.
        
        Args:
            encoding_indices (torch.Tensor): Indices of nearest embeddings
            
        Returns:
            bool: True if reset is needed, False otherwise
        """
        # Check reset condition every reset_frequency steps
        if self._reset_counter % self.reset_frequency == 0:
            total_usage = self._usage_counts.sum().item()
            if total_usage > 0:
                usage_ratio = self._usage_counts / total_usage
                unused_ratio = (usage_ratio < self.reset_threshold).float().mean().item()
                
                if unused_ratio > 0.5:  # If more than 50% of codes are underused
                    return True
        
        return False
    
    def _reset_codebook(self, flat_input):
        """
        Reset codebook by reinitializing unused embeddings.
        
        Args:
            flat_input (torch.Tensor): Current input embeddings for reference
        """
        print("Codebook reset triggered - reinitializing unused embeddings")
        
        # Identify unused embeddings
        usage_ratio = self._usage_counts / (self._usage_counts.sum().item() + 1e-8)
        unused_mask = usage_ratio < self.reset_threshold
        
        # Reinitialize unused embeddings
        if unused_mask.sum().item() > 0:
            # Use random samples from current input as new embeddings
            num_unused = unused_mask.sum().item()
            if flat_input.shape[0] >= num_unused:
                # Sample random inputs for unused embeddings
                indices = torch.randperm(flat_input.shape[0])[:num_unused]
                new_embeddings = flat_input[indices].float()
            else:
                # Use random initialization if not enough inputs
                new_embeddings = torch.randn(num_unused, self.embedding_dim, 
                                           device=flat_input.device) * 0.02
            
            # Update unused embeddings
            unused_indices = torch.where(unused_mask)[0]
            self.embedding.weight.data[unused_indices] = new_embeddings
            
            # Reset EMA statistics for unused embeddings
            if self.use_ema:
                self._ema_cluster_size.data[unused_indices] = 0
                self._ema_w.data[unused_indices] = 0
        
        # Reset usage counts
        self._usage_counts.zero_()
        
    def forward(self, inputs):
        """
        Forward pass with enhanced features.
        
        Args:
            inputs (torch.Tensor): Input embeddings to quantize
            
        Returns:
            tuple: (quantized, total_loss, perplexity, encoding_indices)
        """

        def compute_loss(quantized, inputs, encoding_indices):
            # Standard VQ-VAE losses
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            q_latent_loss = F.mse_loss(quantized, inputs.detach())
            vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
            
            # Additional losses
            diversity_loss = self._compute_diversity_loss(encoding_indices)
            regularization_loss = self._compute_regularization_loss(self.embedding.weight)
            
            # Combined loss
            total_loss = vq_loss + self.diversity_gamma * diversity_loss + 0.01 * regularization_loss
            
            # Check for NaN and clip if necessary
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss detected. vq_loss: {vq_loss}, diversity_loss: {diversity_loss}, reg_loss: {regularization_loss}")
                # Fall back to just VQ loss if other losses are problematic
                total_loss = vq_loss
            
            # Clip the total loss to prevent explosion
            return torch.clamp(total_loss, max=100.0)

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
        
        # Training-specific updates
        current_usage = torch.bincount(encoding_indices, minlength=self.num_embeddings)
        if self.training:
            # Update EMA statistics
            self._update_ema(flat_input, encoding_indices)
            
            # Update training usage counts and reset counter
            self._usage_counts += current_usage
            self._reset_counter += 1
            
            # Check for codebook reset
            if self._check_codebook_reset(encoding_indices):
                self._reset_codebook(flat_input)
        else:
            # During inference, update separate inference usage counts
            if not hasattr(self, '_inference_usage_counts'):
                self.register_buffer('_inference_usage_counts', 
                                     torch.zeros(self.num_embeddings).to(current_usage.device))
            self._inference_usage_counts += current_usage
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)

        # Loss computation (only during training)
        if self.training:
            total_loss = compute_loss(quantized, inputs, encoding_indices)
        else:
            # During inference, do not track gradients
            with torch.no_grad():
                total_loss = compute_loss(quantized, inputs, encoding_indices)
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        # Perplexity: diversity of latent code usage, keep it mid (high=uniform, no learning, low=not used fully)
        perplexity = compute_perplexity(encoding_indices, "indices")
        
        # Handle case where input_shape has only 1 dimension
        if len(input_shape) == 1:
            indices_shape = input_shape
        else:
            indices_shape = input_shape[:-1]
        
        return quantized, total_loss, perplexity, encoding_indices.view(indices_shape)

    def get_codebook_stats(self):
        """
        Get statistics about codebook usage and health.
        
        Returns:
            dict: Dictionary containing various statistics
        """
        thresholds = [0.005, 0.01, 0.05]  # 0.5%, 1%, 5%
        if self.training:
            # During training, return training statistics
            total_usage = self._usage_counts.sum().item()
            usage_ratios = (self._usage_counts / (total_usage + 1e-8)).clone().detach()
            stats = {
                'total_usage': total_usage,
                'usage_counts': self._usage_counts.clone().detach(),
                'usage_ratio': usage_ratios,
                'unused_codes': (self._usage_counts == 0).sum().item(),
                'unused_ratio': (self._usage_counts == 0).float().mean().item(),
                'reset_counter': self._reset_counter.item(),
            }
            for t in thresholds:
                stats[f'codes_below_{int(t*1000)/10:.1f}_percent'] = (usage_ratios < t).sum().item()
            if self.use_ema:
                stats.update({
                    'ema_cluster_sizes': self._ema_cluster_size.clone().detach(),
                    'ema_weights_norm': torch.norm(self._ema_w, p=2, dim=1).mean().item(),
                })
        else:
            # During inference, return inference statistics (read-only)
            if not hasattr(self, '_inference_usage_counts'):
                self.register_buffer('_inference_usage_counts', torch.zeros(self.num_embeddings))
            inference_usage = self._inference_usage_counts
            total_inference_usage = inference_usage.sum().item()
            usage_ratios = (inference_usage / (total_inference_usage + 1e-8)).clone().detach()
            stats = {
                'total_usage': total_inference_usage,
                'usage_counts': inference_usage.clone().detach(),
                'usage_ratio': usage_ratios,
                'unused_codes': (inference_usage == 0).sum().item(),
                'unused_ratio': (inference_usage == 0).float().mean().item(),
                'reset_counter': 0,  # Not tracked during inference
            }
            for t in thresholds:
                stats[f'codes_below_{int(t*1000)/10:.1f}_percent'] = (usage_ratios < t).sum().item()
            if self.use_ema:
                # Return current EMA stats without modifying them
                stats.update({
                    'ema_cluster_sizes': self._ema_cluster_size.clone().detach(),
                    'ema_weights_norm': torch.norm(self._ema_w, p=2, dim=1).mean().item(),
                })
        
        return stats
    
    def get_embedding_diversity(self):
        """
        Compute diversity metrics for the embedding table.
        
        Returns:
            dict: Dictionary containing diversity metrics
        """
        embedding_weight = self.embedding.weight
        
        # Compute pairwise distances between embeddings
        normalized_embeddings = F.normalize(embedding_weight, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Remove diagonal
        mask = torch.eye(self.num_embeddings, device=embedding_weight.device)
        off_diagonal_similarities = similarity_matrix * (1 - mask)
        
        diversity_metrics = {
            'mean_similarity': off_diagonal_similarities.mean().item(),
            'max_similarity': off_diagonal_similarities.max().item(),
            'min_similarity': off_diagonal_similarities.min().item(),
            'similarity_std': off_diagonal_similarities.std().item(),
            'embedding_norm_mean': torch.norm(embedding_weight, p=2, dim=1).mean().item(),
            'embedding_norm_std': torch.norm(embedding_weight, p=2, dim=1).std().item(),
        }
        
        return diversity_metrics
    
    def manual_reset(self, reset_strategy='random'):
        """
        Manually reset the codebook using different strategies.
        
        Args:
            reset_strategy (str): Strategy for reset - 'random', 'uniform', or 'kmeans'
        """
        if not self.training:
            print("Warning: Manual reset disabled during inference mode")
            return
            
        print(f"Manual codebook reset using strategy: {reset_strategy}")
        
        if reset_strategy == 'random':
            # Random initialization
            self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        elif reset_strategy == 'uniform':
            # Uniform initialization in unit sphere
            random_vectors = torch.randn(self.num_embeddings, self.embedding_dim)
            normalized_vectors = F.normalize(random_vectors, p=2, dim=1)
            self.embedding.weight.data.copy_(normalized_vectors * 0.02)
        elif reset_strategy == 'kmeans':
            # This would require input data, so we'll use random for now
            print("K-means reset requires input data - using random instead")
            self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown reset strategy: {reset_strategy}")
        
        # Reset EMA statistics
        if self.use_ema:
            self._ema_cluster_size.zero_()
            self._ema_w.zero_()
        
        # Reset usage counts
        self._usage_counts.zero_()
        self._reset_counter.zero_()
        
        print("Codebook reset completed")
    
    def set_ema_decay(self, new_decay):
        """
        Update the EMA decay rate.
        
        Args:
            new_decay (float): New EMA decay rate
        """
        if not self.training:
            print("Warning: EMA decay update disabled during inference mode")
            return
            
        if not 0 < new_decay < 1:
            raise ValueError("EMA decay must be between 0 and 1")
        
        self.ema_decay = new_decay
        self._ema_decay.data.fill_(new_decay)
        print(f"EMA decay updated to: {new_decay}")
    
    def disable_ema(self):
        """Disable EMA updates."""
        self.use_ema = False
        print("EMA updates disabled")
    
    def enable_ema(self, decay=None):
        """
        Enable EMA updates.
        
        Args:
            decay (float, optional): New EMA decay rate
        """
        self.use_ema = True
        if decay is not None:
            self.set_ema_decay(decay)
        print("EMA updates enabled")

    def is_training_mode(self):
        """
        Check if the model is in training mode.
        
        Returns:
            bool: True if the model is in training mode, False otherwise
        """
        return self.training
    
    def reset_inference_stats(self):
        """
        Reset inference usage statistics.
        This is useful when starting a new inference session.
        """
        if hasattr(self, '_inference_usage_counts'):
            self._inference_usage_counts.zero_()
        print("Inference usage statistics reset")
    
    def get_inference_safety_info(self):
        """
        Get information about which features are disabled during inference for safety.
        
        Returns:
            dict: Dictionary containing information about inference safety features
        """
        return {
            'disabled_features': [
                'EMA updates (_update_ema)',
                'Codebook reset (_check_codebook_reset, _reset_codebook)',
                'Training usage statistics tracking (_usage_counts updates)',
                'Reset counter updates (_reset_counter)',
                'Manual codebook reset (manual_reset)',
                'EMA parameter changes (set_ema_decay, disable_ema, enable_ema)',
            ],
            'enabled_features': [
                'Vector quantization (encoding and decoding)',
                'Loss computation (without gradient accumulation)'
                'Distance calculation',
                'Perplexity computation',
                'Inference usage statistics tracking (_inference_usage_counts)',
                'Statistics retrieval (get_codebook_stats, get_embedding_diversity)',
                'Read-only EMA statistics access'
            ],
            'safety_mechanisms': [
                'Training mode checks in all training-specific methods',
                'Separate inference usage tracking (does not affect training stats)',
                'Warning messages when training features are accessed during inference',
                'No side effects during inference forward pass'
            ]
        }


class EnhancedGPT2VQVAE(GPT2VQVAE):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 num_thoughts=32, n_positions=1024, 
                 use_pretrained_encoder=True, use_pretrained_decoder=True,
                 pretrained_model_name="gpt2",
                 # Vector Quantizer specific parameters
                 ema_decay=0.99, diversity_gamma=0.1, reset_threshold=0.1,
                 reset_frequency=1000, use_ema=True,
                 # Unified parameters (applied to both encoder and decoder if specified)
                 n_layer=12, n_head=12, n_inner=None, dropout=0.1, activation_function="gelu",
                 # Encoder-specific parameters (take precedence over unified if specified)
                 encoder_n_layer=None, encoder_n_head=None, encoder_n_inner=None,
                 encoder_dropout=None, encoder_activation_function=None,
                 # Decoder-specific parameters (take precedence over unified if specified)
                 decoder_n_layer=None, decoder_n_head=None, decoder_n_inner=None,
                 decoder_dropout=None, decoder_activation_function=None):
        """
        Enhanced GPT2-based VQ-VAE model that uses GPT2 as both encoder and decoder.
        
        Args:
            # Shared parameters (must be the same for both encoder and decoder)
            vocab_size (int): Vocabulary size
            d_model (int): Model dimension (default: 768 for GPT2)
            n_positions (int): Maximum sequence length for GPT2 (default: 1024)
            
            # VQ-VAE specific parameters
            num_embeddings (int): VQ codebook size
            commitment_cost (float): VQ commitment cost
            aggregation_hidden_dim (int): Aggregation MLP hidden dimension
            num_thoughts (int): Number of parallel sequences
            
            # Pretrained model settings
            use_pretrained_encoder (bool): Whether to load pretrained weights for encoder
            use_pretrained_decoder (bool): Whether to load pretrained weights for decoder
            pretrained_model_name (str): Name of pretrained model to load (default: "gpt2")
            
            # Vector Quantizer specific parameters
            ema_decay (float): EMA decay rate for codebook updates
            diversity_gamma (float): Weight for diversity-promoting loss
            reset_threshold (float): Threshold for codebook reset (usage ratio)
            reset_frequency (int): Frequency of codebook reset checks
            use_ema (bool): Whether to use EMA updates
            
            # Unified parameters (applied to both encoder and decoder)
            n_layer (int): Number of hidden layers for both encoder and decoder (default: 12)
            n_head (int): Number of attention heads for both encoder and decoder (default: 12)
            n_inner (int, optional): Dimensionality of inner feed-forward layers for both encoder and decoder
            dropout (float): Dropout probability for both encoder and decoder (default: 0.1)
            activation_function (str): Activation function for both encoder and decoder (default: "gelu")
            
            # Encoder-specific parameters (take precedence over unified if specified)
            encoder_n_layer (int, optional): Number of hidden layers in the encoder
            encoder_n_head (int, optional): Number of attention heads for encoder
            encoder_n_inner (int, optional): Dimensionality of encoder inner feed-forward layers
            encoder_dropout (float, optional): Dropout probability for encoder
            encoder_activation_function (str, optional): Activation function for encoder
            
            # Decoder-specific parameters (take precedence over unified if specified)
            decoder_n_layer (int, optional): Number of hidden layers in the decoder
            decoder_n_head (int, optional): Number of attention heads for decoder
            decoder_n_inner (int, optional): Dimensionality of decoder inner feed-forward layers
            decoder_dropout (float, optional): Dropout probability for decoder
            decoder_activation_function (str, optional): Activation function for decoder
        """
        # Call parent constructor with all the base parameters
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            aggregation_hidden_dim=aggregation_hidden_dim,
            num_thoughts=num_thoughts,
            n_positions=n_positions,
            use_pretrained_encoder=use_pretrained_encoder,
            use_pretrained_decoder=use_pretrained_decoder,
            pretrained_model_name=pretrained_model_name,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            dropout=dropout,
            activation_function=activation_function,
            encoder_n_layer=encoder_n_layer,
            encoder_n_head=encoder_n_head,
            encoder_n_inner=encoder_n_inner,
            encoder_dropout=encoder_dropout,
            encoder_activation_function=encoder_activation_function,
            decoder_n_layer=decoder_n_layer,
            decoder_n_head=decoder_n_head,
            decoder_n_inner=decoder_n_inner,
            decoder_dropout=decoder_dropout,
            decoder_activation_function=decoder_activation_function
        )
        
        # Replace the vector quantizer with the enhanced version
        self.vector_quantizer = EnhancedVectorQuantizer(
            num_embeddings, d_model, commitment_cost, ema_decay, 
            diversity_gamma, reset_threshold, reset_frequency, use_ema
        )
        
        # Store enhanced configuration for checkpoint validation
        self._enhanced_config = {
            'ema_decay': ema_decay,
            'diversity_gamma': diversity_gamma,
            'reset_threshold': reset_threshold,
            'reset_frequency': reset_frequency,
            'use_ema': use_ema,
        }
    
    def get_vector_quantizer_stats(self):
        """
        Get statistics from the enhanced vector quantizer.
        
        Returns:
            dict: Dictionary containing vector quantizer statistics
        """
        return self.vector_quantizer.get_codebook_stats()
    
    def get_embedding_diversity(self):
        """
        Get diversity metrics from the enhanced vector quantizer.
        
        Returns:
            dict: Dictionary containing diversity metrics
        """
        return self.vector_quantizer.get_embedding_diversity()
    
    def manual_codebook_reset(self, reset_strategy='random'):
        """
        Manually reset the codebook using the enhanced vector quantizer.
        
        Args:
            reset_strategy (str): Strategy for reset - 'random', 'uniform', or 'kmeans'
        """
        self.vector_quantizer.manual_reset(reset_strategy)
    
    def set_ema_decay(self, new_decay):
        """
        Update the EMA decay rate in the enhanced vector quantizer.
        
        Args:
            new_decay (float): New EMA decay rate
        """
        self.vector_quantizer.set_ema_decay(new_decay)
    
    def disable_ema(self):
        """Disable EMA updates in the enhanced vector quantizer."""
        self.vector_quantizer.disable_ema()
    
    def enable_ema(self, decay=None):
        """
        Enable EMA updates in the enhanced vector quantizer.
        
        Args:
            decay (float, optional): New EMA decay rate
        """
        self.vector_quantizer.enable_ema(decay)