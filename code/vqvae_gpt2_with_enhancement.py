# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from sklearn.cluster import KMeans
from collections import deque

from transformers.cache_utils import DynamicCache

# Import functions and classes from the base file
from vqvae_gpt2 import (
    compute_perplexity,
    GPT2VQVAE
)

logger = logging.getLogger(__name__)

class ReservoirSampler:
    """
    Reservoir sampling for efficient data-dependent initialization.
    Based on the paper's recommendation for handling large datasets.
    """
    def __init__(self, reservoir_size=10000, embedding_size=None):
        self.reservoir_size = reservoir_size
        self.embedding_size = embedding_size
        self.reservoir = deque(maxlen=reservoir_size)
        self.count = 0
    
    def _validate_and_flatten_sample(self, sample):
        """
        Validate sample dimensions and flatten if necessary.
        
        Args:
            sample (torch.Tensor): Input sample
            
        Returns:
            torch.Tensor: Flattened sample with shape [..., embedding_size]
        """
        if self.embedding_size is not None:
            if sample.shape[-1] != self.embedding_size:
                raise ValueError(f"Sample's last dimension {sample.shape[-1]} does not match expected embedding_size {self.embedding_size}")
        
        # Flatten to 2D if more than 2 dimensions, keeping the last dimension
        if sample.dim() > 2:
            # Reshape to [..., embedding_size] where ... represents all dimensions except the last
            sample = sample.view(-1, sample.shape[-1])
        
        return sample
    
    def add_sample(self, sample):
        """
        Add a sample to the reservoir using reservoir sampling.
        
        Args:
            sample (torch.Tensor): Sample to add to reservoir
        """
        sample = self._validate_and_flatten_sample(sample)
        
        self.count += 1
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(sample)
        else:
            # Reservoir sampling: replace with probability reservoir_size/count
            if torch.rand(1).item() < self.reservoir_size / self.count:
                idx = torch.randint(0, self.reservoir_size, (1,)).item()
                self.reservoir[idx] = sample
    
    def add_samples(self, samples):
        """
        Add multiple samples to the reservoir efficiently.
        
        Args:
            samples (torch.Tensor): Batch of samples to add [batch_size, ...]
        """
        if samples.dim() == 0:
            # Single sample case
            self.add_sample(samples)
            return
        
        # Validate and flatten all samples
        samples = self._validate_and_flatten_sample(samples)
        
        # If samples is 1D, treat as single sample
        if samples.dim() == 1:
            self.add_sample(samples)
            return
        
        batch_size = samples.shape[0]
        
        # First, fill the reservoir if it's not full
        remaining_capacity = max(0, self.reservoir_size - len(self.reservoir))
        samples_to_fill = min(remaining_capacity, batch_size)
        
        if samples_to_fill > 0:
            # Add samples to fill the reservoir
            for i in range(samples_to_fill):
                self.reservoir.append(samples[i])
            self.count += samples_to_fill
            samples = samples[samples_to_fill:]  # Remove used samples
            batch_size -= samples_to_fill
        
        # If we still have samples and reservoir is full, use reservoir sampling
        if batch_size > 0 and len(self.reservoir) >= self.reservoir_size:
            # Generate all random numbers at once for efficiency
            random_values = torch.rand(batch_size)
            reservoir_ratio = self.reservoir_size / (self.count + torch.arange(batch_size, dtype=torch.float))
            
            # Find samples that should replace existing ones
            replace_mask = random_values < reservoir_ratio
            
            # Generate random indices for replacement
            replace_indices = torch.randint(0, self.reservoir_size, (batch_size,))
            
            # Apply replacements
            for i, (should_replace, idx) in enumerate(zip(replace_mask, replace_indices)):
                if should_replace:
                    self.reservoir[idx] = samples[i]
            
            self.count += batch_size
    
    def get_samples(self, num_samples=None, shuffle=True):
        """
        Get samples from the reservoir.
        
        Args:
            num_samples (int, optional): Number of samples to return. If None, returns all samples.
            shuffle (bool): Whether to shuffle the samples before returning. If False, returns first num_samples.
            
        Returns:
            torch.Tensor: Requested samples from reservoir, or None if reservoir is empty
        """
        if num_samples is None:
            num_samples = len(self.reservoir)
        if len(self.reservoir) == 0:
            return None
        
        # Convert to tensor
        samples = torch.stack(list(self.reservoir))
        if num_samples > len(samples):
            num_samples = len(samples)
        
        if shuffle:
            # Return shuffled samples
            indices = torch.randperm(len(samples))[:num_samples]
            return samples[indices]
        else:
            # Return first num_samples without shuffling
            return samples[:num_samples]

class EnhancedVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 diversity_gamma: float = 0.1, reset_threshold: float = 0.1,
                 reset_frequency: int = 1000, use_ema: bool = True,
                 max_reset_steps: Optional[int] = None, reservoir_size: int = 10000,
                 reset_strategy: str = 'partial', use_batch_norm: bool = True):
        """
        Enhanced Vector quantizer initialization with EMA updates and diversity mechanisms.
        
        Args:
            num_embeddings: Number of embeddings in the codebook
            embedding_dim: Dimension of each embedding
            commitment_cost: Weight for the commitment loss
            ema_decay: Decay rate for EMA updates
            diversity_gamma: Weight for diversity-promoting loss
            reset_threshold: Threshold for triggering codebook reset (usage ratio)
            reset_frequency: Frequency of checking for codebook reset
            use_ema: Whether to use EMA updates
            max_reset_steps: Maximum training steps during which resets are allowed (None = no limit)
            reservoir_size: Size of reservoir for data-dependent initialization
            reset_strategy: Strategy for automatic resets - 'partial' (reset unused codes) or 'full' (reset entire codebook)
            use_batch_norm: Whether to use batch normalization before vector quantization
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.diversity_gamma = diversity_gamma
        self.reset_threshold = reset_threshold
        self.reset_frequency = reset_frequency
        self.use_ema = use_ema
        self.max_reset_steps = max_reset_steps
        self.reset_strategy = reset_strategy
        self.use_batch_norm = use_batch_norm
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Improved initialization
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        # EMA parameters
        if self.use_ema:
            self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
            self.register_buffer('_ema_decay', torch.tensor(ema_decay))
        
        # Usage tracking
        self.register_buffer('_usage_counts', torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer('_reset_counter', torch.zeros(1, dtype=torch.long))
        
        # Separate inference usage tracking
        self.register_buffer('_inference_usage_counts', torch.zeros(num_embeddings, dtype=torch.long))
        
        # Training step tracking
        self.register_buffer('_current_step', torch.zeros(1, dtype=torch.long))
        
        # Reservoir sampler for data-dependent initialization
        self.reservoir_sampler = ReservoirSampler(reservoir_size, embedding_dim)
        
        # Batch normalization (optional)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01, eps=1e-5)
        else:
            self.batch_norm = None
    
    def _perform_kmeans_clustering(self, samples, num_clusters, device):
        """
        Perform K-means++ clustering on samples and return centroids.
        
        Args:
            samples (torch.Tensor): Input samples for clustering
            num_clusters (int): Number of clusters to create
            device (torch.device): Device to place centroids on
            
        Returns:
            torch.Tensor: Cluster centroids
        """
        # Flatten samples if needed
        flat_samples = samples.view(-1, self.embedding_dim)
        
        # Use K-means++ for clustering
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', 
                       n_init="auto", random_state=42)
        
        # Fit K-means and get centroids
        kmeans.fit(flat_samples.numpy())
        centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
        
        return centroids
    
    def _get_embeddings_from_reservoir(self, num_embeddings, device, context="reset"):
        """
        Get embeddings by performing K-means++ clustering on reservoir samples.
        Falls back to random initialization if not enough samples.
        
        Args:
            num_embeddings (int): Number of embeddings to generate
            device (torch.device): Device to place embeddings on
            context (str): Context for logging (e.g., "reset", "initialization")
            
        Returns:
            torch.Tensor: Generated embeddings
        """
        # Get samples from reservoir - use all samples, no need to shuffle since we do KMeans
        reservoir_samples = self.reservoir_sampler.get_samples(num_samples=None, shuffle=False)
        
        if reservoir_samples is not None and len(reservoir_samples) >= num_embeddings:
            # Use K-means++ clustering for data-dependent initialization
            embeddings = self._perform_kmeans_clustering(reservoir_samples, num_embeddings, device)
            print(f"Data-dependent {context} completed using {len(reservoir_samples)} reservoir samples")
        else:
            # Fallback to random initialization if not enough samples
            embeddings = torch.randn(num_embeddings, self.embedding_dim, device=device) * 0.02
            print(f"Warning: Not enough reservoir samples for {context}, using random initialization for {num_embeddings} embeddings")
        
        return embeddings
        
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
    
    def _check_codebook_reset(self):
        """
        Check if codebook reset is needed based on usage statistics.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        # Only allow resets for the initial portion of training if max_reset_steps is set
        if self.max_reset_steps is not None and self._current_step > self.max_reset_steps:
            return False
        return self._reset_counter % self.reset_frequency == 0
    
    def _reset_codebook(self, device, reset_strategy='partial'):
        """
        Reset codebook using data-dependent K-means++ clustering with reservoir samples.
        
        Args:
            device (torch.device): Device to use for tensor operations
            reset_strategy (str): Reset strategy - 'partial' (reset unused codes) or 'full' (reset entire codebook)
        """
        if reset_strategy == 'partial':
            print("\nCodebook reset triggered - performing partial reset of unused codes")
            
            # Identify unused embeddings
            usage_ratio = self._usage_counts / (self._usage_counts.sum().item() + 1e-8)
            unused_mask = usage_ratio < self.reset_threshold
            num_unused = unused_mask.sum().item()
            
            if num_unused == 0:
                print("No unused codes found for partial reset")
                return
            
            # Get embeddings for unused codes using reservoir samples
            unused_indices = torch.where(unused_mask)[0]
            new_embeddings = self._get_embeddings_from_reservoir(num_unused, device, "partial reset")
            
            # Update only unused embeddings
            self.embedding.weight.data[unused_indices] = new_embeddings
            
            # Reset EMA statistics for unused embeddings only
            if self.use_ema:
                self._ema_cluster_size.data[unused_indices] = 0
                self._ema_w.data[unused_indices] = 0
            
            print(f"Partial reset completed: {num_unused} unused codes reinitialized")
                
        elif reset_strategy == 'full':
            print("\nCodebook reset triggered - performing full data-dependent re-initialization")
            
            # Get embeddings for entire codebook using reservoir samples
            new_embeddings = self._get_embeddings_from_reservoir(self.num_embeddings, device, "full reset")
            
            # Update embedding weights with new embeddings
            self.embedding.weight.data.copy_(new_embeddings)
            
            # Reset EMA statistics for entire codebook
            if self.use_ema:
                self._ema_cluster_size.zero_()
                self._ema_w.zero_()
        else:
            raise ValueError(f"Unknown reset strategy: {reset_strategy}. Use 'partial' or 'full'")
        
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

        # Batch normalization (optional)
        if self.use_batch_norm and self.batch_norm is not None:
            normalized_inputs = self.batch_norm(flat_input)
        else:
            normalized_inputs = flat_input
        # Calculate distances
        distances = (torch.sum(normalized_inputs**2, dim=1, keepdim=True)           # [ (batch x seq_len), emb_num ]    
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(normalized_inputs, self.embedding.weight.T))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1) # [ (batch x seq_len) ]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,  # [ (batch x seq_len), emb_num ]
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Training-specific updates
        current_usage = torch.bincount(encoding_indices, minlength=self.num_embeddings)
        if self.training:
            # Add to reservoir for future re-initialization
            self.reservoir_sampler.add_samples(inputs.detach().clone().cpu())
            
            # Update EMA statistics
            self._update_ema(normalized_inputs, encoding_indices)
            
            # Update training usage counts and reset counter
            self._usage_counts += current_usage
            self._reset_counter += 1
            self._current_step += 1
            
            # Check for codebook reset
            if self._check_codebook_reset():
                self._reset_codebook(normalized_inputs.device, self.reset_strategy)
        else:
            # During inference, update separate inference usage counts
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
        
        # Add reservoir information
        stats['reservoir_size'] = len(self.reservoir_sampler.reservoir)
        stats['reservoir_count'] = self.reservoir_sampler.count
        stats['reset_strategy'] = self.reset_strategy
        stats['batch_norm_enabled'] = self.is_batch_norm_enabled()
        
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
            # Use reservoir samples for K-means reset
            new_embeddings = self._get_embeddings_from_reservoir(self.num_embeddings, self.embedding.weight.device, "manual reset")
            self.embedding.weight.data.copy_(new_embeddings)
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
                'Loss computation (without gradient accumulation)',
                'Distance calculation',
                'Perplexity computation',
                'Inference usage statistics tracking (_inference_usage_counts)',
                'Statistics retrieval (get_codebook_stats, get_embedding_diversity)',
                'Read-only EMA statistics access',
                'Batch normalization (if enabled)'
            ],
            'safety_mechanisms': [
                'Training mode checks in all training-specific methods',
                'Separate inference usage tracking (does not affect training stats)',
                'Warning messages when training features are accessed during inference',
                'No side effects during inference forward pass'
            ]
        }

    def set_current_step(self, current_step):
        """Set the current training step for reset timing control."""
        self._current_step[0] = current_step
    
    def enable_batch_norm(self):
        """Enable batch normalization if it was previously disabled."""
        if not self.use_batch_norm:
            self.use_batch_norm = True
            self.batch_norm = nn.BatchNorm1d(self.embedding_dim, momentum=0.01, eps=1e-5)
            print("Batch normalization enabled")
        else:
            print("Batch normalization is already enabled")
    
    def disable_batch_norm(self):
        """Disable batch normalization."""
        if self.use_batch_norm:
            self.use_batch_norm = False
            self.batch_norm = None
            print("Batch normalization disabled")
        else:
            print("Batch normalization is already disabled")
    
    def is_batch_norm_enabled(self):
        """
        Check if batch normalization is enabled.
        
        Returns:
            bool: True if batch normalization is enabled, False otherwise
        """
        return self.use_batch_norm and self.batch_norm is not None


class EnhancedGPT2VQVAE(GPT2VQVAE):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 num_thoughts=32, n_positions=1024, 
                 use_pretrained_encoder=True, use_pretrained_decoder=True,
                 pretrained_model_name="gpt2",
                 # Vector Quantizer specific parameters
                 ema_decay=0.99, diversity_gamma=0.1, reset_threshold=0.1,
                 reset_frequency=1000, use_ema=True, max_reset_steps=None, reservoir_size=10000,
                 reset_strategy='partial', use_batch_norm=False,
                 # Unified parameters (applied to both encoder and decoder if specified)
                 n_layer=12, n_head=12, n_inner=None, dropout=0.1, activation_function="gelu",
                 # Encoder-specific parameters (take precedence over unified if specified)
                 encoder_n_layer=None, encoder_n_head=None, encoder_n_inner=None,
                 encoder_dropout=None, encoder_activation_function=None,
                 # Decoder-specific parameters (take precedence over unified if specified)
                 decoder_n_layer=None, decoder_n_head=None, decoder_n_inner=None,
                 decoder_dropout=None, decoder_activation_function=None, 
                 reset_stop_fraction=None):
        """
        Enhanced GPT2VQVAE with improved vector quantization.
        
        This enhanced version includes:
        - EMA (Exponential Moving Average) updates for codebook learning
        - Diversity-promoting loss to encourage uniform codebook usage
        - Automatic codebook reset mechanisms for unused embeddings
        - Enhanced monitoring and statistics for codebook health
        
        The enhanced codebook training scheme reduces to normal VQ-VAE training when:
        - ema_decay = 0.0 (no EMA updates)
        - diversity_gamma = 0.0 (no diversity loss)
        - reset_threshold = 0.0 (no automatic resets)
        - use_ema = False (EMA disabled)
        """
        # Extract vector quantizer parameters
        vq_params = {
            'num_embeddings': num_embeddings,
            'embedding_dim': d_model,
            'commitment_cost': commitment_cost,
            'ema_decay': ema_decay,
            'diversity_gamma': diversity_gamma,
            'reset_threshold': reset_threshold,
            'reset_frequency': reset_frequency,
            'use_ema': use_ema,
            'max_reset_steps': max_reset_steps,
            'reservoir_size': reservoir_size,
            'reset_strategy': reset_strategy,
            'use_batch_norm': use_batch_norm
        }
        
        # Call parent constructor with remaining parameters
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            aggregation_hidden_dim=aggregation_hidden_dim,
            num_thoughts=num_thoughts,
            n_positions=n_positions,
            use_pretrained_encoder=use_pretrained_encoder,
            use_pretrained_decoder=use_pretrained_decoder,
            pretrained_model_name=pretrained_model_name,
            # Unified parameters
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            dropout=dropout,
            activation_function=activation_function,
            # Encoder-specific parameters
            encoder_n_layer=encoder_n_layer,
            encoder_n_head=encoder_n_head,
            encoder_n_inner=encoder_n_inner,
            encoder_dropout=encoder_dropout,
            encoder_activation_function=encoder_activation_function,
            # Decoder-specific parameters
            decoder_n_layer=decoder_n_layer,
            decoder_n_head=decoder_n_head,
            decoder_n_inner=decoder_n_inner,
            decoder_dropout=decoder_dropout,
            decoder_activation_function=decoder_activation_function
        )
        
        # Replace the vector quantizer with enhanced version
        self.vector_quantizer = EnhancedVectorQuantizer(**vq_params)
        
        # Store enhanced configuration for checkpoint validation
        self._enhanced_config = {
            'ema_decay': ema_decay,
            'diversity_gamma': diversity_gamma,
            'reset_threshold': reset_threshold,
            'reset_frequency': reset_frequency,
            'use_ema': use_ema,
            'reservoir_size': reservoir_size,
            'reset_strategy': reset_strategy,
            'use_batch_norm': use_batch_norm,
        }
    
    def forward(self, prompt, cot_sequences, cot_mask=None, prompt_mask=None, inference=False, quantize_cot_only=True, pad_token_id=50256, no_vq=False):
        """
        Forward pass through the model with optional vector quantization bypass.
        
        Args:
            prompt (torch.Tensor): Prompt sequences [batch_size, K] where K is prompt length
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            cot_mask (torch.Tensor, optional): Chain-of-thought attention mask for padding
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            inference (bool): If True, performs inference without teacher forcing
            quantize_cot_only (bool): If True, only quantize the COT portion of sequences
            pad_token_id (int): Token ID to use for padding when K=0, defaults to 50256
            no_vq (bool): If True, bypass vector quantization and use aggregated embeddings directly
            
        Returns:
            tuple: (output_sequences, output_logits, vq_loss, perplexity, indices)
                - output_sequences: Generated token sequences [batch_size, M, L]
                - output_logits: Token logits for each position [batch_size, M, L, vocab_size]
                - vq_loss: Vector quantization loss (zero if no_vq=True)
                - perplexity: Codebook usage perplexity (zero if no_vq=True)
                - indices: Codebook usage indices [batch_size, L] or [batch_size, K+L] depending on quantize_cot_only (zeros if no_vq=True)
        """
        batch_size, K = prompt.shape
        _, M, L = cot_sequences.shape
        
        # Encode using the enhanced encode method with optional VQ bypass
        quantized, vq_loss, perplexity, indices = self.encode(
            prompt, cot_sequences, 
            prompt_mask, cot_mask, 
            quantize_cot_only=quantize_cot_only,
            no_vq=no_vq
        )
        
        # quantized shape depends on quantize_cot_only:
        if quantize_cot_only:          # if True: [batch_size, L, M, d_model] (only COT positions)
            cot_quantized = quantized
        else:                          # if False: [batch_size, K+L, M, d_model] (all positions)
            cot_quantized = quantized[:, K:, :, :]
        
        if not inference:
            # During training, use teacher forcing with single forward pass to get all logits
            output_logits = self.decode(cot_quantized, prompt, cot_sequences, prompt_mask, cot_mask, pad_token_id) # [batch_size, M, L, vocab_size]
            
            # Get the predicted tokens from logits
            output_sequences = torch.argmax(output_logits, dim=-1)
        else:
            # Initialize tensor to store generation results and logits
            output_sequences = torch.zeros((batch_size, M, L), dtype=torch.long, device=cot_sequences.device)
            output_logits = torch.empty((batch_size, M, L, self.decoder_config.vocab_size), device=cot_sequences.device)
        
            # TODO IF I FIND THE TIME : USE KV-CACHING TO SPEED UP AUTO-REGRESSIVE GENERATION
            # During inference, generate sequence auto-regressively
            for t in range(L):
                current_output = self.decode(cot_quantized, prompt, output_sequences, prompt_mask, cot_mask, pad_token_id)
                
                # Get next token predictions
                output_logits[:, :, t, :] = current_output[:, :, t, :]  # [batch_size, M, L, vocab_size]
                output_sequences[:, :, t] = torch.argmax(output_logits[:, :, t, :], dim=-1)  # [batch_size, M, L]
        
        return output_sequences, output_logits, vq_loss, perplexity, indices
    
    def encode(self, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None, 
               aggregate_mode="linear", quantize_cot_only=True, no_vq=False):
        """
        Enhanced encode method with optional vector quantization bypass.
        
        Args:
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            aggregate_mode (str): Mode of aggregation
            quantize_cot_only (bool): If True, only quantize COT positions (K to K+L-1). 
                                    If False, quantize all positions (0 to K+L-1).
            no_vq (bool): If True, bypass vector quantization and return aggregated embeddings directly
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, indices) 
        """
        # Call parent encode method to get the aggregated embeddings
        if no_vq:
            # For no_vq mode, we need to get the aggregated embeddings without quantization
            # We'll call the parent encode method but intercept before VQ
            batch_size, K = prompt_sequences.shape
            _, M, L = cot_sequences.shape
            
            # Check if gradient checkpointing is enabled to determine caching strategy
            use_caching = not self.is_gradient_checkpointing_enabled()
            
            if use_caching:
                # CACHING APPROACH: Encode prompt first, then COT with cached activations
                
                # Step 1: Encode prompt sequences with caching
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
                cot_flat = cot_sequences.view(batch_size * M, L)  # [batch_size * M, L]
                if cot_mask is not None:
                    cot_mask_flat = cot_mask.view(batch_size * M, L)  # [batch_size * M, L]
                else:
                    cot_mask_flat = None
                
                # Pad prompt cache M times using helper method
                padded_cache = self._pad_kv_cache(prompt_cache, batch_size, M)
                
                # Step 3: Continue encoding with COT sequences using cached prompt activations
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
                cot_memory = cot_outputs.last_hidden_state
                
                if quantize_cot_only:
                    # Only use COT activations, no need to concatenate with prompt
                    cot_memory = cot_memory.view(batch_size, M, L, -1)
                    memory = cot_memory.transpose(1, 2)  # [batch_size, L, M, d_model]
                else:
                    # Pad prompt activations M times to match COT batch size
                    padded_prompt_activations = prompt_activations.unsqueeze(1).expand(-1, M, -1, -1)
                    padded_prompt_activations = padded_prompt_activations.reshape(batch_size * M, K, -1)
                    
                    # Concatenate prompt activations with COT activations to get full sequence
                    full_memory = torch.cat([padded_prompt_activations, cot_memory], dim=1)  # [batch_size * M, K + L, d_model]
                    
                    # Reshape to group tokens at same positions across sequences
                    full_memory = full_memory.view(batch_size, M, K + L, -1)
                    memory = full_memory.transpose(1, 2)  # [batch_size, K + L, M, d_model]
            
            else:
                # NON-CACHING APPROACH: Process full sequence in one pass
                # Reshape COT sequences to [batch_size * M, L]
                cot_flat = cot_sequences.view(batch_size * M, L)  # [batch_size * M, L]
                
                # Pad prompt sequences M times to match COT batch size
                padded_prompt = prompt_sequences.unsqueeze(1).expand(-1, M, -1)
                padded_prompt = padded_prompt.reshape(batch_size * M, K)
                
                # Concatenate prompt and COT sequences
                full_sequences = torch.cat([padded_prompt, cot_flat], dim=1)  # [batch_size * M, K + L]
                
                # Create combined attention mask using helper method
                if cot_mask is not None:
                    cot_mask_flat = cot_mask.view(batch_size * M, L)  # [batch_size * M, L]
                else:
                    cot_mask_flat = None
                
                combined_mask = self._create_combined_mask(prompt_mask, cot_mask_flat, batch_size, M, K, L, cot_flat.device)
                
                # Encode full sequence in one pass
                full_outputs = self.encoder(
                    input_ids=full_sequences,
                    attention_mask=combined_mask,
                    use_cache=False,  # No caching when gradient checkpointing is enabled
                    return_dict=True
                )
                
                # Extract hidden states
                full_memory = full_outputs.last_hidden_state
                
                if quantize_cot_only:
                    # Only use COT activations (positions K to K+L-1)
                    cot_memory = full_memory[:, K:, :]  # [batch_size * M, L, d_model]
                    # Reshape to group tokens at same positions across sequences
                    cot_memory = cot_memory.view(batch_size, M, L, -1)
                    memory = cot_memory.transpose(1, 2)  # [batch_size, L, M, d_model]
                else:
                    # Use all activations
                    # Reshape to group tokens at same positions across sequences
                    full_memory = full_memory.view(batch_size, M, K + L, -1)
                    memory = full_memory.transpose(1, 2)  # [batch_size, K + L, M, d_model]

            # COMMON PROCESSING: Reshape for aggregation
            memory = memory.reshape(-1, M, memory.size(-1))  # [batch_size*L or batch_size*(K+L), M, d_model]

            # Aggregate the memory content per-prompt into single chains 
            aggregated = self.aggregate(memory, mode=aggregate_mode) # [batch_size*L or batch_size*(K+L), d_model]
            
            # For no_vq mode, tile the aggregated embeddings back to match the expected shape
            # This mimics what the vector quantizer would do after quantization
            quantized = aggregated.unsqueeze(1).expand(-1, M, -1)  # [batch_size*(L or K+L), M, d_model]
            
            # Reshape back using the appropriate length
            quantized = quantized.view(batch_size, -1, M, quantized.size(-1))
            
            # Return dummy values for VQ-related outputs
            vq_loss = torch.tensor(0.0, device=quantized.device)
            perplexity = torch.tensor(0.0, device=quantized.device)
            indices = torch.zeros(batch_size, device=quantized.device, dtype=torch.long)
            
            return quantized, vq_loss, perplexity, indices
        else:
            # Normal VQ mode - call parent encode method
            return super().encode(prompt_sequences, cot_sequences, prompt_mask, cot_mask, 
                                aggregate_mode, quantize_cot_only)
    
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
    
    def enable_batch_norm(self):
        """Enable batch normalization in the enhanced vector quantizer."""
        self.vector_quantizer.enable_batch_norm()
    
    def disable_batch_norm(self):
        """Disable batch normalization in the enhanced vector quantizer."""
        self.vector_quantizer.disable_batch_norm()
    
    def is_batch_norm_enabled(self):
        """
        Check if batch normalization is enabled in the enhanced vector quantizer.
        
        Returns:
            bool: True if batch normalization is enabled, False otherwise
        """
        return self.vector_quantizer.is_batch_norm_enabled()