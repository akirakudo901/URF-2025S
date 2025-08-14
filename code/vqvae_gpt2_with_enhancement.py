# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Optional
from sklearn.cluster import KMeans, MiniBatchKMeans

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
        # Use numpy arrays instead of PyTorch tensors to avoid memory leaks
        self.reservoir = None
        self.reservoir_count = 0
        self.count = 0
        # Flag to control whether reservoir is actively collecting samples
        self.reservoir_open = True
    
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
    
    def _initialize_reservoir(self, sample_shape):
        """Initialize the reservoir numpy array with the correct shape."""
        if self.reservoir is None:
            self.reservoir = np.zeros((self.reservoir_size, sample_shape[-1]), dtype=np.float32)
            self.reservoir_count = 0
    
    def add_sample(self, sample):
        """
        Add a sample to the reservoir using reservoir sampling.
        
        Args:
            sample (torch.Tensor): Sample to add to reservoir
        """
        # Check if reservoir is open for collecting samples
        if not self.reservoir_open:
            return  # Fail silently if reservoir is closed
        
        sample = self._validate_and_flatten_sample(sample)
        
        # Convert to numpy and ensure it's the right shape
        if sample.dim() == 1:
            sample_np = sample.detach().cpu().numpy()
        else:
            sample_np = sample.detach().cpu().numpy()
        
        # Initialize reservoir if needed
        self._initialize_reservoir(sample_np.shape)
        
        self.count += 1
        if self.reservoir_count < self.reservoir_size:
            self.reservoir[self.reservoir_count] = sample_np
            self.reservoir_count += 1
        else:
            # Reservoir sampling: replace with probability reservoir_size/count
            if torch.rand(1).item() < self.reservoir_size / self.count:
                idx = torch.randint(0, self.reservoir_size, (1,)).item()
                self.reservoir[idx] = sample_np
    
    def add_samples(self, samples):
        """
        Add multiple samples to the reservoir efficiently.
        
        Args:
            samples (torch.Tensor): Batch of samples to add [batch_size, ...]
        """
        # Check if reservoir is open for collecting samples
        if not self.reservoir_open:
            return  # Fail silently if reservoir is closed
        
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
        
        # Convert to numpy for efficient processing
        samples_np = samples.detach().cpu().numpy()
        batch_size = samples_np.shape[0]
        
        # Initialize reservoir if needed
        self._initialize_reservoir(samples_np.shape)
        
        # First, fill the reservoir if it's not full
        remaining_capacity = max(0, self.reservoir_size - self.reservoir_count)
        samples_to_fill = min(remaining_capacity, batch_size)
        
        if samples_to_fill > 0:
            # Add samples to fill the reservoir
            start_idx = self.reservoir_count
            end_idx = start_idx + samples_to_fill
            self.reservoir[start_idx:end_idx] = samples_np[:samples_to_fill]
            self.reservoir_count += samples_to_fill
            self.count += samples_to_fill
            samples_np = samples_np[samples_to_fill:]  # Remove used samples
            batch_size -= samples_to_fill
        
        # If we still have samples and reservoir is full, use reservoir sampling
        if batch_size > 0 and self.reservoir_count >= self.reservoir_size:
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
                    self.reservoir[idx] = samples_np[i]
            
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
        if self.reservoir is None or self.reservoir_count == 0:
            return None
            
        if num_samples is None:
            num_samples = self.reservoir_count
        if num_samples > self.reservoir_count:
            num_samples = self.reservoir_count
        
        # Get the actual samples (only the filled portion)
        actual_samples = self.reservoir[:self.reservoir_count]
        
        if shuffle:
            # Shuffle and return requested number of samples
            indices = np.random.permutation(self.reservoir_count)[:num_samples]
            selected_samples = actual_samples[indices]
        else:
            # Return first num_samples without shuffling
            selected_samples = actual_samples[:num_samples]
        
        # Convert back to PyTorch tensor
        return torch.from_numpy(selected_samples).float()
    
    def clear_reservoir(self):
        """Clear the reservoir to free memory."""
        self.reservoir = None
        self.reservoir_count = 0
        self.count = 0
    
    def get_memory_usage(self):
        """Get memory usage information about the reservoir."""
        if self.reservoir is None:
            return {
                'reservoir_size': 0,
                'reservoir_count': 0,
                'memory_bytes': 0,
                'memory_mb': 0.0
            }
        
        memory_bytes = self.reservoir.nbytes
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            'reservoir_size': self.reservoir_size,
            'reservoir_count': self.reservoir_count,
            'memory_bytes': memory_bytes,
            'memory_mb': memory_mb
        }
    
    def disable_reservoir(self):
        """
        Disable the reservoir from collecting new samples.
        Existing samples remain available for retrieval.
        """
        self.reservoir_open = False
        print("Reservoir disabled - no new samples will be collected")
    
    def enable_reservoir(self):
        """
        Enable the reservoir to collect new samples.
        """
        self.reservoir_open = True
        print("Reservoir enabled - samples will be collected again")
    
    def is_reservoir_open(self):
        """
        Check if the reservoir is open for collecting samples.
        
        Returns:
            bool: True if reservoir is open, False otherwise
        """
        return self.reservoir_open

class EnhancedVectorQuantizer(nn.Module):
    MAX_KMEANS_SIZE = 16384

    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 commitment_cost: float = 0.25, ema_decay: float = 0.99,
                 reset_threshold: float = 0.1,
                 reset_frequency: int = 1000, use_ema: bool = True,
                 max_reset_steps: Optional[int] = None, reservoir_size: int = 10000,
                 reset_strategy: str = 'partial', use_batch_norm: bool = True,
                 regularization_loss_weight: float = 0.01):
        """
        Enhanced Vector quantizer initialization with EMA updates and reset mechanisms.
        
        Args:
            num_embeddings: Number of embeddings in the codebook
            embedding_dim: Dimension of each embedding
            commitment_cost: Weight for the commitment loss
            ema_decay: Decay rate for EMA updates
            reset_threshold: Threshold for triggering codebook reset (usage ratio)
            reset_frequency: Frequency of checking for codebook reset
            use_ema: Whether to use EMA updates
            max_reset_steps: Maximum training steps during which resets are allowed (None = no limit)
            reservoir_size: Size of reservoir for data-dependent initialization
            reset_strategy: Strategy for automatic resets - 'partial' (reset unused codes) or 'full' (reset entire codebook)
            use_batch_norm: Whether to use batch normalization before vector quantization
            regularization_loss_weight: Weight for the regularization loss (L2 + orthogonality regularization)
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold
        self.reset_frequency = reset_frequency
        self.use_ema = use_ema
        self.max_reset_steps = max_reset_steps
        self.reset_strategy = reset_strategy
        self.use_batch_norm = use_batch_norm
        self.regularization_loss_weight = regularization_loss_weight
        
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
        
        self.use_minibatch_kmeans = reservoir_size > EnhancedVectorQuantizer.MAX_KMEANS_SIZE
        print("="*60)
        print(f"The reservoir size exceeds EnhancedVectorQuantizer.MAX_KMEANS_SIZE={EnhancedVectorQuantizer.MAX_KMEANS_SIZE}, so we perform MiniBatchKMeans!")
        print("="*60)
            
        
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
        if self.use_minibatch_kmeans:
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', 
                                     batch_size=EnhancedVectorQuantizer.MAX_KMEANS_SIZE, 
                                     n_init='auto')
        else:
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', 
                            n_init="auto")
        
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
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1).detach()

        # Update EMA cluster sizes
        ema_decay_val = self._ema_decay.item()
        self._ema_cluster_size.mul_(ema_decay_val).add_( # [num_embeddings]
            (1 - ema_decay_val) * torch.sum(encodings, 0)
        )
        
        # Update EMA weights
        n = torch.sum(self._ema_cluster_size).item()
        # below, matmul is the sum of all latents mapped to each code
        self._ema_w.mul_(ema_decay_val).add_(                           # [num_embeddings x emb_dim]
            (1 - ema_decay_val) * torch.matmul(encodings.T, flat_input).detach()
        )

        # Normalize EMA weights
        if n > 0:
            # Add small epsilon to prevent division by zero
            cluster_sizes_safe = self._ema_cluster_size + 1e-8
            self.embedding.weight.data.copy_(self._ema_w / cluster_sizes_safe.unsqueeze(1))
    
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
            tuple: (quantized, total_loss, perplexity, encoding_indices, debug_stats)
        """

        def compute_loss(quantized, inputs, encoding_indices):
            # Standard VQ-VAE losses
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            if self.use_ema:
                vq_loss = self.commitment_cost * e_latent_loss
            else:
                q_latent_loss = F.mse_loss(quantized, inputs.detach())
                vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
            # Additional losses
            regularization_loss = self._compute_regularization_loss(self.embedding.weight)
            
            # Combined loss
            weighted_regularization_loss = self.regularization_loss_weight * regularization_loss
            total_loss = vq_loss + weighted_regularization_loss
            
            # Check for NaN and clip if necessary
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss detected. vq_loss: {vq_loss}, reg_loss: {regularization_loss}")
                # Fall back to just VQ loss if other losses are problematic
                total_loss = vq_loss
            
            # Clip the total loss to prevent explosion
            return torch.clamp(total_loss, max=100.0), weighted_regularization_loss

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
                    + torch.sum(self.embedding.weight.detach()**2, dim=1)
                    - 2 * torch.matmul(normalized_inputs, self.embedding.weight.detach().T))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1) # [ (batch x seq_len) ]
        
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,  # [ (batch x seq_len), emb_num ]
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Training-specific updates
        current_usage = encodings.sum(dim=0).detach().long()
        
        if self.training:
            # Add to reservoir for future re-initialization
            # Use detach() only to avoid gradient tracking, no need for clone() since we convert to numpy
            self.reservoir_sampler.add_samples(normalized_inputs.detach())
            
            # Update EMA statistics
            if self.use_ema:
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
        
        # Quantize & Reshape
        quantized = torch.matmul(encodings, self.embedding.weight.detach()) # [ (batch x seq_len), emb_dim ]
        quantized = quantized.view(input_shape)
        normalized_inputs = normalized_inputs.view(input_shape)

        # Loss computation (only during training)
        if self.training:
            total_loss, weighted_regularization_loss = compute_loss(
                quantized, normalized_inputs, encoding_indices)
        else:
            # During inference, do not track gradients
            with torch.no_grad():
                total_loss, weighted_regularization_loss = compute_loss(
                    quantized, normalized_inputs, encoding_indices)
        
        quantized = normalized_inputs + (quantized - normalized_inputs).detach()  # Straight-through estimator
        # Perplexity: measure of latent code usage, keep it mid (high=uniform, no learning, low=not used fully)
        perplexity = compute_perplexity(encoding_indices, "indices")
        
        # Handle case where input_shape has only 1 dimension
        if len(input_shape) == 1:
            indices_shape = input_shape
        else:
            indices_shape = input_shape[:-1]
        
        # Compute post-batch-norm input norm stats for debug
        post_bn_norms = torch.norm(normalized_inputs.detach().view(-1, self.embedding_dim), dim=-1)
        debug_stats = {
            'vq_post_bn_input_norm_mean': post_bn_norms.mean().item(),
            'vq_post_bn_input_norm_std': post_bn_norms.std().item(),
            'weighted_regularization_loss': weighted_regularization_loss.item()
        }
        return quantized, total_loss, perplexity, encoding_indices.view(indices_shape), debug_stats
    
    def get_codebook_stats(self):
        """
        Get statistics about codebook usage and health.
        
        Returns:
            dict: Dictionary containing various statistics
        """
        thresholds = [0.000001, 0.00005, 0.0001, 0.0005, 0.01]  # 0.0001%, 0.005%, 0.01%, 0.05%, 1%
        if self.training:
            # During training, return training statistics
            total_usage = self._usage_counts.sum().item()
            usage_ratios = (self._usage_counts / (total_usage + 1e-8)).clone().detach().cpu()
            stats = {
                'total_usage': total_usage,
                'usage_counts': self._usage_counts.clone().detach().cpu(),
                'usage_ratio': usage_ratios,
                'unused_codes': (self._usage_counts == 0).sum().item(),
                'unused_ratio': (self._usage_counts == 0).float().mean().item(),
                'reset_counter': self._reset_counter.item(),
            }
            for t in thresholds:
                stats[f'codes_below_{int(t*1000)/10:.1f}_percent'] = (usage_ratios < t).sum().item()
            if self.use_ema:
                stats.update({
                    'ema_cluster_sizes': self._ema_cluster_size.clone().detach().cpu(),
                    'ema_weights_norm': torch.norm(self._ema_w, p=2, dim=1).mean().item(),
                })
        else:
            # During inference, return inference statistics (read-only)
            inference_usage = self._inference_usage_counts
            total_inference_usage = inference_usage.sum().item()
            usage_ratios = (inference_usage / (total_inference_usage + 1e-8)).clone().detach().cpu()
            stats = {
                'total_usage': total_inference_usage,
                'usage_counts': inference_usage.clone().detach().cpu(),
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
                    'ema_cluster_sizes': self._ema_cluster_size.clone().detach().cpu(),
                    'ema_weights_norm': torch.norm(self._ema_w, p=2, dim=1).mean().item(),
                })
        
        # Add reservoir information
        stats['reservoir_size'] = self.reservoir_sampler.reservoir_size
        stats['reservoir_count'] = self.reservoir_sampler.reservoir_count
        stats['reservoir_memory_usage'] = self.reservoir_sampler.get_memory_usage()
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

    def get_reservoir_state(self):
        """
        Get the current state of the reservoir sampler for checkpointing.
        
        Returns:
            dict: Dictionary containing reservoir state information
        """
        if not hasattr(self, 'reservoir_sampler') or self.reservoir_sampler is None:
            return {
                'reservoir_size': 0,
                'embedding_size': None,
                'reservoir': None,
                'reservoir_count': 0,
                'count': 0
            }
        
        sampler = self.reservoir_sampler
        return {
            'reservoir_size': sampler.reservoir_size,
            'embedding_size': sampler.embedding_size,
            'reservoir': sampler.reservoir.copy() if sampler.reservoir is not None else None,
            'reservoir_count': sampler.reservoir_count,
            'count': sampler.count,
            'reservoir_open': sampler.reservoir_open
        }
    
    def set_reservoir_state(self, reservoir_state):
        """
        Restore the reservoir sampler state from checkpoint.
        
        Args:
            reservoir_state (dict): Dictionary containing reservoir state information
        """
        if not hasattr(self, 'reservoir_sampler') or self.reservoir_sampler is None:
            print("Warning: No reservoir sampler found, cannot restore state")
            return
        
        sampler = self.reservoir_sampler
        
        # Validate the state
        if not isinstance(reservoir_state, dict):
            raise ValueError("reservoir_state must be a dictionary")
        
        required_keys = ['reservoir_size', 'embedding_size', 'reservoir', 'reservoir_count', 'count', 'reservoir_open']
        for key in required_keys:
            if key not in reservoir_state:
                raise ValueError(f"reservoir_state missing required key: {key}")
        
        # Restore the state
        sampler.reservoir_size = reservoir_state['reservoir_size']
        sampler.embedding_size = reservoir_state['embedding_size']
        sampler.reservoir_count = reservoir_state['reservoir_count']
        sampler.count = reservoir_state['count']
        sampler.reservoir_open = reservoir_state.get('reservoir_open', True)  # Default to True for backward compatibility
        
        # Handle the reservoir numpy array
        if reservoir_state['reservoir'] is not None:
            # Convert back to numpy array if it was stored as a list
            if isinstance(reservoir_state['reservoir'], list):
                sampler.reservoir = np.array(reservoir_state['reservoir'], dtype=np.float32)
            else:
                sampler.reservoir = reservoir_state['reservoir'].copy()
        else:
            sampler.reservoir = None
        
        print(f"Reservoir state restored: size={sampler.reservoir_size}, count={sampler.reservoir_count}, total_samples={sampler.count}")
    
    def disable_reservoir(self):
        """
        Disable the reservoir from collecting new samples.
        Existing samples remain available for retrieval.
        """
        if hasattr(self, 'reservoir_sampler') and self.reservoir_sampler is not None:
            self.reservoir_sampler.disable_reservoir()
        else:
            print("Warning: No reservoir sampler found")
    
    def enable_reservoir(self):
        """
        Enable the reservoir to collect new samples.
        """
        if hasattr(self, 'reservoir_sampler') and self.reservoir_sampler is not None:
            self.reservoir_sampler.enable_reservoir()
        else:
            print("Warning: No reservoir sampler found")
    
    def is_reservoir_open(self):
        """
        Check if the reservoir is open for collecting samples.
        
        Returns:
            bool: True if reservoir is open, False otherwise
        """
        if hasattr(self, 'reservoir_sampler') and self.reservoir_sampler is not None:
            return self.reservoir_sampler.is_reservoir_open()
        return False
    
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
    
    def clear_reservoir(self):
        """
        Clear the reservoir sampler to free memory.
        This is useful when you want to free up memory during training.
        """
        if hasattr(self, 'reservoir_sampler'):
            self.reservoir_sampler.clear_reservoir()
            print("Reservoir cleared to free memory")
    
    def get_reservoir_memory_usage(self):
        """
        Get memory usage information about the reservoir sampler.
        
        Returns:
            dict: Dictionary containing memory usage information
        """
        if hasattr(self, 'reservoir_sampler'):
            return self.reservoir_sampler.get_memory_usage()
        return {'error': 'reservoir_sampler not found'}


class EnhancedGPT2VQVAE(GPT2VQVAE):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 aggregation_hidden_dim2=None,
                 num_thoughts=32, n_positions=1024, 
                 use_pretrained_encoder=True, use_pretrained_decoder=True,
                 pretrained_model_name="gpt2",
                 # Text embedding loading and freezing parameters
                 load_text_embeddings_encoder=False, freeze_text_embeddings_encoder=False,
                 load_text_embeddings_decoder=False, freeze_text_embeddings_decoder=False,
                 # Vector Quantizer specific parameters
                 ema_decay=0.99, reset_threshold=0.1,
                 diversity_gamma=None, #kept in case we are loading older models - deprecated
                 reset_frequency=1000, use_ema=True, max_reset_steps=None, reservoir_size=10000,
                 reset_strategy='partial', use_batch_norm=True, regularization_loss_weight=0.01,
                 # Unified parameters (applied to both encoder and decoder if specified)
                 n_layer=12, n_head=12, n_inner=None, dropout=0.1, activation_function="gelu",
                 # Encoder-specific parameters (take precedence over unified if specified)
                 encoder_n_layer=None, encoder_n_head=None, encoder_n_inner=None,
                 encoder_dropout=None, encoder_activation_function=None,
                 # Decoder-specific parameters (take precedence over unified if specified)
                 decoder_n_layer=None, decoder_n_head=None, decoder_n_inner=None,
                 decoder_dropout=None, decoder_activation_function=None, 
                 reset_stop_fraction=None,
                 # Only latent decode mode
                 only_latent_decode=False,
                 # Simple decoder mode
                 simple_decoder=False,
                 embed_sum_decode=False,
                 compress_beam_search=False,
                 interchain=False,
                 interchain_positional=False):
        """
        Enhanced GPT2VQVAE with improved vector quantization.
        
        This enhanced version includes:
        - EMA (Exponential Moving Average) updates for codebook learning
        - Automatic codebook reset mechanisms for unused embeddings
        - Enhanced monitoring and statistics for codebook health
        - Support for loading and freezing text embeddings separately for encoder and decoder
        - Configurable regularization loss weight for embedding diversity
        
        The enhanced codebook training scheme reduces to normal VQ-VAE training when:
        - ema_decay = 0.0 (no EMA updates)
        - reset_threshold = 0.0 (no automatic resets)
        - use_ema = False (EMA disabled)
        - regularization_loss_weight = 0.0 (no regularization)
        
        Text embedding loading and freezing:
        - When use_pretrained_encoder/decoder=False, you can still load text embeddings from pretrained models
        - This allows initialization with good token representations while keeping other weights random
        - Text embeddings can be frozen to prevent updates during training
        """
        if diversity_gamma is not None:
            print("Warning: diversity_gamma is deprecated as the diversity loss has been removed.")
        # Extract vector quantizer parameters
        vq_params = {
            'num_embeddings': num_embeddings,
            'embedding_dim': d_model,
            'commitment_cost': commitment_cost,
            'ema_decay': ema_decay,
            'reset_threshold': reset_threshold,
            'reset_frequency': reset_frequency,
            'use_ema': use_ema,
            'max_reset_steps': max_reset_steps,
            'reservoir_size': reservoir_size,
            'reset_strategy': reset_strategy,
            'use_batch_norm': use_batch_norm,
            'regularization_loss_weight': regularization_loss_weight
        }
        
        # Call parent constructor with remaining parameters
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            aggregation_hidden_dim=aggregation_hidden_dim,
            aggregation_hidden_dim2=aggregation_hidden_dim2,
            num_thoughts=num_thoughts,
            n_positions=n_positions,
            use_pretrained_encoder=use_pretrained_encoder,
            use_pretrained_decoder=use_pretrained_decoder,
            pretrained_model_name=pretrained_model_name,
            # Text embedding loading and freezing parameters
            load_text_embeddings_encoder=load_text_embeddings_encoder,
            freeze_text_embeddings_encoder=freeze_text_embeddings_encoder,
            load_text_embeddings_decoder=load_text_embeddings_decoder,
            freeze_text_embeddings_decoder=freeze_text_embeddings_decoder,
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
            decoder_activation_function=decoder_activation_function,
            # Decoder mode
            only_latent_decode=only_latent_decode,
            simple_decoder=simple_decoder,
            embed_sum_decode=embed_sum_decode,
            compress_beam_search=compress_beam_search,
            interchain=interchain,
            interchain_positional=interchain_positional
        )
        
        # Replace the vector quantizer with enhanced version
        self.vector_quantizer = EnhancedVectorQuantizer(**vq_params)
        
        # Store enhanced configuration for checkpoint validation
        self._enhanced_config = {
            'ema_decay': ema_decay,
            'reset_threshold': reset_threshold,
            'reset_frequency': reset_frequency,
            'use_ema': use_ema,
            'reservoir_size': reservoir_size,
            'reset_strategy': reset_strategy,
            'use_batch_norm': use_batch_norm,
            'regularization_loss_weight': regularization_loss_weight
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
    
    def clear_reservoir(self):
        """
        Clear the reservoir sampler to free memory.
        This is useful when you want to free up memory during training.
        """
        self.vector_quantizer.clear_reservoir()
    
    def get_reservoir_memory_usage(self):
        """
        Get memory usage information about the reservoir sampler.
        
        Returns:
            dict: Dictionary containing memory usage information
        """
        return self.vector_quantizer.get_reservoir_memory_usage()
    
    def get_reservoir_state(self):
        """
        Get the current state of the reservoir sampler for checkpointing.
        
        Returns:
            dict: Dictionary containing reservoir state information
        """
        return self.vector_quantizer.get_reservoir_state()
    
    def set_reservoir_state(self, reservoir_state):
        """
        Restore the reservoir sampler state from checkpoint.
        
        Args:
            reservoir_state (dict): Dictionary containing reservoir state information
        """
        self.vector_quantizer.set_reservoir_state(reservoir_state)
    
    def disable_reservoir(self):
        """
        Disable the reservoir from collecting new samples.
        Existing samples remain available for retrieval.
        """
        self.vector_quantizer.disable_reservoir()
    
    def enable_reservoir(self):
        """
        Enable the reservoir to collect new samples.
        """
        self.vector_quantizer.enable_reservoir()
    
    def is_reservoir_open(self):
        """
        Check if the reservoir is open for collecting samples.
        
        Returns:
            bool: True if reservoir is open, False otherwise
        """
        return self.vector_quantizer.is_reservoir_open()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None, **kwargs):
        """
        Initialize and load an EnhancedGPT2VQVAE model from a checkpoint file.
        
        This method extends the parent from_checkpoint method to handle enhanced-specific
        configuration parameters like EMA decay, reset thresholds, etc.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            device (str, optional): Device to load the model on (if None, uses 'cuda' if available, else 'cpu')
            **kwargs: Additional arguments to override the checkpoint configuration
            
        Returns:
            EnhancedGPT2VQVAE: Fully initialized and loaded model
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint is missing required configuration
            ValueError: If configuration is invalid
            
        Example:
            >>> # Load enhanced model with checkpoint configuration
            >>> model = EnhancedGPT2VQVAE.from_checkpoint('checkpoints/enhanced_model_epoch_10.pt')
            
            >>> # Load enhanced model with overridden parameters
            >>> model = EnhancedGPT2VQVAE.from_checkpoint(
            ...     'checkpoints/enhanced_model_epoch_10.pt',
            ...     device='cpu',
            ...     ema_decay=0.95,
            ...     use_ema=False
            ... )
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint to extract configuration
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_config' not in checkpoint:
            raise KeyError(f"Checkpoint file {checkpoint_path} does not contain 'model_config'")
        
        # Extract model configuration from checkpoint
        model_config = checkpoint['model_config'].copy()
        
        # Check if this is an enhanced model checkpoint by looking for enhanced configuration
        is_enhanced_checkpoint = False
        enhanced_config = {}
        
        # Check for enhanced configuration in the checkpoint
        enhanced_keys = ['ema_decay', 'reset_threshold', 'reset_frequency', 
                           'use_ema', 'max_reset_steps', 'reservoir_size', 'reset_strategy', 'use_batch_norm', 'regularization_loss_weight']
        if 'enhanced_config' in checkpoint:
            enhanced_config = checkpoint['enhanced_config'].copy()
            is_enhanced_checkpoint = True
            print("Detected enhanced model checkpoint with enhanced configuration")
        elif any(key in model_config for key in enhanced_keys):
            # Extract enhanced parameters from model_config if they exist
            enhanced_config = {key: model_config.pop(key) for key in enhanced_keys if key in model_config}
            is_enhanced_checkpoint = bool(enhanced_config)
            if is_enhanced_checkpoint:
                print("Detected enhanced model checkpoint with enhanced parameters in model_config")
        
        # Handle enhanced-specific overrides
        enhanced_overrides = {}
        
        for key in enhanced_keys:
            if key in kwargs:
                enhanced_overrides[key] = kwargs.pop(key)
        
        # Update enhanced configuration with overrides
        enhanced_config.update(enhanced_overrides)
        
        # If this was an enhanced checkpoint, ensure we have the enhanced parameters
        if is_enhanced_checkpoint:
            # Merge enhanced configuration into model_config for initialization
            model_config.update(enhanced_config)
        else:
            # This might be a regular GPT2VQVAE checkpoint, use default enhanced parameters
            print("Warning: Loading regular GPT2VQVAE checkpoint into EnhancedGPT2VQVAE model")
            print("Using default enhanced parameters. Consider using GPT2VQVAE.from_checkpoint() for regular checkpoints.")
        
        # Override with any remaining kwargs
        model_config.update(kwargs)
        
        # Use parent's implementation for the rest
        model = cls._from_checkpoint_impl(checkpoint_path, device, model_config, **kwargs)
        
        # Print enhanced configuration summary
        if enhanced_config:
            print("Enhanced configuration:")
            for key, value in enhanced_config.items():
                print(f"  {key}: {value}")
        
        return model