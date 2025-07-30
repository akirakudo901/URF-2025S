# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import torch
import torch.optim as optim
from typing import Optional, Dict, Any
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc
import numpy as np
import sys

# Add the current directory to the path to import dependencies
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules and functions
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE

# Import the parent trainer class
from gpt2_vqvae_trainer import GPT2VQVAETrainer

class EnhancedGPT2VQVAETrainer(GPT2VQVAETrainer):
    """
    Trainer for EnhancedGPT2VQVAE, inherits from GPT2VQVAETrainer but uses EnhancedGPT2VQVAE as the model.
    
    This enhanced trainer provides additional functionality for the enhanced vector quantizer:
    - EMA (Exponential Moving Average) updates for codebook learning
    - Automatic codebook reset mechanisms for unused embeddings
    - Enhanced monitoring and statistics for codebook health
    
    The enhanced codebook training scheme reduces to normal VQ-VAE training when:
    - ema_decay = 0.0 (no EMA updates)
    - reset_threshold = 0.0 (no automatic resets)
    - use_ema = False (EMA disabled)
    """
    CLEAN_PLOT_PATTERNS = [
        "epoch_*_history_batchnorm*.png",
        "epoch_*_history_progressive_loss.png",
        "epoch_*_history_post_bn_input_norm.png",
        ] + GPT2VQVAETrainer.CLEAN_PLOT_PATTERNS
    
    def __init__(self, 
                 model_config: Dict[str, Any], 
                 training_config: Dict[str, Any], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 tracking_functions: Optional[Dict[str, Any]] = None, 
                 run_name : Optional[str] = "ANONYM_RUN",
                 overwrite_plots: bool = True):
        # Filter out enhanced VQ-VAE specific parameters for parent constructor
        enhanced_vq_params = {
            'ema_decay', 'reset_threshold', 
            'reset_frequency', 'use_ema', 'reset_stop_fraction',
            # for further enhancement
            'max_reset_steps', 'reservoir_size', 'reset_strategy', 'use_batch_norm'
        }
        
        # Create filtered configs for parent constructor
        filtered_model_config = {k: v for k, v in model_config.items() if k not in enhanced_vq_params}
        filtered_training_config = {k: v for k, v in training_config.items() if k != 'enhanced_codebook_tracking'}
        
        # Set up enhanced tracking functions, for those not provided
        enhanced_tracking_functions = {
            'track_codebook_usage': tracking_functions.get('track_codebook_usage', self._enhanced_track_codebook_usage),
            'save_codebook_plots': tracking_functions.get('save_codebook_plots', self._enhanced_save_codebook_plots),
            'tracking_enabled': tracking_functions.get('tracking_enabled', training_config.get('enhanced_codebook_tracking', True))
        }
        
        # Call parent constructor with filtered configs and enhanced tracking functions
        super().__init__(filtered_model_config, filtered_training_config, device, enhanced_tracking_functions, run_name, overwrite_plots)

        # Store original configs for enhanced features
        self.model_config = model_config
        self.training_config = training_config
        
        self.ensure_numeric_types(self.model_config)
        self.ensure_numeric_types(self.training_config)
        
        # Manage some model_config entries
        if 'max_reset_steps' not in model_config:
            model_config['max_reset_steps'] = None
        
        # Replace the model with EnhancedGPT2VQVAE using original config
        self.model = EnhancedGPT2VQVAE(**model_config).to(device)
        
        # Re-initialize optimizer and scheduler for the new model
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        if training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for EnhancedGPT2VQVAE")
        
        # Enhanced codebook tracking with granular control
        self.enhanced_codebook_tracking = training_config.get('enhanced_codebook_tracking', True)
        # Track batch norm parameter history per epoch
        self.detailed_bn_param_history = []  # List of dicts, one per epoch
        
        # Granular tracking flags for memory optimization
        tracking_config = training_config.get('codebook_tracking_config', {})
        self.track_usage_stats = tracking_config.get('track_usage_stats', True)
        self.track_diversity_metrics = tracking_config.get('track_diversity_metrics', True)
        self.track_ema_stats = tracking_config.get('track_ema_stats', True)
        self.track_reset_stats = tracking_config.get('track_reset_stats', True)
        self.track_reservoir_stats = tracking_config.get('track_reservoir_stats', True)
        self.save_tracking_history = tracking_config.get('save_tracking_history', True)
        self.print_tracking_info = tracking_config.get('print_tracking_info', True)
        
        # Initialize tracking history only if needed
        if self.enhanced_codebook_tracking and self.save_tracking_history:
            self.codebook_stats_history = []
            self.diversity_history = []
        else:
            self.codebook_stats_history = None
            self.diversity_history = None
        
        # Enhanced loss tracking for regularization losses
        self.weighted_regularization_losses = []
        self.detailed_weighted_regularization_losses = []
        
        if self.enhanced_codebook_tracking:
            print("Enhanced codebook tracking enabled with granular control:")
            print(f"  - Usage stats: {self.track_usage_stats}")
            print(f"  - Diversity metrics: {self.track_diversity_metrics}")
            print(f"  - EMA stats: {self.track_ema_stats}")
            print(f"  - Reset stats: {self.track_reset_stats}")
            print(f"  - Reservoir stats: {self.track_reservoir_stats}")
            print(f"  - Save history: {self.save_tracking_history}")
            print(f"  - Print info: {self.print_tracking_info}")
            print(f"EMA decay: {model_config.get('ema_decay', 0.99)}")
            print(f"Reset threshold: {model_config.get('reset_threshold', 0.1)}")
            print(f"Reset frequency: {model_config.get('reset_frequency', 1000)}")
            print(f"Use EMA: {model_config.get('use_ema', True)}")
            print(f"Reset stop fraction: {self.model_config.get('reset_stop_fraction', 0.2)}")
        else:
            print("Enhanced codebook tracking disabled")
        print("EnhancedGPT2VQVAE trainer initialized successfully.")

        # Training phase tracking - no current use, but might be useful later
        self.current_step = 0
    
    
    def train_epoch(self, train_loader, num_measurements_per_epoch, current_epoch=0, detailed_metrics_callback=None):
        """
        Enhanced train_epoch that tracks regularization losses in addition to standard losses.
        """
        # Set max_reset_steps on the first epoch if it's still None
        if self.model.vector_quantizer.max_reset_steps is None:
            # Calculate total training steps
            num_epochs = self.training_config['num_epochs']
            steps_per_epoch = len(train_loader)
            total_training_steps = num_epochs * steps_per_epoch
            
            # Calculate max_reset_steps from reset_stop_fraction
            reset_stop_fraction = self.model_config.get('reset_stop_fraction', 0.2)
            max_reset_steps = int(reset_stop_fraction * total_training_steps)
            
            # Set max_reset_steps on the vector quantizer
            self.model.vector_quantizer.max_reset_steps = max_reset_steps
            print(f"Set max_reset_steps to {max_reset_steps} (based on {total_training_steps} total steps, {reset_stop_fraction*100:.0f}% of training)")
        
        # Initialize tracking lists for this epoch
        self._current_bn_param_history = []
        self._detailed_post_bn_norm_history = []
        self._current_detailed_weighted_regularization_losses = []

        def bn_tracking_callback(**kwargs):
            vq = self.model.vector_quantizer
            if hasattr(vq, 'batch_norm') and vq.batch_norm is not None:
                self._current_bn_param_history.append({
                    'weight': vq.batch_norm.weight.detach().cpu().numpy(),
                    'bias': vq.batch_norm.bias.detach().cpu().numpy(),
                    'running_mean': vq.batch_norm.running_mean.detach().cpu().numpy(),
                    'running_var': vq.batch_norm.running_var.detach().cpu().numpy(),
                    'step': kwargs.get('batch_idx', None),
                })
            # Track post-batch-norm input norm stats if present
            if 'debug_stats' in kwargs:
                debug_stats = kwargs['debug_stats']
                if 'vq_post_bn_input_norm_mean' in debug_stats and 'vq_post_bn_input_norm_std' in debug_stats:
                    self._detailed_post_bn_norm_history.append({
                        'mean': debug_stats['vq_post_bn_input_norm_mean'],
                        'std': debug_stats['vq_post_bn_input_norm_std'],
                        'step': kwargs.get('batch_idx', None),
                    })
                if 'weighted_regularization_loss' in debug_stats:
                    self._current_detailed_weighted_regularization_losses.append(debug_stats['weighted_regularization_loss'])
                    
            if detailed_metrics_callback is not None:
                detailed_metrics_callback(**kwargs)

        # Call the parent train_epoch method with the callback
        result = super().train_epoch(train_loader, num_measurements_per_epoch, current_epoch, detailed_metrics_callback=bn_tracking_callback)
        
        # After each epoch, store a copy of the batch norm parameter history for this epoch
        self.detailed_bn_param_history.append(self._current_bn_param_history)
        del self._current_bn_param_history
        self._current_bn_param_history = None
        
        # Store post-bn norm history for this epoch
        if not hasattr(self, 'detailed_post_bn_norm_history'):
            self.detailed_post_bn_norm_history = []
        self.detailed_post_bn_norm_history.append(self._detailed_post_bn_norm_history)
        del self._detailed_post_bn_norm_history
        self._detailed_post_bn_norm_history = None
        
        # Store enhanced loss history for this epoch
        reg_losses = self._current_detailed_weighted_regularization_losses
        if len(reg_losses) > 0:
            result['avg_weighted_regularization_loss'] = sum(reg_losses) / len(reg_losses)
        else:
            result['avg_weighted_regularization_loss'] = 0
        self.detailed_weighted_regularization_losses.append(self._current_detailed_weighted_regularization_losses)
        del self._current_detailed_weighted_regularization_losses
        self._current_detailed_weighted_regularization_losses = None
        
        return result

    def _update_weights(self):
        """
        Enhanced weight update that increments step counter.
        """
        # Call parent weight update
        super()._update_weights()
        # Increment step counter
        self.current_step += 1

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None, remove_other_best_models: bool = True, **kwargs):
        """
        Enhanced checkpoint saving that includes phased training state and enhanced loss tracking.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
            checkpoint_path: Optional custom checkpoint path
            remove_other_best_models: If True, remove other best model checkpoints (default: True)
        """
        kwargs.update({ 
            "current_step": self.current_step,
            "weighted_regularization_losses": self.weighted_regularization_losses,
            "detailed_weighted_regularization_losses": self.detailed_weighted_regularization_losses,
            "detailed_bn_param_history": self.detailed_bn_param_history,
            "detailed_post_bn_norm_history": self.detailed_post_bn_norm_history
        })
        # Call parent save_checkpoint with enhanced data as kwargs
        super().save_checkpoint(epoch, metrics, is_best, checkpoint_path, remove_other_best_models, **kwargs)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Enhanced checkpoint loading that restores phased training state and enhanced loss tracking.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load checkpoint using parent method
        checkpoint = super().load_checkpoint(checkpoint_path)
        
        # Restore phased training state if available
        if 'current_step' in checkpoint:
            self.current_step = checkpoint['current_step']
            print(f"Restored training step: {self.current_step}")
        else:
            raise Exception("Checkpoint does not contain 'current_step' information. Phased training cannot be resumed properly without this data.")
        
        # Restore enhanced loss tracking if available
        if 'weighted_regularization_losses' in checkpoint:
            self.weighted_regularization_losses = checkpoint['weighted_regularization_losses']
            print(f"Restored weighted regularization losses history: {len(self.weighted_regularization_losses)} epochs")
        else:
            print("No weighted regularization losses history found in checkpoint")
            
        if 'detailed_weighted_regularization_losses' in checkpoint:
            self.detailed_weighted_regularization_losses = checkpoint['detailed_weighted_regularization_losses']
            print(f"Restored detailed weighted regularization losses history: {len(self.detailed_weighted_regularization_losses)} epochs")
        else:
            print("No detailed weighted regularization losses history found in checkpoint")
        
        # Restore batch norm parameter history if available
        if 'detailed_bn_param_history' in checkpoint:
            self.detailed_bn_param_history = checkpoint['detailed_bn_param_history']
            print(f"Restored detailed batch norm parameter history: {len(self.detailed_bn_param_history)} epochs")
        else:
            print("No detailed batch norm parameter history found in checkpoint")
            
        # Restore post-batch-norm norm history if available
        if 'detailed_post_bn_norm_history' in checkpoint:
            self.detailed_post_bn_norm_history = checkpoint['detailed_post_bn_norm_history']
            print(f"Restored detailed post-batch-norm norm history: {len(self.detailed_post_bn_norm_history)} epochs")
        else:
            print("No detailed post-batch-norm norm history found in checkpoint")
        
        return checkpoint
    
    def _enhanced_track_codebook_usage(self, dataset: Any, measurement_point: int) -> None:
        """
        Enhanced codebook tracking with additional statistics.
        
        Args:
            dataset: Dataset to sample from
            measurement_point: Current measurement point index
        """
        # First, call the parent class's default codebook tracking
        super()._default_track_codebook_usage(dataset, measurement_point)
        
        # Then, call the enhanced codebook tracking
        if not self.enhanced_codebook_tracking:
            return
        
        try:
            # Initialize stats dictionaries
            codebook_stats = {}
            diversity_metrics = {}
            
            # Get codebook statistics based on tracking flags
            if self.track_usage_stats or self.track_ema_stats or self.track_reset_stats or self.track_reservoir_stats:
                codebook_stats = self.model.get_vector_quantizer_stats()
            
            # Get diversity metrics only if needed
            if self.track_diversity_metrics:
                diversity_metrics = self.model.get_embedding_diversity()
            
            # Store results only if history saving is enabled
            if self.save_tracking_history:
                # Convert any tensors in codebook_stats to CPU
                cpu_codebook_stats = {}
                for key, value in codebook_stats.items():
                    if isinstance(value, torch.Tensor):
                        cpu_codebook_stats[key] = value.cpu().detach()
                    else:
                        cpu_codebook_stats[key] = value
                
                # Convert any tensors in diversity_metrics to CPU
                cpu_diversity_metrics = {}
                for key, value in diversity_metrics.items():
                    if isinstance(value, torch.Tensor):
                        cpu_diversity_metrics[key] = value.cpu().detach()
                    else:
                        cpu_diversity_metrics[key] = value
                
                self.codebook_stats_history.append(cpu_codebook_stats)
                self.diversity_history.append(cpu_diversity_metrics)
            
            # Print enhanced statistics only if enabled
            if self.print_tracking_info:
                print(f"\nEnhanced Codebook tracking (point {measurement_point}):")
                
                # Print usage statistics
                if self.track_usage_stats and codebook_stats:
                    print(f"  Total usage: {codebook_stats.get('total_usage', 'N/A')}")
                    print(f"  Unused codes: {codebook_stats.get('unused_codes', 'N/A')}/{self.model.vector_quantizer.num_embeddings} "
                          f"({codebook_stats.get('unused_ratio', 0)*100:.1f}%)")
                    
                    # Print rarely used codes statistics
                    # third position corresponds to threshold in %
                    codes_below_x_percent = [(k.split("_")[2], v) for (k, v) in codebook_stats.items()
                                              if k.startswith("codes_below_")]
                    for (thresh_perc, v) in codes_below_x_percent:
                        print(f"  Codes below {thresh_perc}%: {v}")

                # Print reset statistics
                if self.track_reset_stats and codebook_stats:
                    print(f"  Reset counter: {codebook_stats.get('reset_counter', 'N/A')}")
                
                # Print diversity metrics
                if self.track_diversity_metrics and diversity_metrics:
                    print(f"  Mean similarity: {diversity_metrics.get('mean_similarity', 0):.4f}")
                    print(f"  Embedding norm mean: {diversity_metrics.get('embedding_norm_mean', 0):.4f}")
                
                # Print EMA statistics
                if self.track_ema_stats and codebook_stats and 'ema_cluster_sizes' in codebook_stats:
                    ema_usage = (codebook_stats['ema_cluster_sizes'] > 0).sum().item()
                    print(f"  EMA active clusters: {ema_usage}/{self.model.vector_quantizer.num_embeddings}")
                
                # Print reservoir statistics
                if self.track_reservoir_stats and codebook_stats:
                    reservoir_size = codebook_stats.get('reservoir_size', 'N/A')
                    reservoir_count = codebook_stats.get('reservoir_count', 'N/A')
                    print(f"  Reservoir size: {reservoir_size}, count: {reservoir_count}")
            
            # Clear cache after enhanced tracking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to track enhanced codebook usage: {e}")
    
    def _print_epoch_metrics(self, train_metrics: Dict[str, Any], test_metrics: Dict[str, float]) -> None:
        """
        Print epoch metrics for enhanced trainer with additional regularization losses.
        
        Args:
            train_metrics: Training metrics from train_epoch
            test_metrics: Validation metrics from validate
        """
        # Call parent method to print basic metrics
        super()._print_epoch_metrics(train_metrics, test_metrics)
        
        # Print enhanced metrics
        print(f"Weighted Regularization Loss: {train_metrics['avg_weighted_regularization_loss']:.4f}")


    def _enhanced_save_codebook_plots(self, save_dir: str, epoch: int) -> None:
        """
        Save enhanced codebook tracking visualizations.
        
        Args:
            save_dir: Directory to save plots
            epoch: Current epoch number
        """
        # First, call the parent class's default codebook plotting
        super()._default_save_codebook_plots(save_dir, epoch)
        
        # Then, save the enhanced codebook plots only if tracking is enabled and history exists
        if not self.enhanced_codebook_tracking or not self.save_tracking_history or not self.codebook_stats_history:
            return
        
        try:
            # Save enhanced statistics plots only if we have usage/reset/EMA stats
            if self.track_usage_stats or self.track_reset_stats or self.track_ema_stats:
                stats_path = os.path.join(save_dir, f"enhanced_codebook_stats_epoch_{epoch}.png")
                self._plot_enhanced_codebook_stats(stats_path)
            
            # Save diversity evolution plots only if diversity tracking is enabled
            if self.track_diversity_metrics and self.diversity_history:
                diversity_path = os.path.join(save_dir, f"codebook_diversity_evolution_epoch_{epoch}.png")
                self._plot_diversity_evolution(diversity_path)
            
            print(f"Enhanced codebook plots saved to {save_dir}")
            
            # Clear cache after enhanced plotting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to save enhanced codebook plots: {e}")
    
    def _plot_enhanced_codebook_stats(self, save_path: str) -> None:
        """Plot enhanced codebook statistics over time."""
        if not self.codebook_stats_history or not self.save_tracking_history:
            return
        
        # Create a larger figure to accommodate more plots
        _, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Extract data
        epochs = list(range(len(self.codebook_stats_history)))
        total_usage = [stats['total_usage'] for stats in self.codebook_stats_history]
        unused_ratio = [stats['unused_ratio'] for stats in self.codebook_stats_history]
        reset_counter = [stats['reset_counter'] for stats in self.codebook_stats_history]
        
        # Extract rarely used codes data
        # Dynamically extract all codes_below_X_percent keys and sort by threshold value
        codes_below_keys = [k for k in self.codebook_stats_history[0].keys() if k.startswith("codes_below_")]
        # Extract threshold values as floats for sorting
        def extract_threshold(key):
            # key format: 'codes_below_X_percent'
            try:
                return float(key.split("_")[2].replace("percent", ""))
            except Exception:
                return float('inf')
        codes_below_keys_sorted = sorted(codes_below_keys, key=extract_threshold)
        # Build a dict of arrays for each threshold
        codes_below_arrays = {
            key: np.array([stats.get(key, 0) for stats in self.codebook_stats_history])
            for key in codes_below_keys_sorted
        }
        unused_codes = np.array([stats.get('unused_codes', 0) for stats in self.codebook_stats_history])

        # Plot total usage
        axes[0, 0].plot(epochs, total_usage, marker='o')
        axes[0, 0].set_title('Total Codebook Usage')
        axes[0, 0].set_xlabel('Measurement Point')
        axes[0, 0].set_ylabel('Total Usage')
        axes[0, 0].grid(True)
        
        # Plot unused ratio
        axes[0, 1].plot(epochs, unused_ratio, marker='s', color='red')
        axes[0, 1].set_title('Unused Code Ratio')
        axes[0, 1].set_xlabel('Measurement Point')
        axes[0, 1].set_ylabel('Unused Ratio')
        axes[0, 1].grid(True)
        
        # Plot reset counter
        axes[1, 0].plot(epochs, reset_counter, marker='^', color='green')
        axes[1, 0].set_title('Reset Counter')
        axes[1, 0].set_xlabel('Measurement Point')
        axes[1, 0].set_ylabel('Reset Count')
        axes[1, 0].grid(True)
        
        # --- Adaptively plot rarely used codes (stacked area plot) ---
        # Build stacked values: first unused_codes, then each codes_below_X_percent in order
        color_list = [
            'darkred', 'lightcoral', 'orange', 'gold', 'yellow', 'lime', 'cyan', 'skyblue', 'blue', 'purple', 'pink'
        ]
        # Label for 0% (unused)
        axes[1, 1].fill_between(epochs, 0, unused_codes, alpha=0.7, label='0% (Unused)', color=color_list[0])
        stacked_bottom = unused_codes.copy()
        # Now plot each codes_below_X_percent in order, stacking them
        for i, key in enumerate(codes_below_keys_sorted):
            arr = codes_below_arrays[key]
            # The label is the threshold, e.g. 'codes_below_0.5_percent'
            threshold_val = extract_threshold(key)
            if i == 0:
                label = f'Below {threshold_val:.4f}%'
            else:
                prev_threshold_val = extract_threshold(codes_below_keys_sorted[i-1])
                label = f'{prev_threshold_val:.4f}-{threshold_val:.4f}%'
            # Pick color, cycle if too many
            color = color_list[(i+1) % len(color_list)]
            axes[1, 1].fill_between(epochs, stacked_bottom, arr, alpha=0.7, label=label, color=color)
            stacked_bottom = arr
        axes[1, 1].set_title('Code Usage Distribution')
        axes[1, 1].set_xlabel('Measurement Point')
        axes[1, 1].set_ylabel('Number of Codes')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot EMA cluster sizes if available
        if 'ema_cluster_sizes' in self.codebook_stats_history[0]:
            ema_active = [(stats['ema_cluster_sizes'] > 0).sum().item() for stats in self.codebook_stats_history]
            axes[2, 0].plot(epochs, ema_active, marker='d', color='purple')
            axes[2, 0].set_title('EMA Active Clusters')
            axes[2, 0].set_xlabel('Measurement Point')
            axes[2, 0].set_ylabel('Active Clusters')
            axes[2, 0].grid(True)
        else:
            axes[2, 0].text(0.5, 0.5, 'EMA not enabled', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('EMA Active Clusters')
        
        # --- Adaptively plot individual rarely used codes lines ---
        axes[2, 1].plot(epochs, unused_codes, marker='o', label='0% (Unused)', color=color_list[0], linewidth=2)
        for i, key in enumerate(codes_below_keys_sorted):
            arr = codes_below_arrays[key]
            threshold_val = extract_threshold(key)
            label = f'Below {threshold_val:.3f}%'
            color = color_list[(i+1) % len(color_list)]
            axes[2, 1].plot(epochs, arr, marker='o', label=label, color=color, linewidth=2)
        axes[2, 1].set_title('Code Usage Thresholds (Individual Lines)')
        axes[2, 1].set_xlabel('Measurement Point')
        axes[2, 1].set_ylabel('Number of Codes')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_diversity_evolution(self, save_path: str) -> None:
        """Plot codebook diversity metrics over time."""
        if not self.diversity_history or not self.save_tracking_history:
            return
        
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        epochs = list(range(len(self.diversity_history)))
        mean_similarity = [metrics['mean_similarity'] for metrics in self.diversity_history]
        max_similarity = [metrics['max_similarity'] for metrics in self.diversity_history]
        embedding_norm_mean = [metrics['embedding_norm_mean'] for metrics in self.diversity_history]
        embedding_norm_std = [metrics['embedding_norm_std'] for metrics in self.diversity_history]
        
        # Plot mean similarity
        axes[0, 0].plot(epochs, mean_similarity, marker='o', color='blue')
        axes[0, 0].set_title('Mean Embedding Similarity')
        axes[0, 0].set_xlabel('Measurement Point')
        axes[0, 0].set_ylabel('Mean Similarity')
        axes[0, 0].grid(True)
        
        # Plot max similarity
        axes[0, 1].plot(epochs, max_similarity, marker='s', color='red')
        axes[0, 1].set_title('Max Embedding Similarity')
        axes[0, 1].set_xlabel('Measurement Point')
        axes[0, 1].set_ylabel('Max Similarity')
        axes[0, 1].grid(True)
        
        # Plot embedding norm mean
        axes[1, 0].plot(epochs, embedding_norm_mean, marker='^', color='green')
        axes[1, 0].set_title('Mean Embedding Norm')
        axes[1, 0].set_xlabel('Measurement Point')
        axes[1, 0].set_ylabel('Mean Norm')
        axes[1, 0].grid(True)
        
        # Plot embedding norm std
        axes[1, 1].plot(epochs, embedding_norm_std, marker='d', color='purple')
        axes[1, 1].set_title('Embedding Norm Std')
        axes[1, 1].set_xlabel('Measurement Point')
        axes[1, 1].set_ylabel('Norm Std')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, save_path: Optional[str] = None, log_scale: bool = False):
        # First call the parent method to get the standard plots
        super().plot_training_history(save_path, log_scale)
        
        # Add progressive loss plot if we have the necessary data
        if (self.detailed_recon_losses and self.detailed_vq_losses and 
            self.detailed_weighted_regularization_losses):
            
            # Create a new figure for the progressive loss plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Flatten all epochs into single lists
            all_recon_losses = [item for epoch in self.detailed_recon_losses for item in epoch]
            all_vq_losses = [item for epoch in self.detailed_vq_losses for item in epoch]
            all_weighted_regularization_losses = [item for epoch in self.detailed_weighted_regularization_losses for item in epoch]
            
            # Calculate progressive loss components
            # 1. Reconstruction loss only
            recon_only = all_recon_losses
            
            # 2. Reconstruction loss + VQ loss
            recon_plus_vq = [r + v for r, v in zip(all_recon_losses, all_vq_losses)]
            
            # 3. Total loss (reconstruction + VQ + weighted regularization)
            total_loss = [r + v + reg for r, v, reg in zip(
                all_recon_losses, all_vq_losses, all_weighted_regularization_losses)]
            
            # Plot the progressive loss components
            ax.plot(range(len(recon_only)), recon_only, label='1. Reconstruction Loss', color='blue', linewidth=2, alpha=0.8)
            ax.plot(range(len(recon_plus_vq)), recon_plus_vq, label='2. Reconstruction + VQ Loss', color='green', linewidth=2, alpha=0.8)
            ax.plot(range(len(total_loss)), total_loss, label='3. Total Loss (All Components)', color='red', linewidth=2, alpha=0.8)
            
            # Add epoch boundaries
            epoch_boundaries = []
            global_batch_idx = 0
            for epoch_idx, epoch_losses in enumerate(self.detailed_recon_losses):
                epoch_boundaries.append((global_batch_idx, epoch_idx + 1))
                global_batch_idx += len(epoch_losses)
            
            for boundary, epoch_num in epoch_boundaries:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                ax.text(boundary, ax.get_ylim()[1], f'Epoch {epoch_num}', 
                       rotation=90, va='top', ha='right', fontsize=8)
            
            ax.set_title('Progressive Loss Components (Within Epochs)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Measurement Index')
            ax.set_ylabel('Loss Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            if log_scale:
                ax.set_yscale('log')
            
            plt.tight_layout()
            
            # Save the progressive loss plot
            if save_path:
                progressive_path = save_path.replace('.png', '_progressive_loss.png')
                fig.savefig(progressive_path, dpi=300, bbox_inches='tight')
                print(f"Progressive loss plot saved to {progressive_path}")
            
            plt.close(fig)
        
        # Plot batch norm parameter evolution if available
        if self.detailed_bn_param_history and any(len(epoch) > 0 for epoch in self.detailed_bn_param_history):
            # Flatten all epochs into a single list
            all_bn_params = [item for epoch in self.detailed_bn_param_history for item in epoch]
            if all_bn_params:
                weights = np.array([p['weight'] for p in all_bn_params])
                biases = np.array([p['bias'] for p in all_bn_params])
                running_means = np.array([p['running_mean'] for p in all_bn_params])
                running_vars = np.array([p['running_var'] for p in all_bn_params])
                steps = np.arange(len(all_bn_params))
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                for arr, ax, name in zip([weights, biases, running_means, running_vars], axes.flat,
                                         ['Weight', 'Bias', 'Running Mean', 'Running Var']):
                    arr_mean = arr.mean(axis=1)
                    arr_std = arr.std(axis=1)
                    ax.plot(steps, arr_mean, label=f'{name} Mean')
                    ax.fill_between(steps, arr_mean - arr_std, arr_mean + arr_std, alpha=0.3, label=f'{name} ±1 Std')
                    ax.set_title(f'BatchNorm {name} Evolution')
                    ax.set_xlabel('Step')
                    ax.set_ylabel(name)
                    ax.legend()
                    ax.grid(True)
                plt.tight_layout()
                if save_path:
                    bn_fig_path = save_path.replace('.png', '_batchnorm.png')
                    fig.savefig(bn_fig_path, dpi=300, bbox_inches='tight')
                    print(f"BatchNorm parameter evolution plot saved to {bn_fig_path}")
                plt.close(fig)
                # 3D mesh plots for each parameter
                param_arrays = {'Weight': weights, 'Bias': biases, 'Running Mean': running_means, 'Running Var': running_vars}
                for name, arr in param_arrays.items():
                    fig3d = plt.figure(figsize=(10, 7))
                    ax3d = fig3d.add_subplot(111, projection='3d')
                    # arr: (time, dim)
                    T, D = arr.shape
                    X, Y = np.meshgrid(np.arange(D), np.arange(T))
                    Z = arr
                    surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)
                    ax3d.set_title(f'BatchNorm {name} (Dim x Time)')
                    ax3d.set_xlabel('Dimension')
                    ax3d.set_ylabel('Step')
                    ax3d.set_zlabel(name)
                    fig3d.colorbar(surf, shrink=0.5, aspect=10)
                    plt.tight_layout()
                    if save_path:
                        mesh_path = save_path.replace('.png', f'_batchnorm_{name.lower().replace(" ", "_")}_3d.png')
                        fig3d.savefig(mesh_path, dpi=300, bbox_inches='tight')
                        print(f"BatchNorm 3D mesh plot for {name} saved to {mesh_path}")
                    plt.close(fig3d)
        # Plot post-batch-norm input norm evolution if available
        if hasattr(self, 'detailed_post_bn_norm_history') and self.detailed_post_bn_norm_history and any(len(epoch) > 0 for epoch in self.detailed_post_bn_norm_history):
            # Flatten all epochs into a single list
            all_post_bn = [item for epoch in self.detailed_post_bn_norm_history for item in epoch]
            if all_post_bn:
                means = np.array([p['mean'] for p in all_post_bn])
                stds = np.array([p['std'] for p in all_post_bn])
                steps = np.arange(len(all_post_bn))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(steps, means, label='Post-BN Input Norm Mean')
                ax.fill_between(steps, means - stds, means + stds, alpha=0.3, label='±1 Std')
                ax.set_title('Post-BatchNorm Input Norm Evolution')
                ax.set_xlabel('Step')
                ax.set_ylabel('Norm')
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                if save_path:
                    post_bn_fig_path = save_path.replace('.png', '_post_bn_input_norm.png')
                    fig.savefig(post_bn_fig_path, dpi=300, bbox_inches='tight')
                    print(f"Post-BatchNorm input norm plot saved to {post_bn_fig_path}")
                plt.close(fig)