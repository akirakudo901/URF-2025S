# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import traceback
import socket
from datetime import datetime
from torch.autograd.profiler import record_function
import sys
import psutil
import numpy as np

# Add the current directory to the path to import dependencies
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules and functions
from train_utils import compute_reconstruction_loss, create_codebook_usage_heatmap, create_codebook_usage_timeline_plot, sample_and_compute_codebook_usage
from vqvae_gpt2 import GPT2VQVAE
from phone_notification import send_notification

# GPU memory monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: nvidia-ml-py3 not available. Install with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False

TRACK_MEMORY = False
TRACK_IN_EPOCH_MEMORY_EVERY_N = 200
TRACK_IN_EPOCH_MEMORY = False
TRACK_IN_EPOCH_MEMORY_LOGGING = True
SEND_NOTIFICATION = True

# Profiler configuration
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile):
    """Handler for profiler traces - saves Chrome trace and memory timeline."""
    # Prefix for file names
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file
    print("Exporting chrome trace...", end="")
    prof.export_chrome_trace(f"{file_prefix}.json.gz")
    print("DONE!")

    # Construct the memory timeline file
    print("Exporting memory timeline...", end="")
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    print("DONE!")

class TrainingAbortedException(Exception):
    """
    Custom exception raised when training is aborted due to perplexity threshold or other conditions.
    """
    def __init__(self, reason: str, epoch: int, metrics: Dict[str, Any], final_perplexity: Optional[float] = None):
        self.reason = reason
        self.epoch = epoch
        self.metrics = metrics
        self.final_perplexity = final_perplexity
        super().__init__(f"Training aborted at epoch {epoch}: {reason}")

class GPUMemoryMonitor:
    """
    Monitor GPU memory usage using nvidia-ml-py3.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU memory monitor.
        
        Args:
            device_id: GPU device ID to monitor
        """
        self.device_id = device_id
        self.nvml_available = NVML_AVAILABLE
        
        if self.nvml_available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.device_name = pynvml.nvmlDeviceGetName(self.handle)
                print(f"GPU Memory Monitor initialized for: {self.device_name}")
            except Exception as e:
                print(f"Warning: Could not initialize NVML for device {device_id}: {e}")
                self.nvml_available = False
            
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory information.
        
        Returns:
            Dictionary with memory usage in GB
        """
        if not self.nvml_available:
            return {
                'total_gb': 0.0,
                'used_gb': 0.0,
                'free_gb': 0.0,
                'utilization_percent': 0.0
            }
        
        try:
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            total_gb = mem_info.total / (1024**3)
            used_gb = mem_info.used / (1024**3)
            free_gb = mem_info.free / (1024**3)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            utilization_percent = util.memory
            
            return {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'utilization_percent': utilization_percent
            }
        except Exception as e:
            print(f"Warning: Could not get GPU memory info: {e}")
            return {
                'total_gb': 0.0,
                'used_gb': 0.0,
                'free_gb': 0.0,
                'utilization_percent': 0.0
            }
    
    def log_memory_usage(self, stage: str, print_info: bool = True) -> Dict[str, float]:
        """
        Log memory usage at a specific stage.
        
        Args:
            stage: Description of the current stage
            print_info: Whether to print the memory information
            
        Returns:
            Dictionary with memory usage information
        """
        mem_info = self.get_memory_info()
        
        if print_info:
            print(f"GPU Memory Usage ({stage}):")
            print(f"  Total: {mem_info['total_gb']:.2f} GB")
            print(f"  Used: {mem_info['used_gb']:.2f} GB ({mem_info['utilization_percent']:.1f}%)")
            print(f"  Free: {mem_info['free_gb']:.2f} GB")
        
        return mem_info
    
    def get_pytorch_memory_info(self) -> Dict[str, float]:
        """
        Get PyTorch-specific memory information.
        
        Returns:
            Dictionary with PyTorch memory usage in GB
        """
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'max_allocated_gb': 0.0
            }
        
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'max_allocated_gb': max_allocated_gb
        }
    
    def log_pytorch_memory_usage(self, stage: str, print_info: bool = True) -> Dict[str, float]:
        """
        Log PyTorch-specific memory usage.
        
        Args:
            stage: Description of the current stage
            print_info: Whether to print the memory information
            
        Returns:
            Dictionary with PyTorch memory usage information
        """
        pytorch_mem = self.get_pytorch_memory_info()
        
        if print_info:
            print(f"PyTorch Memory Usage ({stage}):")
            print(f"  Allocated: {pytorch_mem['allocated_gb']:.2f} GB")
            print(f"  Reserved: {pytorch_mem['reserved_gb']:.2f} GB")
            print(f"  Max Allocated: {pytorch_mem['max_allocated_gb']:.2f} GB")
        
        return pytorch_mem

class GPT2VQVAETrainer:
    """
    Trainer class for GPT2VQVAE model with configurable hyperparameters and memory optimizations.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 tracking_functions: Optional[Dict[str, Any]] = None, 
                 run_name : Optional[str] = "ANONYM_RUN"):
        """
        Initialize the trainer.
        
        Args:
            model_config: Dictionary containing model hyperparameters
            training_config: Dictionary containing training hyperparameters
            device: Device to train on
            tracking_functions: Optional dict containing custom tracking functions:
                - 'track_codebook_usage': Function to track codebook usage
                - 'save_codebook_plots': Function to save codebook plots
                - 'tracking_enabled': Boolean to enable/disable tracking
            run_name: Optional string for the name of the run, as texted via send_notification.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.run_name = run_name
        
        # Initialize training start time for checkpoint notifications
        self.training_start_time = None
        
        # Set up tracking functions (default to base class methods)
        if tracking_functions is None:
            tracking_functions = {}
        
        self.track_codebook_usage_func = tracking_functions.get('track_codebook_usage', self._default_track_codebook_usage)
        self.save_codebook_plots_func = tracking_functions.get('save_codebook_plots', self._default_save_codebook_plots)
        self.tracking_enabled = tracking_functions.get('tracking_enabled', True)
        
        # Initialize GPU memory monitor
        if device.startswith("cuda"):
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            self.memory_monitor = GPUMemoryMonitor(device_id)
        else:
            self.memory_monitor = None
        
        # Memory optimization settings
        self.use_mixed_precision = training_config.get('use_mixed_precision', True)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.use_gradient_checkpointing = training_config.get('use_gradient_checkpointing', True)
        
        self.ensure_numeric_types(self.model_config)
        self.ensure_numeric_types(self.training_config)

        # Memory monitoring
        self.memory_stats = []
        
        # Log memory before model initialization
        if self.memory_monitor:
            self.log_memory_usage("before_model_init")
        
        # Initialize model
        self.model = GPT2VQVAE(**model_config).to(device)
        
        # Log memory after model weights loaded to GPU
        if self.memory_monitor:
            self.log_memory_usage("after_model_weights_loaded")
            
        # Enable gradient checkpointing if specified
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
            # Log gradient checkpointing status
            if hasattr(self.model, 'get_gradient_checkpointing_status'):
                status = self.model.get_gradient_checkpointing_status()
                print(f"Gradient checkpointing status:")
                print(f"  Model enabled: {status['model_enabled']}")
                print(f"  Encoder enabled: {status['encoder_enabled']}")
                print(f"  Decoder enabled: {status['decoder_enabled']}")
            elif hasattr(self.model, 'is_gradient_checkpointing_enabled'):
                print(f"Model gradient checkpointing: {self.model.is_gradient_checkpointing_enabled()}")
            else:
                print("Model gradient checkpointing: Not supported by this model")
        
        # Log memory before optimizer initialization
        if self.memory_monitor:
            self.log_memory_usage("before_optimizer_init")
            
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        
        # Log memory after optimizer initialization
        if self.memory_monitor:
            self.log_memory_usage("after_optimizer_init")
        
        # Initialize mixed precision training
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
            # self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Initialize learning rate scheduler
        if training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=training_config.get('pad_token_id', 50256))
        
        # Training history - epoch-level metrics
        self.train_losses = []
        self.val_losses = []
        self.vq_losses = []
        self.perplexities = []
        self.recon_losses = []
        self.val_recon_losses = []
        self.epoch_chain_embeddings = []   # List of np.arrays, one per epoch
        
        # Training history - detailed metrics within epochs
        self.detailed_train_losses = []  # List of lists: [epoch_1_metrics, epoch_2_metrics, ...]
        self.detailed_vq_losses = []
        self.detailed_perplexities = []
        self.detailed_batch_indices = []  # List of lists: [epoch_1_indices, epoch_2_indices, ...]
        self.detailed_recon_losses = []
        # New: VQ input norm tracking (per measurement interval)
        self.detailed_vq_input_means = []  # List of lists: [epoch_1_means, epoch_2_means, ...]
        self.detailed_vq_input_stds = []   # List of lists: [epoch_1_stds, epoch_2_stds, ...]
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Initialize gradient checkpointing as disabled by default
        self._gradient_checkpointing_enabled = False
        
        # Codebook usage tracking (legacy - now handled by tracking functions)
        self.codebook_tracking_enabled = self.training_config.get('codebook_tracking_enabled', True)
        self.codebook_sample_size = self.training_config.get('codebook_sample_size', 100)
        self.codebook_history = []  # List of count numpy arrays over time
        self.codebook_perplexities = []  # List of perplexities over time
        self.codebook_measurement_points = []  # List of measurement point indices
        
        if self.tracking_enabled:
            print(f"Codebook tracking enabled with custom functions")
        else:
            print("Codebook tracking disabled")
        
        # Profiler configuration
        self.profiler_enabled = training_config.get('profiler_enabled', False)
        self.profiler_schedule = training_config.get('profiler_schedule', {
            'wait': 0,      
            'warmup': 1,    
            'active': 2,    
            'repeat': 1     
        })
        self.profiler_activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        if self.profiler_enabled:
            print("PyTorch profiler enabled")
            print(f"Profiler schedule: {self.profiler_schedule}")
    
    def _default_track_codebook_usage(self, dataset: Any, measurement_point: int) -> None:
        """Default codebook tracking function for base trainer."""
        if not self.codebook_tracking_enabled:
            return
        
        try:
            # Sample and compute codebook usage
            counts, perplexity = sample_and_compute_codebook_usage(
                self.model, 
                dataset, 
                self.codebook_sample_size, 
                self.device,
                use_vq=self.training_config.get('use_vq', True)
            )
            
            # Store results - convert to numpy array to prevent GPU memory accumulation
            self.codebook_history.append(counts.cpu().detach().numpy())
            self.codebook_perplexities.append(perplexity)
            self.codebook_measurement_points.append(measurement_point)
            
            # Print current statistics
            unique_codes = (counts > 0).sum().item()
            print(f"\nCodebook tracking (point {measurement_point}): "
                  f"Unique codes: {unique_codes}/{self.model.vector_quantizer.num_embeddings} "
                  f"({unique_codes/self.model.vector_quantizer.num_embeddings*100:.1f}%), "
                  f"Perplexity: {perplexity:.2f}")
            
            # Explicitly delete GPU tensors to prevent memory accumulation
            del counts
            
        except Exception as e:
            print(f"Warning: Failed to track codebook usage: {e}")
        finally:
            # Ensure cleanup even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _default_save_codebook_plots(self, save_dir: str, epoch: int) -> None:
        """Default codebook plotting function for base trainer."""
        if not self.codebook_tracking_enabled or not self.codebook_history:
            return
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save current heatmap
            current_counts = self.codebook_history[-1]
            heatmap_path = os.path.join(save_dir, f"codebook_usage_epoch_{epoch}.png")
            create_codebook_usage_heatmap(
                current_counts,
                num_embeddings=self.model.vector_quantizer.num_embeddings,
                title=f"Codebook Usage - Epoch {epoch}",
                save_path=heatmap_path
            )
            
            # Save timeline plot
            timeline_path = os.path.join(save_dir, f"codebook_usage_timeline_epoch_{epoch}.png")
            create_codebook_usage_timeline_plot(
                self.codebook_history,
                num_embeddings=self.model.vector_quantizer.num_embeddings,
                measurement_points=self.codebook_measurement_points,
                title=f"Codebook Usage Over Time - Up to Epoch {epoch}",
                save_path=timeline_path
            )
            
            print(f"Codebook tracking plots saved to {save_dir}")
            
            # Clear cache after plotting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to save codebook tracking plots: {e}")
    
    # Ensure all numeric values are properly typed
    def ensure_numeric_types(self, config_dict):
        """Ensure all numeric values in config are properly typed."""
        for key, value in config_dict.items():
            if isinstance(value, str):
                try:
                    if 'e' in value.lower() or '.' in value:
                        config_dict[key] = float(value)
                    else:
                        config_dict[key] = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
            elif isinstance(value, bool):
                config_dict[key] = bool(value)
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage using the GPU memory monitor and CPU RAM usage."""
        # CPU RAM usage
        ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        if self.memory_monitor:
            # Log both GPU and PyTorch memory usage
            gpu_mem = self.memory_monitor.log_memory_usage(stage, print_info=False)
            pytorch_mem = self.memory_monitor.log_pytorch_memory_usage(stage, print_info=False)
            self.memory_stats.append({
                'stage': stage,
                'gpu_total_gb': gpu_mem['total_gb'],
                'gpu_used_gb': gpu_mem['used_gb'],
                'gpu_free_gb': gpu_mem['free_gb'],
                'gpu_utilization_percent': gpu_mem['utilization_percent'],
                'pytorch_allocated_gb': pytorch_mem['allocated_gb'],
                'pytorch_reserved_gb': pytorch_mem['reserved_gb'],
                'pytorch_max_allocated_gb': pytorch_mem['max_allocated_gb'],
                'cpu_ram_gb': ram_gb
            })
            print(f"Memory Usage ({stage}):")
            print(f"  GPU: {gpu_mem['used_gb']:.2f}GB used / {gpu_mem['total_gb']:.2f}GB total ({gpu_mem['utilization_percent']:.1f}%)")
            print(f"  PyTorch: {pytorch_mem['allocated_gb']:.2f}GB allocated, {pytorch_mem['reserved_gb']:.2f}GB reserved")
            print(f"  CPU RAM: {ram_gb:.2f}GB used")
        else:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                self.memory_stats.append({
                    'stage': stage,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'max_allocated_gb': max_allocated,
                    'cpu_ram_gb': ram_gb
                })
                print(f"Memory usage ({stage}): {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
                print(f"  CPU RAM: {ram_gb:.2f}GB used")
            else:
                self.memory_stats.append({
                    'stage': stage,
                    'cpu_ram_gb': ram_gb
                })
                print(f"Memory usage ({stage}): CPU RAM: {ram_gb:.2f}GB used")
    

    
    def create_data_loader(self, 
                          dataset: torch.utils.data.Dataset,
                          batch_size: int,
                          shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for training/validation with memory optimizations.
        
        Args:
            dataset: PyTorch dataset (e.g., from random_split)
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the dataset
        """
        # Use dynamic batching if specified
        if self.training_config.get('use_dynamic_batching', False):
            max_tokens = self.training_config.get('max_tokens_per_batch', 8192)
            # Create sampler with proper type handling
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)  # type: ignore
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)  # type: ignore
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
            return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            
    def train_epoch(self, train_loader: DataLoader, num_measurements_per_epoch: int, current_epoch: int = 0, detailed_metrics_callback=None) -> Dict[str, Any]:
        """
        Train for one epoch with memory optimizations.
        
        Args:
            train_loader: Training data loader
            num_measurements_per_epoch: Number of equally spaced measurements to log during the epoch
            current_epoch: Current epoch number (for exception handling)
            detailed_metrics_callback: Optional function to call at each measurement interval for custom tracking
            
        Returns:
            Dictionary containing training metrics with both detailed and average metrics
        """
        self.model.train()
        total_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        accumulation_steps = 0
        
        # Calculate measurement intervals
        total_batches = len(train_loader)
        measurement_interval = max(1, total_batches // num_measurements_per_epoch)
        
        # Storage for detailed metrics
        detailed_losses = []
        detailed_vq_losses = []
        detailed_perplexities = []
        detailed_batch_indices = []
        detailed_recon_losses = []
        # New: VQ input norm tracking for this epoch
        detailed_vq_input_means = []
        detailed_vq_input_stds = []
        
        # Perplexity threshold monitoring
        perplexity_threshold = self.training_config.get('perplexity_threshold', 1.5)
        perplexity_window_size = self.training_config.get('perplexity_window_size', 20)
        recent_perplexities = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (prompts, cots, prompt_masks, cot_masks) in enumerate(progress_bar):
            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                self.log_memory_usage(f"before batch {batch_idx}")
            
            # Profiler step if enabled
            # if hasattr(self, 'profiler_enabled') and self.profiler_enabled and self.profiler_context is not None:
            #     self.profiler_context.step()
            # Move to device
            prompts = prompts.to(self.device, non_blocking=True)
            cots = cots.to(self.device, non_blocking=True)
            prompt_masks = prompt_masks.to(self.device, non_blocking=True)
            cot_masks = cot_masks.to(self.device, non_blocking=True)

            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                self.log_memory_usage(f"after data loading, batch {batch_idx}")
            
            # Forward pass and loss calculation
            if TRACK_MEMORY:
                with record_function("## forward_pass ##"):
                    total_loss_batch, recon_loss, vq_loss, perplexity, _, debug_stats = self._forward_pass(
                        prompts, cots, prompt_masks, cot_masks
                    )
            else:
                total_loss_batch, recon_loss, vq_loss, perplexity, _, debug_stats = self._forward_pass(
                    prompts, cots, prompt_masks, cot_masks
                )
            
            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                self.log_memory_usage(f"after forward pass, batch {batch_idx}")
            
            # Scale loss and backward pass
            if TRACK_MEMORY:
                with record_function("## backward_pass ##"):
                    scaled_loss = total_loss_batch / self.gradient_accumulation_steps
                    if self.use_mixed_precision and self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
            else:
                scaled_loss = total_loss_batch / self.gradient_accumulation_steps
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            
            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                self.log_memory_usage(f"after backward pass, batch {batch_idx}")

            accumulation_steps += 1
            
            # Update weights every gradient_accumulation_steps
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                if TRACK_MEMORY:
                    with record_function("## optimizer_step ##"):
                        self._update_weights()
                else:
                    self._update_weights()
                # TODO DEBUG PURPOSE
                if TRACK_IN_EPOCH_MEMORY and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                    self.log_memory_usage(f"after weight update, batch {batch_idx}")

            
            # Extract scalar values and detach tensors to prevent memory accumulation
            total_loss_batch_item = total_loss_batch.item()
            recon_loss_item = recon_loss.item()
            vq_loss_item = vq_loss.item()
            perplexity_item = perplexity.item()
            
            # Update metrics
            total_loss += total_loss_batch_item
            total_recon_loss += recon_loss_item
            total_vq_loss += vq_loss_item
            total_perplexity += perplexity_item
            num_batches += 1
            
            # Update perplexity monitoring
            recent_perplexities.append(perplexity_item)
            if len(recent_perplexities) > perplexity_window_size:
                recent_perplexities.pop(0)
            
            # Check perplexity threshold
            if len(recent_perplexities) >= perplexity_window_size:
                avg_perplexity = sum(recent_perplexities) / len(recent_perplexities)
                if avg_perplexity < perplexity_threshold:
                    print(f"\n🎯 Perplexity threshold reached! Average perplexity over last {perplexity_window_size} steps: {avg_perplexity:.4f} < {perplexity_threshold}")
                    print(f"Training aborted.")
                    
                    # Calculate final metrics
                    final_metrics = {
                        'detailed_losses': detailed_losses,
                        'detailed_recon_losses': detailed_recon_losses,
                        'detailed_vq_losses': detailed_vq_losses,
                        'detailed_perplexities': detailed_perplexities,
                        'detailed_batch_indices': detailed_batch_indices,
                        'avg_loss': total_loss / num_batches,
                        'avg_recon_loss': total_recon_loss / num_batches if num_batches > 0 else 0.0,
                        'avg_vq_loss': total_vq_loss / num_batches,
                        'avg_perplexity': total_perplexity / num_batches,
                        'aborted': True,
                        'abort_reason': f'perplexity_threshold_{perplexity_threshold}',
                        'final_avg_perplexity': avg_perplexity
                    }
                    
                    # Raise exception with all the necessary information
                    raise TrainingAbortedException(
                        reason=f'perplexity_threshold_{perplexity_threshold}',
                        epoch=current_epoch + 1,  # Current epoch number
                        metrics=final_metrics,
                        final_perplexity=avg_perplexity
                    )

            # Log detailed metrics at regular intervals
            if batch_idx % measurement_interval == 0:
                # TODO DEBUG PURPOSE
                if TRACK_IN_EPOCH_MEMORY_LOGGING and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                    self.log_memory_usage(f"before detailed info, batch {batch_idx}")

                detailed_losses.append(total_loss_batch_item)
                detailed_recon_losses.append(recon_loss_item)
                detailed_vq_losses.append(vq_loss_item)
                detailed_perplexities.append(perplexity_item)
                detailed_batch_indices.append(batch_idx)
                # New: Use debug_stats from model forward for VQ input norm
                if debug_stats is not None:
                    detailed_vq_input_means.append(debug_stats['vq_input_norm_mean'])
                    detailed_vq_input_stds.append(debug_stats['vq_input_norm_std'])
                # Call the detailed metrics callback if provided
                if detailed_metrics_callback is not None:
                    detailed_metrics_callback(
                        batch_idx=batch_idx,
                        model=self.model,
                        trainer=self,
                        prompts=prompts,
                        cots=cots,
                        prompt_masks=prompt_masks,
                        cot_masks=cot_masks,
                        debug_stats=debug_stats,
                        current_epoch=current_epoch
                    )
            else:
                del debug_stats
                
                # Track codebook usage at measurement intervals
                if self.tracking_enabled:

                    # TODO DEBUG PURPOSE
                    if TRACK_IN_EPOCH_MEMORY_LOGGING and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                        self.log_memory_usage(f"before getting the dataset, batch {batch_idx}")

                    # Get the dataset from the data loader
                    dataset: Any = train_loader.dataset
                    if hasattr(dataset, 'dataset'):  # Handle SubsetRandomSampler case
                        dataset = dataset.dataset

                    # TODO DEBUG PURPOSE
                    if TRACK_IN_EPOCH_MEMORY_LOGGING and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                        self.log_memory_usage(f"after getting the dataset, batch {batch_idx}")

                    self.track_codebook_usage_func(dataset, batch_idx)

                    # TODO DEBUG PURPOSE
                    if TRACK_IN_EPOCH_MEMORY_LOGGING and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                        self.log_memory_usage(f"after running codebook usage function, batch {batch_idx}")
                    
                
                # TODO DEBUG PURPOSE
                if TRACK_IN_EPOCH_MEMORY_LOGGING and (batch_idx + 1) % TRACK_IN_EPOCH_MEMORY_EVERY_N == 0:
                    self.log_memory_usage(f"after detailed info, batch {batch_idx}")
            
            # Update progress bar
            acc_step = (accumulation_steps % self.gradient_accumulation_steps) + 1
            current_avg_perplexity = sum(recent_perplexities) / len(recent_perplexities) if recent_perplexities else 0
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch_item:.4f}",
                'vq_loss': f"{vq_loss_item:.4f}",
                'perplexity': f"{perplexity_item:.2f}",
                'avg_perplexity': f"{current_avg_perplexity:.2f}",
                'accum_steps': f"{acc_step}/{self.gradient_accumulation_steps}"
            })

            # Explicitly delete intermediate tensors to prevent memory accumulation
            del total_loss_batch, recon_loss, vq_loss, perplexity, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate averages
        avg_metrics = self._get_average_metrics(total_loss, total_vq_loss, total_perplexity, num_batches)
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
        # Store VQ input stats for this epoch
        self.detailed_vq_input_means.append(detailed_vq_input_means)
        self.detailed_vq_input_stds.append(detailed_vq_input_stds)
        
        # Return both detailed and average metrics
        return {
            'detailed_losses': detailed_losses,
            'detailed_recon_losses': detailed_recon_losses,
            'detailed_vq_losses': detailed_vq_losses,
            'detailed_perplexities': detailed_perplexities,
            'detailed_batch_indices': detailed_batch_indices,
            'avg_loss': avg_metrics['loss'],
            'avg_recon_loss': avg_recon_loss,
            'avg_vq_loss': avg_metrics['vq_loss'],
            'avg_perplexity': avg_metrics['perplexity'],
            'aborted': False
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model with memory optimizations.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        model_was_training = self.model.training 
        
        self.model.eval()
        total_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (prompts, cots, prompt_masks, cot_masks) in enumerate(tqdm(val_loader, desc="Validation")):
                # Move to device
                prompts = prompts.to(self.device, non_blocking=True)
                cots = cots.to(self.device, non_blocking=True)
                prompt_masks = prompt_masks.to(self.device, non_blocking=True)
                cot_masks = cot_masks.to(self.device, non_blocking=True)
                
                # Forward pass and loss calculation
                if TRACK_MEMORY:
                    with record_function("## validation_forward ##"):
                        total_loss_batch, recon_loss, vq_loss, perplexity, _, _ = self._forward_pass(
                            prompts, cots, prompt_masks, cot_masks
                        )
                else:
                    total_loss_batch, recon_loss, vq_loss, perplexity, _, _ = self._forward_pass(
                        prompts, cots, prompt_masks, cot_masks
                    )
                
                # Update metrics
                total_loss += total_loss_batch.item()
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
                total_perplexity += perplexity.item()
                num_batches += 1

                del prompts, cots, prompt_masks, cot_masks
                
        
        # Calculate averages
        avg_metrics = self._get_average_metrics(total_loss, total_vq_loss, total_perplexity, num_batches)
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0

        if model_was_training:
            self.model.train()

        return {
            'loss': avg_metrics['loss'],
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_metrics['vq_loss'],
            'perplexity': avg_metrics['perplexity']
        }
    
    def _forward_pass(self, prompts, cots, prompt_masks, cot_masks):
        """Helper function for forward pass and loss calculation"""
        if self.use_mixed_precision:
            with autocast('cuda'):
                _, output_logits, vq_loss, perplexity, indices, debug_stats = self.model(
                    prompt=prompts,
                    cot_sequences=cots,
                    cot_mask=cot_masks,
                    prompt_mask=prompt_masks,
                    inference=False,
                    quantize_cot_only=self.training_config.get('quantize_cot_only', True)
                )
                recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
                total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        else:
            _, output_logits, vq_loss, perplexity, indices, debug_stats = self.model(
                prompt=prompts,
                cot_sequences=cots,
                cot_mask=cot_masks,
                prompt_mask=prompt_masks,
                inference=False,
                quantize_cot_only=self.training_config.get('quantize_cot_only', True)
            )
            recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
            total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
            
        return total_loss_batch, recon_loss, vq_loss, perplexity, indices, debug_stats
    
    def _update_weights(self):
        """Helper function for updating weights"""
        if self.use_mixed_precision and self.scaler is not None:
            if self.training_config.get('gradient_clip', 1.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip']
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.training_config.get('gradient_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip']
                )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _get_average_metrics(self, total_loss, total_vq_loss, total_perplexity, num_batches):
        """Helper function for calculating average metrics"""
        return {
            'loss': total_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches,
            'perplexity': total_perplexity / num_batches
        }
    
    def send_training_start_phone_notification(self) -> bool:
        try:
            # Format the message
            message = f"Training Started for {self.run_name}!\n"
            return send_notification(message)
        except Exception as e:
            print(f"Error sending training completion phone notification: {e}")
            return False

    def send_checkpoint_notification(self, epoch: int, is_best: bool = False) -> bool:
        """
        Send a short notification when a checkpoint is saved.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            if self.training_start_time is None:
                return False
                
            # Calculate time elapsed
            current_time = datetime.now()
            elapsed = (current_time - self.training_start_time).total_seconds()
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            
            # Format the message
            if is_best:
                message = f"💾 Best checkpoint saved for {self.run_name}!\nEpoch {epoch}, Time: {hours:02d}:{minutes:02d}"
            else:
                message = f"💾 Checkpoint saved for {self.run_name}\nEpoch {epoch}, Time: {hours:02d}:{minutes:02d}"
            
            # Send the phone notification
            return send_notification(message)
            
        except Exception as e:
            print(f"Error sending checkpoint notification: {e}")
            return False

    def send_training_completion_phone_notification(self, final_metrics: Dict[str, float], 
                                                  training_duration: Optional[float] = None,
                                                  aborted: bool = False) -> bool:
        """
        Send a formatted phone notification when training is complete.
        
        Args:
            final_metrics: Dictionary containing final training metrics
            training_duration: Duration of training in seconds (optional)
            aborted: Whether training was aborted (default: False)
            auth_info_path: Path to the auth info YAML file (optional)
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Format the message
            if aborted:
                message = f"🛑 Training Aborted for {self.run_name}!\n\n"
            else:
                message = f"🎉 Training Complete for {self.run_name}!\n\n"
            
            # Add basic metrics
            if 'loss' in final_metrics:
                message += f"Final Train Loss: {final_metrics['loss']:.4f}\n"
            if 'recon_loss' in final_metrics:
                message += f"Final Train Recon Loss: {final_metrics['recon_loss']:.4f}\n"
            
            if 'val_loss' in final_metrics:
                message += f"Final Val Loss: {final_metrics['val_loss']:.4f}\n"
            if 'val_recon_loss' in final_metrics:
                message += f"Final Val Recon Loss: {final_metrics['val_recon_loss']:.4f}\n"
            
            if 'vq_loss' in final_metrics:
                message += f"VQ Loss: {final_metrics['vq_loss']:.4f}\n"
            if 'perplexity' in final_metrics:
                message += f"Perplexity: {final_metrics['perplexity']:.2f}\n"
            
            # Add training duration if provided
            if training_duration is not None:
                hours = int(training_duration // 3600)
                minutes = int((training_duration % 3600) // 60)
                seconds = int(training_duration % 60)
                message += f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
            
            # Add additional information
            message += f"Model: {self.model_config.get('model_type', 'GPT2VQVAE')}\n"
            message += f"Device: {self.device}\n"
            message += f"Best Val Loss: {self.best_val_loss:.4f}\n"
            message += f"Epochs: {len(self.train_losses)}\n"
            
            # Send the phone notification
            return send_notification(message)
            
        except Exception as e:
            print(f"Error sending training completion phone notification: {e}")
            return False
    

    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None, remove_other_best_models: bool = True, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
            checkpoint_path: Path to save the checkpoint (optional)
            remove_other_best_models: If True, remove other best model checkpoints (default: True)
            **kwargs: Additional data to append to the checkpoint
        """
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vq_losses': self.vq_losses,
            'perplexities': self.perplexities,
            'recon_losses': self.recon_losses,
            'val_recon_losses': self.val_recon_losses,
            'detailed_train_losses': self.detailed_train_losses,
            'detailed_vq_losses': self.detailed_vq_losses,
            'detailed_perplexities': self.detailed_perplexities,
            'detailed_batch_indices': self.detailed_batch_indices,
            'detailed_recon_losses': self.detailed_recon_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        # Add any additional data passed as kwargs
        checkpoint.update(kwargs)
        
        # Save best model if this is the best so far
        if is_best:
            if remove_other_best_models:
                # Remove any existing best model checkpoints
                for file in os.listdir(checkpoint_dir):
                    if 'best_model' in file and file.endswith('.pt'):
                        os.remove(os.path.join(checkpoint_dir, file))
            if checkpoint_path is None:
                best_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            else:
                best_path = checkpoint_path
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved (epoch {epoch}) with validation loss: {metrics['loss']:.4f}")
        else:
            # Save regular checkpoint
            if checkpoint_path is None:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Validate model and training configurations
        def _check_config_mismatch(checkpoint_config, current_config, config_type):
            """Helper function to check and report configuration mismatches."""
            if checkpoint_config != current_config:
                print(f"Warning: {config_type} configuration mismatch detected!")
                print(f"Differences between checkpoint and current {config_type} config:")
                for key in set(checkpoint_config.keys()) | set(current_config.keys()):
                    if key not in checkpoint_config:
                        print(f"  {key}: missing in checkpoint, current: {current_config[key]}")
                    elif key not in current_config:
                        print(f"  {key}: missing in current, checkpoint: {checkpoint_config[key]}")
                    elif checkpoint_config[key] != current_config[key]:
                        print(f"  {key}: checkpoint={checkpoint_config[key]}, current={current_config[key]}")
                print(f"Continuing with current {config_type} configuration...")

        if 'model_config' in checkpoint:
            _check_config_mismatch(checkpoint['model_config'], self.model_config, "model")
        
        if 'training_config' in checkpoint:
            _check_config_mismatch(checkpoint['training_config'], self.training_config, "training")
    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Check and update scheduler parameters if needed
            loaded_T_max = self.scheduler.T_max
            loaded_eta_min = self.scheduler.eta_min
            config_T_max = self.training_config['num_epochs']
            config_eta_min = self.training_config.get('min_lr', 1e-6)
            mismatch = False
            if loaded_T_max != config_T_max:
                print(f"Warning: Scheduler T_max from checkpoint ({loaded_T_max}) does not match current config ({config_T_max}). Overriding to config value.")
                self.scheduler.T_max = config_T_max
                mismatch = True
            if loaded_eta_min != config_eta_min:
                print(f"Warning: Scheduler eta_min from checkpoint ({loaded_eta_min}) does not match current config ({config_eta_min}). Overriding to config value.")
                self.scheduler.eta_min = config_eta_min
                mismatch = True
            if mismatch:
                print("Scheduler parameters have been updated to match the current training configuration.")
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.vq_losses = checkpoint.get('vq_losses', [])
        self.perplexities = checkpoint.get('perplexities', [])
        self.recon_losses = checkpoint.get('recon_losses', [])
        self.val_recon_losses = checkpoint.get('val_recon_losses', [])
        
        self.detailed_train_losses = checkpoint.get('detailed_train_losses', [])
        self.detailed_vq_losses = checkpoint.get('detailed_vq_losses', [])
        self.detailed_perplexities = checkpoint.get('detailed_perplexities', [])
        self.detailed_batch_indices = checkpoint.get('detailed_batch_indices', [])
        self.detailed_recon_losses = checkpoint.get('detailed_recon_losses', [])
        
        # Restore best_val_loss if present
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print("Checkpoint loaded successfully. Configuration validation completed.")
        
        return checkpoint
    
    def _train_with_loaders(self, 
                           train_loader: DataLoader,
                           test_loader: DataLoader,
                           resume_from: Optional[str] = None,
                           num_measurements_per_epoch: Optional[int] = None,
                           seed: int = 42):
        """
        Core training logic that works with pre-created data loaders.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            resume_from: Path to checkpoint to resume from
            num_measurements_per_epoch: Number of metrics saved per epoch
            seed: Random seed for reproducibility
        """
        # Start timing
        training_start_time = datetime.now()
        self.training_start_time = training_start_time  # Store for checkpoint notifications

        # FOR DEBUG
        if TRACK_MEMORY:
            torch.cuda.memory._record_memory_history(max_entries=10000)

        # Log initial memory usage
        self.log_memory_usage("training_start")
        
        # Validate dataset sizes
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)

        if train_size == 0:
            raise Exception(f"The training dataset has size 0.")
        elif test_size == 0:
            raise Exception(f"The test dataset has size 0.")
        
        if num_measurements_per_epoch is None:
            num_measurements_per_epoch = self.training_config.get("num_measurements_per_epoch", 25)
            print(f"Logging {num_measurements_per_epoch} measurements per epoch as per training config (or if not given, default)")
        else:
            print(f"Logging {num_measurements_per_epoch} measurements per epoch as given as parameter")
        
        # Ensure num_measurements_per_epoch is an integer for type checking
        assert num_measurements_per_epoch is not None
        num_measurements_per_epoch = int(num_measurements_per_epoch)
        
        print(f"Training samples: {train_size}")
        print(f"Test samples: {test_size}")
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = len(self.train_losses)
        
        # Training loop with profiler
        try:
            # Send message that training has started
            if SEND_NOTIFICATION:
                self.send_training_start_phone_notification()

            # Set up profiler for this epoch if enabled
            self.profiler_context = None
            if self.profiler_enabled:
                print(f"Start profiler tracking.")
                self.profiler_context = torch.profiler.profile(
                    activities=self.profiler_activities,
                    schedule=torch.profiler.schedule(**self.profiler_schedule),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    on_trace_ready=trace_handler,
                )
                self.profiler_context.__enter__()

            for epoch in range(start_epoch, self.training_config['num_epochs']):
                print(f"\nEpoch {epoch + 1}/{self.training_config['num_epochs']}")
                print("-" * 50)

                # Take a profiler step
                if hasattr(self, 'profiler_enabled') and self.profiler_enabled and self.profiler_context is not None:
                    self.profiler_context.step()
            
                
                # Log memory before epoch
                self.log_memory_usage(f"epoch_{epoch+1}_start")
                
                # Train with resume support
                train_metrics = self.train_epoch(
                    train_loader,
                    num_measurements_per_epoch,  # This is guaranteed to be int from earlier logic
                    epoch
                )
                
                # Log memory after training
                self.log_memory_usage(f"epoch_{epoch+1}_after_train")
                
                # Test
                test_metrics = self.validate(test_loader)
                
                # Log memory after testing
                self.log_memory_usage(f"epoch_{epoch+1}_after_test")
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Store epoch-level metrics
                self.train_losses.append(train_metrics['avg_loss'])
                self.recon_losses.append(train_metrics['avg_recon_loss'])
                self.val_losses.append(test_metrics['loss'])
                self.val_recon_losses.append(test_metrics['recon_loss'])
                self.vq_losses.append(train_metrics['avg_vq_loss'])
                self.perplexities.append(train_metrics['avg_perplexity'])
                
                # Store detailed metrics
                self.detailed_train_losses.append(train_metrics['detailed_losses'])
                self.detailed_recon_losses.append(train_metrics['detailed_recon_losses'])
                self.detailed_vq_losses.append(train_metrics['detailed_vq_losses'])
                self.detailed_perplexities.append(train_metrics['detailed_perplexities'])
                self.detailed_batch_indices.append(train_metrics['detailed_batch_indices'])
                
                # Print metrics
                print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
                print(f"Train Recon Loss: {train_metrics['avg_recon_loss']:.4f}")
                print(f"Val Loss: {test_metrics['loss']:.4f}")
                print(f"Val Recon Loss: {test_metrics['recon_loss']:.4f}")
                print(f"VQ Loss: {train_metrics['avg_vq_loss']:.4f}")
                print(f"Perplexity: {train_metrics['avg_perplexity']:.2f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save checkpoint
                is_best = self.is_new_best(test_metrics['loss'])
                if is_best:
                    self.update_best(test_metrics['loss'])
                    if TRACK_MEMORY:
                        with record_function("## save_checkpoint ##"):
                            self.save_checkpoint(epoch + 1, test_metrics, True)
                    else:
                        self.save_checkpoint(epoch + 1, test_metrics, True)
                    # Send checkpoint notification
                    if SEND_NOTIFICATION:
                        self.send_checkpoint_notification(epoch + 1, is_best=True)
                    # save training visualizations when saving a checkpoint
                    save_training_visualizations(self, prefix=f"epoch_{epoch+1}")
                
                if (epoch + 1) % self.training_config.get('save_every', 5) == 0:
                    if TRACK_MEMORY:
                        with record_function("## save_checkpoint ##"):
                            self.save_checkpoint(epoch + 1, test_metrics, False)
                    else:
                        self.save_checkpoint(epoch + 1, test_metrics, False)
                    # Send checkpoint notification
                    if SEND_NOTIFICATION:
                        self.send_checkpoint_notification(epoch + 1, is_best=False)
                    # save training visualizations when saving a checkpoint
                    save_training_visualizations(self, prefix=f"epoch_{epoch+1}")
                
                    self.log_memory_usage(f"epoch_{epoch+1}_after_saving")
                
                
                # Save codebook tracking plots
                if self.tracking_enabled:
                    if TRACK_MEMORY:
                        with record_function("## save_codebook_plots ##"):
                            checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
                            codebook_dir = os.path.join(checkpoint_dir, 'codebook_tracking')
                            self.save_codebook_plots_func(codebook_dir, epoch + 1)
                    else:
                        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
                        codebook_dir = os.path.join(checkpoint_dir, 'codebook_tracking')
                        self.save_codebook_plots_func(codebook_dir, epoch + 1)
                
                    self.log_memory_usage(f"epoch_{epoch+1}_after_saving_codebook")
                
                # New: Track chain embeddings at end of epoch
                self.epoch_chain_embeddings.append(self.model.chain_embeddings.weight.detach().cpu().numpy())
                
                # Clear cache after each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # DEBUG
                if TRACK_MEMORY:
                    # after the third epoch and repeated
                    if (epoch + 1) % 3 == 0:
                        try:
                            torch.cuda.memory._dump_snapshot(f"training_cuda_memory_tracking_epoch{epoch+1}.pickle")
                        except Exception as e:
                            print("Failed to log memory tracking.")
            
            # Clean up profiler if active
            if self.profiler_context is not None:
                self.profiler_context.__exit__(None, None, None)
                self.profiler_context = None
                print(f"Profiler completed for epoch {epoch}")

            if TRACK_MEMORY:
                # Stop recording memory snapshot history.
                torch.cuda.memory._record_memory_history(enabled=None)

        
        except TrainingAbortedException as e:
            # Handle aborted training
            print(f"\n🛑 Training aborted: {e.reason}")
            if e.final_perplexity is not None:
                print(f"Final average perplexity: {e.final_perplexity:.4f}")
            
            # Store the current epoch's metrics even though training was aborted
            self.train_losses.append(e.metrics['avg_loss'])
            self.vq_losses.append(e.metrics['avg_vq_loss'])
            self.perplexities.append(e.metrics['avg_perplexity'])
            self.recon_losses.append(e.metrics['avg_recon_loss'])
            
            # Add a dummy validation loss for plotting purposes (use training loss as proxy)
            self.val_losses.append(e.metrics['avg_loss'])
            self.val_recon_losses.append(e.metrics['avg_loss'])
            
            # Store detailed metrics from the aborted epoch
            self.detailed_train_losses.append(e.metrics['detailed_losses'])
            self.detailed_vq_losses.append(e.metrics['detailed_vq_losses'])
            self.detailed_perplexities.append(e.metrics['detailed_perplexities'])
            self.detailed_batch_indices.append(e.metrics['detailed_batch_indices'])
            self.detailed_recon_losses.append(e.metrics['detailed_recon_losses'])
            
            # Create a dummy validation metrics for checkpoint saving
            # Use the training metrics as a proxy since we didn't complete validation
            dummy_val_metrics = {
                'loss': e.metrics['avg_loss'],  # Use training loss as proxy
                'vq_loss': e.metrics['avg_vq_loss'],
                'perplexity': e.metrics['avg_perplexity'],
                'recon_loss': e.metrics['avg_recon_loss'],
            }
            
            # Save checkpoint for the aborted training
            checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Get minimum batches threshold for saving checkpoint
            minimum_batches = self.training_config.get('minimum_batches_for_checkpoint', 200)
            
            # Calculate total batches trained (detailed_losses contains one entry per measurement interval)
            total_batches_trained = len(e.metrics['detailed_losses'])
            # Estimate total batches based on measurements and measurement frequency
            total_batches_trained = total_batches_trained * num_measurements_per_epoch
            
            # Save as a special "aborted" checkpoint if we've trained enough batches
            if total_batches_trained >= minimum_batches:
                aborted_checkpoint_path = os.path.join(checkpoint_dir, f'aborted_training_epoch_{e.epoch}.pt')
                if TRACK_MEMORY:
                    with record_function("## save_checkpoint ##"):
                        self.save_checkpoint(e.epoch, dummy_val_metrics, is_best=False, checkpoint_path=aborted_checkpoint_path)
                else:
                    self.save_checkpoint(e.epoch, dummy_val_metrics, is_best=False, checkpoint_path=aborted_checkpoint_path)
                # Send checkpoint notification for aborted training
                if SEND_NOTIFICATION:
                    self.send_checkpoint_notification(e.epoch, is_best=False)
                print(f"Aborted training checkpoint saved (trained {total_batches_trained} batches, threshold: {minimum_batches})")

                # Also save as best model if it's better than previous best
                if self.is_new_best(e.metrics['avg_loss']):
                    self.update_best(e.metrics['avg_loss'])
                    best_aborted_path = os.path.join(checkpoint_dir, f'best_model_aborted_epoch_{e.epoch}.pt')
                    if TRACK_MEMORY:
                        with record_function("## save_checkpoint ##"):
                            self.save_checkpoint(e.epoch, dummy_val_metrics, is_best=True, checkpoint_path=best_aborted_path)
                    else:
                        self.save_checkpoint(e.epoch, dummy_val_metrics, is_best=True, checkpoint_path=best_aborted_path)
                    # Send checkpoint notification for best aborted model
                    if SEND_NOTIFICATION:
                        self.send_checkpoint_notification(e.epoch, is_best=True)
                    print(f"New best model (from aborted training) saved to: {best_aborted_path}")
            else:
                print(f"Skipping checkpoint save - only trained {total_batches_trained} batches, need at least {minimum_batches}")
            
            # Log final memory usage for aborted training
            self.log_memory_usage("training_aborted_end")
            
            print(f"\nTraining aborted! Best validation loss so far: {self.best_val_loss:.4f}")
            print(f"Training completed at epoch {e.epoch} due to perplexity threshold.")
            
            # Clear cache after aborted training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Calculate training duration for aborted training
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            
            # Get final metrics from the aborted epoch
            final_metrics = {
                'loss': e.metrics['avg_loss'],
                'vq_loss': e.metrics['avg_vq_loss'],
                'perplexity': e.metrics['avg_perplexity'],
                'recon_loss': e.metrics['avg_recon_loss'],
            }
            
            # Send the phone notification with aborted status
            if SEND_NOTIFICATION:
                _ = self.send_training_completion_phone_notification(
                    final_metrics=final_metrics,
                    training_duration=training_duration,
                    aborted=True
                )
            
            # Save comprehensive training visualizations for aborted training
            save_training_visualizations(self, prefix="aborted_training")
            
            # Re-raise the exception to be caught by the main function
            raise
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()  
            # Send the phone notification with aborted status
            if SEND_NOTIFICATION:
                # Format the message
                message = f"Training halted for {self.run_name} with error: {e}!\n"
                send_notification(message)
        
        # Log final memory usage
        self.log_memory_usage("training_end")
        
        # Calculate training duration
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
        print(f"Training duration: {training_duration:.2f} seconds ({training_duration/3600:.2f} hours)")
        
        
        # Get final metrics from the last validation
        final_metrics = {
            'val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'vq_loss': self.vq_losses[-1] if self.vq_losses else 0.0,
            'perplexity': self.perplexities[-1] if self.perplexities else 0.0,
            'val_recon_loss': self.val_recon_losses[-1] if self.val_recon_losses else 0.0,
        }
        
        # Send the phone notification
        if SEND_NOTIFICATION:
            _ = self.send_training_completion_phone_notification(
                final_metrics=final_metrics,
                training_duration=training_duration
            )
        
        # Save comprehensive training visualizations
        save_training_visualizations(self, prefix="training")

    def train(self, 
              train_prompt_sequences: torch.Tensor,
              train_cot_sequences: torch.Tensor,
              train_prompt_mask: torch.Tensor,
              train_cot_mask: torch.Tensor,
              test_prompt_sequences: torch.Tensor,
              test_cot_sequences: torch.Tensor,
              test_prompt_mask: torch.Tensor,
              test_cot_mask: torch.Tensor,
              resume_from: Optional[str] = None,
              num_measurements_per_epoch: Optional[int] = None,
              seed: int = 42):
        """
        Train the model with memory optimizations using pre-split train and test data.
        
        Args:
            train_prompt_sequences: Training prompt sequences
            train_cot_sequences: Training CoT sequences
            train_prompt_mask: Training prompt masks
            train_cot_mask: Training CoT masks
            test_prompt_sequences: Test prompt sequences
            test_cot_sequences: Test CoT sequences
            test_prompt_mask: Test prompt masks
            test_cot_mask: Test CoT masks
            resume_from: Path to checkpoint to resume from
            num_measurements_per_epoch: Number of metrics saved per epoch
            seed: Random seed for reproducibility
        """
        # Create train and test datasets directly from provided tensors
        train_dataset = TensorDataset(train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask)
        test_dataset = TensorDataset(test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
        
        # Create data loaders with memory optimizations
        train_loader = self.create_data_loader(
            train_dataset,
            batch_size=self.training_config['batch_size'], 
            shuffle=True
        )
        
        test_loader = self.create_data_loader(
            test_dataset,
            batch_size=self.training_config['batch_size'], 
            shuffle=False
        )
        
        # Call the helper function with the created loaders
        return self._train_with_loaders(
            train_loader=train_loader,
            test_loader=test_loader,
            resume_from=resume_from,
            num_measurements_per_epoch=num_measurements_per_epoch,
            seed=seed
        )
    
    def plot_training_history(self, save_path: Optional[str] = None, log_scale: bool = False):
        """
        Plot training history including detailed metrics within epochs.
        
        Args:
            save_path: Path to save the plot
            log_scale: If True, use log scale for y-axes
        """
        # Create a larger figure to accommodate detailed plots
        _, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Epoch-level metrics (top row)
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Total Train Loss')
        axes[0, 0].plot(self.recon_losses, label='Train Recon Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].plot(self.val_recon_losses, label='Val Recon Loss')
        axes[0, 0].set_title('Training and Validation Loss (Epoch Level)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        if log_scale:
            axes[0, 0].set_yscale('log')
        
        # VQ Loss plot
        axes[0, 1].plot(self.vq_losses, label='VQ Loss', color='red')
        axes[0, 1].set_title('Vector Quantization Loss (Epoch Level)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('VQ Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        if log_scale:
            axes[0, 1].set_yscale('log')
        
        # Plot epoch-level perplexity
        axes[1, 0].plot(self.perplexities, label='Epoch Perplexity', color='orange')
        axes[1, 0].set_title('Codebook Perplexity (Epoch Level)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        if log_scale:
            axes[1, 0].set_yscale('log')
        
        # Detailed metrics within epochs
        if self.detailed_train_losses and self.detailed_recon_losses:
            all_detailed_losses = []
            all_detailed_recon_losses = []
            all_detailed_vq_losses = []
            all_detailed_perplexities = []
            all_detailed_indices = []
            epoch_boundaries = []
            global_batch_idx = 0
            for epoch_idx, (epoch_losses, epoch_recon_losses, epoch_vq_losses, epoch_perplexities, epoch_indices) in enumerate(
                zip(self.detailed_train_losses, self.detailed_recon_losses, self.detailed_vq_losses, self.detailed_perplexities, self.detailed_batch_indices)
            ):
                epoch_boundaries.append((global_batch_idx, epoch_idx + 1))
                for batch_idx, (loss, recon_loss, vq_loss, perplexity) in enumerate(
                    zip(epoch_losses, epoch_recon_losses, epoch_vq_losses, epoch_perplexities)
                ):
                    all_detailed_losses.append(loss)
                    all_detailed_recon_losses.append(recon_loss)
                    all_detailed_vq_losses.append(vq_loss)
                    all_detailed_perplexities.append(perplexity)
                    all_detailed_indices.append(global_batch_idx + batch_idx)
                global_batch_idx += len(epoch_losses)
            
            # Plot detailed training loss
            axes[1, 1].plot(all_detailed_indices, all_detailed_losses, label='Detailed Total Loss', alpha=0.7)
            axes[1, 1].plot(all_detailed_indices, all_detailed_recon_losses, label='Detailed Recon Loss', alpha=0.7)
            for boundary, epoch_num in epoch_boundaries:
                axes[1, 1].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                axes[1, 1].text(boundary, axes[1, 1].get_ylim()[1], f'Epoch {epoch_num}', 
                              rotation=90, va='top', ha='right')
            axes[1, 1].set_title('Detailed Training and Recon Loss (Within Epochs)')
            axes[1, 1].set_xlabel('Measurement Index')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            if log_scale:
                axes[1, 1].set_yscale('log')
            
            # Plot detailed VQ loss
            axes[2, 0].plot(all_detailed_indices, all_detailed_vq_losses, label='Detailed VQ Loss', color='red', alpha=0.7)
            for boundary, epoch_num in epoch_boundaries:
                axes[2, 0].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                axes[2, 0].text(boundary, axes[2, 0].get_ylim()[1], f'Epoch {epoch_num}',
                              rotation=90, va='top', ha='right')
            axes[2, 0].set_title('Detailed VQ Loss (Within Epochs)')
            axes[2, 0].set_xlabel('Measurement Index')
            axes[2, 0].set_ylabel('VQ Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            if log_scale:
                axes[2, 0].set_yscale('log')
            
            # Plot detailed perplexity
            axes[2, 1].plot(all_detailed_indices, all_detailed_perplexities, label='Detailed Perplexity', color='green', alpha=0.7)
            for boundary, epoch_num in epoch_boundaries:
                axes[2, 1].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                axes[2, 1].text(boundary, axes[2, 1].get_ylim()[1], f'Epoch {epoch_num}',
                              rotation=90, va='top', ha='right')
            axes[2, 1].set_title('Detailed Codebook Perplexity (Within Epochs)')
            axes[2, 1].set_xlabel('Measurement Index')
            axes[2, 1].set_ylabel('Perplexity')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
            if log_scale:
                axes[2, 1].set_yscale('log')
            # New: Plot VQ input mean/std
            all_vq_means = np.concatenate(self.detailed_vq_input_means) if self.detailed_vq_input_means else []
            all_vq_stds = np.concatenate(self.detailed_vq_input_stds) if self.detailed_vq_input_stds else []
            if len(all_vq_means) > 0:
                fig2, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(all_detailed_indices, all_vq_means, label='VQ Input Mean', color='blue')
                ax2.plot(all_detailed_indices, all_vq_stds, label='VQ Input Std', color='red')
                for boundary, epoch_num in epoch_boundaries:
                    ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                    ax2.text(boundary, ax2.get_ylim()[1], f'Epoch {epoch_num}', rotation=90, va='top', ha='right')
                ax2.set_title('VQ Input Mean/Std (Within Epochs)')
                ax2.set_xlabel('Measurement Index')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid(True)
                if save_path:
                    vq_input_path = save_path.replace('.png', '_vq_input.png')
                    fig2.savefig(vq_input_path, dpi=300, bbox_inches='tight')
                    print(f"VQ input mean/std plot saved to {vq_input_path}")
                plt.close(fig2)
            # New: Plot chain embedding stats per epoch
            if self.epoch_chain_embeddings:
                means = [np.mean(e) for e in self.epoch_chain_embeddings]
                stds = [np.std(e) for e in self.epoch_chain_embeddings]
                mins = [np.min(e) for e in self.epoch_chain_embeddings]
                maxs = [np.max(e) for e in self.epoch_chain_embeddings]
                fig3, ax3 = plt.subplots(figsize=(12, 4))
                ax3.plot(means, label='Chain Embedding Mean', color='blue')
                ax3.plot(stds, label='Chain Embedding Std', color='red')
                ax3.plot(mins, label='Chain Embedding Min', color='green')
                ax3.plot(maxs, label='Chain Embedding Max', color='orange')
                ax3.set_title('Chain Embedding Stats per Epoch')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Value')
                ax3.legend()
                ax3.grid(True)
                if save_path:
                    chain_emb_path = save_path.replace('.png', '_chain_emb.png')
                    fig3.savefig(chain_emb_path, dpi=300, bbox_inches='tight')
                    print(f"Chain embedding stats plot saved to {chain_emb_path}")
                plt.close(fig3)

                # 3D mesh: visualize how each embedding dimension evolves over epochs
                # epoch_chain_embeddings: list of (num_embeddings, emb_dim) arrays, one per epoch
                chain_embs = np.stack(self.epoch_chain_embeddings, axis=0)  # shape: (epochs, num_embeddings, emb_dim)
                num_epochs, num_embeddings, emb_dim = chain_embs.shape

                # For each embedding index, plot a 2D heatmap (epoch x dim) of its vector evolution
                for idx in range(num_embeddings):
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
                    # Z: (epochs, emb_dim) for this embedding index
                    Z = chain_embs[:, idx, :]  # shape: (epochs, emb_dim)
                    im = ax_heatmap.imshow(Z, aspect='auto', cmap='viridis', origin='lower')
                    ax_heatmap.set_title(f'Chain Embedding {idx} Evolution (Epoch x Dim)')
                    ax_heatmap.set_xlabel('Embedding Dimension')
                    ax_heatmap.set_ylabel('Epoch')
                    fig_heatmap.colorbar(im, ax=ax_heatmap, orientation='vertical', label='Value')

                    # Add visible horizontal lines to separate each epoch
                    num_epochs = Z.shape[0]
                    for i in range(1, num_epochs):
                        ax_heatmap.axhline(i - 0.5, color='black', linewidth=3, alpha=0.8, linestyle='-')

                    plt.tight_layout()
                    if save_path:
                        heatmap_path = save_path.replace('.png', f'_chain_emb_{idx}_heatmap.png')
                        fig_heatmap.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                        print(f"Chain embedding heatmap for embedding {idx} saved to {heatmap_path}")
                    plt.close(fig_heatmap)

                # 2D heatmap: show the final chain embeddings (num_embeddings x emb_dim)
                final_chain_emb = self.epoch_chain_embeddings[-1]  # shape: (num_embeddings, emb_dim)
                fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
                im = ax_heat.imshow(final_chain_emb, aspect='auto', cmap='viridis')
                ax_heat.set_title('Final Chain Embeddings (Heatmap)')
                ax_heat.set_xlabel('Embedding Dimension')
                ax_heat.set_ylabel('Embedding Index')
                fig_heat.colorbar(im, ax=ax_heat, orientation='vertical', label='Value')

                # Add visible horizontal lines to separate each embedding
                num_embeddings = final_chain_emb.shape[0]
                for i in range(1, num_embeddings):
                    ax_heat.axhline(i - 0.5, color='black', linewidth=3, alpha=0.8, linestyle='-')

                plt.tight_layout()
                if save_path:
                    heatmap_path = save_path.replace('.png', '_chain_emb_final_heatmap.png')
                    fig_heat.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    print(f"Final chain embedding heatmap saved to {heatmap_path}")
                plt.close(fig_heat)
        else:
            # Fallback to original plots if no detailed data
            axes[1, 1].text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Detailed Training and Recon Loss')
            
            axes[2, 0].text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Detailed VQ Loss')
            
            axes[2, 1].text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Detailed Perplexity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.close()
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage throughout training, including CPU RAM."""
        USE_SEPARATE_AXIS_FOR_CPURAM = True

        if not self.memory_stats:
            print("No memory statistics available")
            return
        # Check if we have GPU memory stats (indicates memory_monitor was used)
        has_gpu_stats = 'gpu_total_gb' in self.memory_stats[0]
        stages = [stat['stage'] for stat in self.memory_stats]
        cpu_ram = [stat.get('cpu_ram_gb', 0.0) for stat in self.memory_stats]
        if has_gpu_stats:
            fig, axes = plt.subplots(3, 1, figsize=(12, 14))
            gpu_used = [stat['gpu_used_gb'] for stat in self.memory_stats]
            gpu_total = [stat['gpu_total_gb'] for stat in self.memory_stats]
            gpu_utilization = [stat['gpu_utilization_percent'] for stat in self.memory_stats]
            pytorch_allocated = [stat['pytorch_allocated_gb'] for stat in self.memory_stats]
            pytorch_reserved = [stat['pytorch_reserved_gb'] for stat in self.memory_stats]
            pytorch_max_allocated = [stat['pytorch_max_allocated_gb'] for stat in self.memory_stats]
            # Plot GPU memory usage
            axes[0].plot(range(len(stages)), gpu_used, label='GPU Used', marker='o', color='blue')
            axes[0].plot(range(len(stages)), gpu_total, label='GPU Total', marker='s', color='red', linestyle='--')
            
            ax2 = axes[0].twinx() if USE_SEPARATE_AXIS_FOR_CPURAM else axes[0]

            ax2.plot(range(len(stages)), cpu_ram, label='CPU RAM Used', color='purple', marker='x', linestyle=':')
            axes[0].set_title('GPU and CPU Memory Usage Throughout Training')
            axes[0].set_xlabel('Training Stage')
            axes[0].set_ylabel('GPU Memory (GB)')
            ax2.set_ylabel('CPU RAM (GB)')
            axes[0].legend(loc='upper left')
            ax2.legend(loc='upper right')
            axes[0].grid(True)
            axes[0].set_xticks(range(len(stages)))
            axes[0].set_xticklabels(stages, rotation=45, ha='right')
            # Plot GPU utilization
            axes[1].plot(range(len(stages)), gpu_utilization, label='GPU Utilization', color='green', marker='^')
            axes[1].set_title('GPU Memory Utilization')
            axes[1].set_xlabel('Training Stage')
            axes[1].set_ylabel('Utilization (%)')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_xticks(range(len(stages)))
            axes[1].set_xticklabels(stages, rotation=45, ha='right')
            # Plot PyTorch memory usage
            axes[2].plot(range(len(stages)), pytorch_allocated, label='PyTorch Allocated', marker='o')
            axes[2].plot(range(len(stages)), pytorch_reserved, label='PyTorch Reserved', marker='s')
            axes[2].plot(range(len(stages)), pytorch_max_allocated, label='PyTorch Max Allocated', color='red', marker='^')
            
            # again, can choose between using the same or a separate axis
            ax3 = axes[2].twinx() if USE_SEPARATE_AXIS_FOR_CPURAM else axes[2]
            
            ax3.plot(range(len(stages)), cpu_ram, label='CPU RAM Used', color='purple', marker='x', linestyle=':')
            axes[2].set_title('PyTorch and CPU Memory Usage')
            axes[2].set_xlabel('Training Stage')
            axes[2].set_ylabel('PyTorch Memory (GB)')
            ax3.set_ylabel('CPU RAM (GB)')
            axes[2].legend(loc='upper left')
            ax3.legend(loc='upper right')
            axes[2].grid(True)
            axes[2].set_xticks(range(len(stages)))
            axes[2].set_xticklabels(stages, rotation=45, ha='right')
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            allocated = [stat['allocated_gb'] for stat in self.memory_stats]
            reserved = [stat['reserved_gb'] for stat in self.memory_stats]
            max_allocated = [stat['max_allocated_gb'] for stat in self.memory_stats]
            # Plot allocated vs reserved memory
            axes[0].plot(range(len(stages)), allocated, label='Allocated', marker='o')
            axes[0].plot(range(len(stages)), reserved, label='Reserved', marker='s')

            ax2 = axes[0].twinx() if USE_SEPARATE_AXIS_FOR_CPURAM else axes[0]

            ax2.plot(range(len(stages)), cpu_ram, label='CPU RAM Used', color='purple', marker='x', linestyle=':')
            axes[0].set_title('PyTorch and CPU Memory Usage Throughout Training')
            axes[0].set_xlabel('Training Stage')
            axes[0].set_ylabel('PyTorch Memory (GB)')
            ax2.set_ylabel('CPU RAM (GB)')
            axes[0].legend(loc='upper left')
            ax2.legend(loc='upper right')
            axes[0].grid(True)
            axes[0].set_xticks(range(len(stages)))
            axes[0].set_xticklabels(stages, rotation=45, ha='right')
            # Plot max allocated memory
            axes[1].plot(range(len(stages)), max_allocated, label='Max Allocated', color='red', marker='^')
            
            ax3 = axes[1].twinx() if USE_SEPARATE_AXIS_FOR_CPURAM else axes[1]
            
            ax3.plot(range(len(stages)), cpu_ram, label='CPU RAM Used', color='purple', marker='x', linestyle=':')
            axes[1].set_title('Maximum PyTorch and CPU Memory Usage')
            axes[1].set_xlabel('Training Stage')
            axes[1].set_ylabel('PyTorch Memory (GB)')
            ax3.set_ylabel('CPU RAM (GB)')
            axes[1].legend(loc='upper left')
            ax3.legend(loc='upper right')
            axes[1].grid(True)
            axes[1].set_xticks(range(len(stages)))
            axes[1].set_xticklabels(stages, rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory usage plot saved to {save_path}")
        plt.close()

    def is_new_best(self, val_loss: float) -> bool:
        """Return True if val_loss is better than the current best_val_loss."""
        return val_loss < self.best_val_loss

    def update_best(self, val_loss: float):
        """Update the best_val_loss if val_loss is better."""
        if self.is_new_best(val_loss):
            self.best_val_loss = val_loss


def save_training_visualizations(trainer, save_dir: str = None, prefix: str = "training"):
    """
    Save comprehensive training visualizations including all available plots and metrics.
    
    Args:
        trainer: The trainer instance with training data
        save_dir: Directory to save visualizations (defaults to trainer's checkpoint directory)
        prefix: Prefix for saved files
    """
    DEFAULT_FOLDER = './training_visualizations'
    try:
        if save_dir is None:
            training_config = getattr(trainer, 'training_config', None)
            if training_config is None:
                save_dir = DEFAULT_FOLDER
            else:
                save_dir = training_config.get('checkpoint_dir', DEFAULT_FOLDER)
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📊 Saving comprehensive training visualizations to {save_dir}")
        
        # 1. Training history plots
        history_path = os.path.join(save_dir, f"{prefix}_history.png")
        trainer.plot_training_history(history_path)
        
        # 2. Memory usage plots
        memory_path = os.path.join(save_dir, f"{prefix}_memory_usage.png")
        trainer.plot_memory_usage(memory_path)
        
        # 3. Codebook tracking plots (if enabled)
        if trainer.tracking_enabled:
            codebook_dir = os.path.join(save_dir, f"{prefix}_codebook_tracking")
            os.makedirs(codebook_dir, exist_ok=True)
            
            # Save current codebook plots
            if hasattr(trainer, 'save_codebook_plots_func'):
                trainer.save_codebook_plots_func(codebook_dir, len(trainer.train_losses))
        
        print(f"✅ All training visualizations saved successfully!")
        print(f"   📈 Training history: {history_path}")
        print(f"   💾 Memory usage: {memory_path}")
        if trainer.tracking_enabled:
            print(f"   🎯 Codebook tracking: {codebook_dir}")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to save some training visualizations: {e}")
        # Try to save at least basic plots
        try:
            if save_dir is None:
                save_dir = getattr(trainer, 'checkpoint_dir', './training_visualizations')
            os.makedirs(save_dir, exist_ok=True)
            
            # Fallback: save basic training history
            history_path = os.path.join(save_dir, f"{prefix}_history_fallback.png")
            trainer.plot_training_history(history_path)
            print(f"   📈 Basic training history saved: {history_path}")
        except Exception as fallback_error:
            print(f"   ❌ Failed to save even basic plots: {fallback_error}")