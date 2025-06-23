# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional, Dict, Any
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
# import wandb  # Optional: for experiment tracking
import argparse
import yaml
import gc
# import psutil
from torch.amp import autocast, GradScaler
from transformers import GPT2Tokenizer

# GPU memory monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: nvidia-ml-py3 not available. Install with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False

# Import the GPT2VQVAE model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vqvae_gpt2 import GPT2VQVAE

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

def dynamic_batch_sampler(dataset, max_tokens_per_batch: int = 8192):
    """
    Dynamic batch sampler that groups samples by sequence length to minimize padding.
    
    Args:
        dataset: Dataset to sample from
        max_tokens_per_batch: Maximum tokens per batch
    
    Yields:
        Batch indices
    """
    # Group samples by sequence length
    length_groups = {}
    for idx in range(len(dataset)):
        _, cot_seq, _, _ = dataset[idx]
        seq_len = cot_seq.size(-1)
        if seq_len not in length_groups:
            length_groups[seq_len] = []
        length_groups[seq_len].append(idx)
    
    # Create batches for each length group
    for seq_len, indices in length_groups.items():
        max_samples_per_batch = max_tokens_per_batch // seq_len
        
        for i in range(0, len(indices), max_samples_per_batch):
            batch_indices = indices[i:i + max_samples_per_batch]
            yield batch_indices

class GPT2VQVAETrainer:
    """
    Trainer class for GPT2VQVAE model with configurable hyperparameters and memory optimizations.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the trainer.
        
        Args:
            model_config: Dictionary containing model hyperparameters
            training_config: Dictionary containing training hyperparameters
            device: Device to train on
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        
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
        self.max_memory_gb = training_config.get('max_memory_gb', 4.0)
        
        # Ensure all numeric values are properly typed
        def ensure_numeric_types(config_dict):
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
        
        ensure_numeric_types(self.model_config)
        ensure_numeric_types(self.training_config)
        
        # Log memory before model initialization
        if self.memory_monitor:
            self.memory_monitor.log_memory_usage("before_model_init")
            self.memory_monitor.log_pytorch_memory_usage("before_model_init")
        
        # Initialize model
        self.model = GPT2VQVAE(**model_config).to(device)
        
        # Log memory after model weights loaded to GPU
        if self.memory_monitor:
            self.memory_monitor.log_memory_usage("after_model_weights_loaded")
            self.memory_monitor.log_pytorch_memory_usage("after_model_weights_loaded")
        
        # Enable gradient checkpointing if specified
        if self.use_gradient_checkpointing:
            print("use_gradient_checkpointing current does nothing.")
        #     self.model.gradient_checkpointing_enable()
        #     print("Gradient checkpointing enabled")
        
        # Log memory before optimizer initialization
        if self.memory_monitor:
            self.memory_monitor.log_memory_usage("before_optimizer_init")
            self.memory_monitor.log_pytorch_memory_usage("before_optimizer_init")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        
        # Log memory after optimizer initialization
        if self.memory_monitor:
            self.memory_monitor.log_memory_usage("after_optimizer_init")
            self.memory_monitor.log_pytorch_memory_usage("after_optimizer_init")
        
        # Initialize mixed precision training
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
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
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.vq_losses = []
        self.perplexities = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Memory monitoring
        self.memory_stats = []
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage using the GPU memory monitor."""
        if self.memory_monitor:
            # Log both GPU and PyTorch memory usage
            gpu_mem = self.memory_monitor.log_memory_usage(stage, print_info=False)
            pytorch_mem = self.memory_monitor.log_pytorch_memory_usage(stage, print_info=False)
            
            # Store combined memory stats
            self.memory_stats.append({
                'stage': stage,
                'gpu_total_gb': gpu_mem['total_gb'],
                'gpu_used_gb': gpu_mem['used_gb'],
                'gpu_free_gb': gpu_mem['free_gb'],
                'gpu_utilization_percent': gpu_mem['utilization_percent'],
                'pytorch_allocated_gb': pytorch_mem['allocated_gb'],
                'pytorch_reserved_gb': pytorch_mem['reserved_gb'],
                'pytorch_max_allocated_gb': pytorch_mem['max_allocated_gb']
            })
            
            # Print combined information
            print(f"Memory Usage ({stage}):")
            print(f"  GPU: {gpu_mem['used_gb']:.2f}GB used / {gpu_mem['total_gb']:.2f}GB total ({gpu_mem['utilization_percent']:.1f}%)")
            print(f"  PyTorch: {pytorch_mem['allocated_gb']:.2f}GB allocated, {pytorch_mem['reserved_gb']:.2f}GB reserved")
        else:
            # Fallback to original method if no memory monitor
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                
                self.memory_stats.append({
                    'stage': stage,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'max_allocated_gb': max_allocated
                })
                
                print(f"Memory usage ({stage}): {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
    
    
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
            sampler = torch.utils.data.SequentialSampler(dataset) if not shuffle else torch.utils.data.RandomSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
            return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with memory optimizations.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        accumulation_steps = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (prompts, cots, prompt_masks, cot_masks) in enumerate(progress_bar):
            # Move to device
            prompts = prompts.to(self.device, non_blocking=True)
            cots = cots.to(self.device, non_blocking=True)
            prompt_masks = prompt_masks.to(self.device, non_blocking=True)
            cot_masks = cot_masks.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast('cuda'):
                    _, output_logits, vq_loss, perplexity = self.model(
                        prompt=prompts,
                        cot_sequences=cots,
                        cot_mask=cot_masks,
                        prompt_mask=prompt_masks,
                        inference=False,
                        quantize_cot_only=self.training_config.get('quantize_cot_only', True)
                    )
                    
                    # Calculate reconstruction loss
                    logits_flat = output_logits.view(-1, output_logits.size(-1))
                    targets_flat = cots.view(-1)
                    mask_flat = cot_masks.view(-1).bool()
                    
                    if mask_flat.sum() > 0:
                        recon_loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                    
                    # Total loss
                    total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = total_loss_batch / self.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(scaled_loss).backward()
            else:
                _, output_logits, vq_loss, perplexity = self.model(
                    prompt=prompts,
                    cot_sequences=cots,
                    cot_mask=cot_masks,
                    prompt_mask=prompt_masks,
                    inference=False,
                    quantize_cot_only=self.training_config.get('quantize_cot_only', True)
                )
                
                # Calculate reconstruction loss
                logits_flat = output_logits.view(-1, output_logits.size(-1))
                targets_flat = cots.view(-1)
                mask_flat = cot_masks.view(-1).bool()
                
                if mask_flat.sum() > 0:
                    recon_loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
                
                # Scale loss for gradient accumulation
                scaled_loss = total_loss_batch / self.gradient_accumulation_steps
                scaled_loss.backward()
            
            accumulation_steps += 1
            
            # Update weights every gradient_accumulation_steps
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                if self.use_mixed_precision:
                    # Gradient clipping with scaler
                    if self.training_config.get('gradient_clip', 1.0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config['gradient_clip']
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.training_config.get('gradient_clip', 1.0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config['gradient_clip']
                        )
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'vq_loss': f"{vq_loss.item():.4f}",
                'perplexity': f"{perplexity.item():.2f}",
                'accum_steps': f"{accumulation_steps}/{self.gradient_accumulation_steps}"
            })
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'loss': avg_loss,
            'vq_loss': avg_vq_loss,
            'perplexity': avg_perplexity
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model with memory optimizations.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for prompts, cots, prompt_masks, cot_masks in tqdm(val_loader, desc="Validation"):
                # Move to device
                prompts = prompts.to(self.device, non_blocking=True)
                cots = cots.to(self.device, non_blocking=True)
                prompt_masks = prompt_masks.to(self.device, non_blocking=True)
                cot_masks = cot_masks.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with autocast('cuda'):
                        output_sequences, output_logits, vq_loss, perplexity = self.model(
                            prompt=prompts,
                            cot_sequences=cots,
                            cot_mask=cot_masks,
                            prompt_mask=prompt_masks,
                            inference=False,
                            quantize_cot_only=self.training_config.get('quantize_cot_only', True)
                        )
                        
                        # Calculate reconstruction loss
                        logits_flat = output_logits.view(-1, output_logits.size(-1))
                        targets_flat = cots.view(-1)
                        mask_flat = cot_masks.view(-1).bool()
                        
                        if mask_flat.sum() > 0:
                            recon_loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                        else:
                            recon_loss = torch.tensor(0.0, device=self.device)
                        
                        total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
                else:
                    output_sequences, output_logits, vq_loss, perplexity = self.model(
                        prompt=prompts,
                        cot_sequences=cots,
                        cot_mask=cot_masks,
                        prompt_mask=prompt_masks,
                        inference=False,
                        quantize_cot_only=self.training_config.get('quantize_cot_only', True)
                    )
                    
                    # Calculate reconstruction loss
                    logits_flat = output_logits.view(-1, output_logits.size(-1))
                    targets_flat = cots.view(-1)
                    mask_flat = cot_masks.view(-1).bool()
                    
                    if mask_flat.sum() > 0:
                        recon_loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
                
                # Update metrics
                total_loss += total_loss_batch.item()
                total_vq_loss += vq_loss.item()
                total_perplexity += perplexity.item()
                num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        
        return {
            'loss': avg_loss,
            'vq_loss': avg_vq_loss,
            'perplexity': avg_perplexity
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
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
            'perplexities': self.perplexities
        }
        
        # Save best model if this is the best so far
        if is_best:
            # Remove any existing best model checkpoints
            for file in os.listdir(checkpoint_dir):
                if 'best_model' in file and file.endswith('.pt'):
                    os.remove(os.path.join(checkpoint_dir, file))
            
            best_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved (epoch {epoch}) with validation loss: {metrics['loss']:.4f}")
        else:
            # Save regular checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
        
    
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
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.vq_losses = checkpoint.get('vq_losses', [])
        self.perplexities = checkpoint.get('perplexities', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print("Checkpoint loaded successfully. Configuration validation completed.")
    
    def train(self, 
              prompt_sequences: torch.Tensor,
              cot_sequences: torch.Tensor,
              prompt_mask: torch.Tensor,
              cot_mask: torch.Tensor,
              val_split: float = 0.1,
              resume_from: Optional[str] = None,
              seed: int = 42):
        """
        Train the model with memory optimizations.
        
        Args:
            prompt_sequences: Training prompt sequences
            cot_sequences: Training CoT sequences
            prompt_mask: Training prompt masks
            cot_mask: Training CoT masks
            val_split: Fraction of data to use for validation
            resume_from: Path to checkpoint to resume from
            seed: Random seed for reproducible train/validation split
        """
        # Log initial memory usage
        self.log_memory_usage("training_start")
        
        # Create dataset and split into train/validation using PyTorch's random_split
        dataset = TensorDataset(prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        
        # Calculate split sizes
        num_samples = len(dataset)
        val_size = int(num_samples * val_split)
        train_size = num_samples - val_size    

        if train_size == 0:
            raise Exception(f"The training dataset has size 0 given : {num_samples} samples, val_split={val_split}.")
        elif val_size == 0:
            raise Exception(f"The validation dataset has size 0 given : {num_samples} samples, val_split={val_split}.")
        
        # Use PyTorch's random_split for reproducible train/validation split
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)  # Use provided seed for reproducibility
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create data loaders with memory optimizations
        train_loader = self.create_data_loader(
            train_dataset,
            batch_size=self.training_config['batch_size'], 
            shuffle=True
        )
        
        val_loader = self.create_data_loader(
            val_dataset,
            batch_size=self.training_config['batch_size'], 
            shuffle=False
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = len(self.train_losses)
        
        # Training loop
        for epoch in range(start_epoch, self.training_config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.training_config['num_epochs']}")
            print("-" * 50)
            
            # Log memory before epoch
            self.log_memory_usage(f"epoch_{epoch+1}_start")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Log memory after training
            self.log_memory_usage(f"epoch_{epoch+1}_after_train")
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log memory after validation
            self.log_memory_usage(f"epoch_{epoch+1}_after_val")
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.vq_losses.append(train_metrics['vq_loss'])
            self.perplexities.append(train_metrics['perplexity'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"VQ Loss: {train_metrics['vq_loss']:.4f}")
            print(f"Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch + 1, val_metrics, True)
            
            if (epoch + 1) % self.training_config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch + 1, val_metrics, False)
            
            # Clear cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Log final memory usage
        self.log_memory_usage("training_end")
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # VQ Loss plot
        axes[0, 1].plot(self.vq_losses, label='VQ Loss', color='red')
        axes[0, 1].set_title('Vector Quantization Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('VQ Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Perplexity plot
        axes[1, 0].plot(self.perplexities, label='Perplexity', color='green')
        axes[1, 0].set_title('Codebook Perplexity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Validation loss only
        axes[1, 1].plot(self.val_losses, label='Val Loss', color='orange')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """
        Plot memory usage throughout training.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.memory_stats:
            print("No memory statistics available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extract data
        stages = [stat['stage'] for stat in self.memory_stats]
        allocated = [stat['allocated_gb'] for stat in self.memory_stats]
        reserved = [stat['reserved_gb'] for stat in self.memory_stats]
        max_allocated = [stat['max_allocated_gb'] for stat in self.memory_stats]
        
        # Plot allocated vs reserved memory
        axes[0].plot(range(len(stages)), allocated, label='Allocated', marker='o')
        axes[0].plot(range(len(stages)), reserved, label='Reserved', marker='s')
        axes[0].set_title('GPU Memory Usage Throughout Training')
        axes[0].set_xlabel('Training Stage')
        axes[0].set_ylabel('Memory (GB)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xticks(range(len(stages)))
        axes[0].set_xticklabels(stages, rotation=45, ha='right')
        
        # Plot max allocated memory
        axes[1].plot(range(len(stages)), max_allocated, label='Max Allocated', color='red', marker='^')
        axes[1].set_title('Maximum GPU Memory Usage')
        axes[1].set_xlabel('Training Stage')
        axes[1].set_ylabel('Memory (GB)')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xticks(range(len(stages)))
        axes[1].set_xticklabels(stages, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory usage plot saved to {save_path}")
        
        plt.show()

    def load_training_data_memory_efficient(self, data_dir: str, max_samples: Optional[int] = None, num_thoughts: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load training data with memory-efficient loading and truncate based on num_thoughts.
        
        Args:
            data_dir: Directory containing the preprocessed data files
            max_samples: Maximum number of samples to load (for debugging/memory constraints)
            num_thoughts: Number of parallel sequences to use (truncates dataset if needed)
            
        Returns:
            Tuple of (prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        """
        # Log memory before loading data
        if self.memory_monitor:
            self.memory_monitor.log_memory_usage("before_data_loading")
            self.memory_monitor.log_pytorch_memory_usage("before_data_loading")
        
        required_files = [
            "prompt_sequences.pt",
            "cot_sequences_tensor.pt", 
            "prompt_mask.pt",
            "cot_mask.pt"
        ]
        
        # Check if all required files exist
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing data files in {data_dir}: {missing_files}")
        
        # Load tensors with memory mapping if available
        try:
            # Try to use memory mapping for large files
            prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"), map_location='cpu')
            cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"), map_location='cpu')
            prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"), map_location='cpu')
            cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"), map_location='cpu')
        except Exception as e:
            print(f"Warning: Could not use memory mapping: {e}")
            # Fallback to regular loading
            prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"))
            cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"))
            prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"))
            cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"))
        
        print(f"Original data shapes:")
        print(f"  prompt_sequences: {prompt_sequences.shape}")
        print(f"  cot_sequences: {cot_sequences.shape}")
        print(f"  prompt_mask: {prompt_mask.shape}")
        print(f"  cot_mask: {cot_mask.shape}")
        
        # Validate and truncate based on num_thoughts
        if num_thoughts is not None:
            current_num_thoughts = cot_sequences.shape[1]  # Should be the second dimension
            print(f"Current num_thoughts in dataset: {current_num_thoughts}")
            print(f"Requested num_thoughts: {num_thoughts}")
            
            if current_num_thoughts < num_thoughts:
                raise ValueError(f"Dataset only has {current_num_thoughts} parallel sequences, "
                               f"but model requires {num_thoughts}. Please regenerate dataset with more sequences.")
            
            if current_num_thoughts > num_thoughts:
                print(f"Truncating dataset from {current_num_thoughts} to {num_thoughts} parallel sequences")
                # Truncate the middle dimension (num_thoughts)
                cot_sequences = cot_sequences[:, :num_thoughts, :]
                cot_mask = cot_mask[:, :num_thoughts, :]
                
                print(f"Truncated data shapes:")
                print(f"  prompt_sequences: {prompt_sequences.shape}")
                print(f"  cot_sequences: {cot_sequences.shape}")
                print(f"  prompt_mask: {prompt_mask.shape}")
                print(f"  cot_mask: {cot_mask.shape}")
        
        # Limit samples if specified
        if max_samples is not None:
            prompt_sequences = prompt_sequences[:max_samples]
            cot_sequences = cot_sequences[:max_samples]
            prompt_mask = prompt_mask[:max_samples]
            cot_mask = cot_mask[:max_samples]
        
        print(f"Final data shapes:")
        print(f"  prompt_sequences: {prompt_sequences.shape}")
        print(f"  cot_sequences: {cot_sequences.shape}")
        print(f"  prompt_mask: {prompt_mask.shape}")
        print(f"  cot_mask: {cot_mask.shape}")
        
        # Calculate memory usage
        total_memory_gb = (
            prompt_sequences.element_size() * prompt_sequences.numel() +
            cot_sequences.element_size() * cot_sequences.numel() +
            prompt_mask.element_size() * prompt_mask.numel() +
            cot_mask.element_size() * cot_mask.numel()
        ) / 1e9
        
        print(f"Total data memory usage: {total_memory_gb:.2f} GB")
        
        return prompt_sequences, cot_sequences, prompt_mask, cot_mask

def validate_model_data_compatibility(model_config: Dict[str, Any], 
                                    prompt_sequences: torch.Tensor,
                                    cot_sequences: torch.Tensor,
                                    prompt_mask: torch.Tensor,
                                    cot_mask: torch.Tensor) -> None:
    """
    Validate that model configuration is compatible with the loaded data.
    
    Args:
        model_config: Model configuration dictionary
        prompt_sequences: Prompt sequences tensor
        cot_sequences: Chain-of-thought sequences tensor
        prompt_mask: Prompt mask tensor
        cot_mask: Chain-of-thought mask tensor
    """
    print("Validating model-data compatibility...")
    
    # Check num_thoughts compatibility
    num_thoughts = model_config.get('num_thoughts', None)
    if num_thoughts is not None:
        data_num_thoughts = cot_sequences.shape[1]
        if data_num_thoughts != num_thoughts:
            print(f"⚠️  Warning: Data has {data_num_thoughts} parallel sequences, "
                  f"but model config specifies {num_thoughts}.")
            if data_num_thoughts > num_thoughts:
                print(f"   → Dataset will be truncated to {num_thoughts} sequences during loading.")
            else:
                print(f"   → Error: Dataset has insufficient parallel sequences!")
                raise ValueError(f"Dataset only has {data_num_thoughts} parallel sequences, "
                               f"but model requires {num_thoughts}.")
        else:
            print(f"✅ num_thoughts compatibility: {num_thoughts} sequences")
    
    # Check vocabulary size compatibility
    vocab_size = model_config.get('vocab_size', None)
    if vocab_size is not None:
        # Check if any token IDs exceed the vocabulary size
        max_prompt_token = prompt_sequences.max().item()
        max_cot_token = cot_sequences.max().item()
        max_token = max(max_prompt_token, max_cot_token)
        
        if max_token >= vocab_size:
            print(f"⚠️  Warning: Data contains token ID {max_token}, "
                  f"but model vocab_size is {vocab_size}.")
        else:
            print(f"✅ Vocabulary compatibility: max token {max_token} < vocab_size {vocab_size}")
    
    # Check sequence length compatibility
    n_positions = model_config.get('n_positions', None)
    if n_positions is not None:
        prompt_len = prompt_sequences.shape[1]
        cot_len = cot_sequences.shape[2]
        total_len = prompt_len + cot_len
        
        if total_len > n_positions:
            print(f"⚠️  Warning: Total sequence length {total_len} exceeds model's n_positions {n_positions}.")
        else:
            print(f"✅ Sequence length compatibility: {total_len} <= n_positions {n_positions}")
    
    # Check tensor shapes consistency
    batch_size = prompt_sequences.shape[0]
    expected_shapes = {
        'prompt_sequences': (batch_size, prompt_sequences.shape[1]),
        'cot_sequences': (batch_size, cot_sequences.shape[1], cot_sequences.shape[2]),
        'prompt_mask': (batch_size, prompt_mask.shape[1]),
        'cot_mask': (batch_size, cot_mask.shape[1], cot_mask.shape[2])
    }
    
    actual_shapes = {
        'prompt_sequences': prompt_sequences.shape,
        'cot_sequences': cot_sequences.shape,
        'prompt_mask': prompt_mask.shape,
        'cot_mask': cot_mask.shape
    }
    
    print("✅ Tensor shape consistency:")
    for name, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[name]
        if actual_shape == expected_shape:
            print(f"   {name}: {actual_shape}")
        else:
            print(f"   ⚠️  {name}: expected {expected_shape}, got {actual_shape}")
    
    print("✅ Model-data compatibility validation complete!")

def load_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (model_config, training_config)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")
    
    # Extract model and training configs
    model_config = config.get('model_config', {})
    training_config = config.get('training_config', {})
    
    # Validate required fields
    required_model_fields = ['vocab_size', 'd_model', 'num_embeddings']
    required_training_fields = ['learning_rate', 'num_epochs', 'batch_size']
    
    missing_model = [field for field in required_model_fields if field not in model_config]
    missing_training = [field for field in required_training_fields if field not in training_config]
    
    if missing_model:
        raise ValueError(f"Missing required model config fields: {missing_model}")
    if missing_training:
        raise ValueError(f"Missing required training config fields: {missing_training}")
    
    return model_config, training_config

def create_default_config(output_path: str):
    """
    Create a default configuration file with memory optimizations.
    
    Args:
        output_path: Path where to save the default config
    """
    default_config = {
        'model_config': {
            'vocab_size': 50257,  # GPT2 vocabulary size
            'd_model': 768,       # GPT2 model dimension
            'num_embeddings': 512,  # VQ codebook size
            'commitment_cost': 0.25,  # VQ commitment cost
            'aggregation_hidden_dim': 1024,  # Aggregation MLP hidden dim
            'num_thoughts': 40,   # Number of parallel sequences
            'n_positions': 1024,   # Maximum sequence length
            # Pretrained model settings
            'use_pretrained_encoder': True,  # Load pretrained weights for encoder
            'use_pretrained_decoder': True,  # and decoder
            'pretrained_model_name': 'gpt2'  # Use GPT2-Small (124M parameters)
        },
        'training_config': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'use_lr_scheduler': True,
            'min_lr': 1e-6,
            'num_epochs': 50,
            'batch_size': 2,  # Reduced for memory efficiency
            'gradient_clip': 1.0,
            'vq_loss_weight': 1.0,
            'quantize_cot_only': True,
            'save_every': 5,
            'checkpoint_dir': 'checkpoints/gpt2vqvae',
            'pad_token_id': 50256, # eos_token for GPT2TokenizerFast
            'val_split': 0.1,
            # Memory optimization settings
            'use_mixed_precision': True,
            'gradient_accumulation_steps': 4,  # Effective batch size = batch_size * gradient_accumulation_steps
            'use_gradient_checkpointing': True,
            'max_memory_gb': 4.0,  # Maximum memory for dataset caching
            'use_dynamic_batching': False,  # Enable for variable sequence lengths
            'max_tokens_per_batch': 8192,  # For dynamic batching
            'max_samples': None  # Limit samples for debugging (set to number for testing)
        },
        'data_config': {
            'data_dir': 'data/GSM8K'
        }
    }
    
    # Determine file format based on extension
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    else:
        # Default to YAML
        output_path = output_path + '.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to: {output_path}")
    print("Memory optimization settings included:")
    print("  - Mixed precision training: Enabled")
    print("  - Gradient accumulation: 4 steps")
    print("  - Gradient checkpointing: Enabled")
    print("  - Memory-efficient dataset: Enabled")
    print("  - Reduced batch size: 2 (effective batch size: 8)")
    print("Pretrained model settings:")
    print("  - Encoder: GPT2-Small pretrained weights (use_pretrained_encoder: True)")
    print("  - Decoder: GPT2-Small pretrained weights (use_pretrained_decoder: True)")
    print("You can modify this file and use it for training.")

def demonstrate_model_from_checkpoint(checkpoint_path: str, 
                                    model_config: Dict[str, Any],
                                    data_dir: str,
                                    num_examples: int = 3,
                                    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Demonstrate GPT2VQVAE model generation capabilities from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration dictionary
        data_dir: Directory containing training data
        num_examples: Number of examples to generate
        device: Device to run the model on
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = GPT2VQVAE(**model_config).to(device)
    
    # Load checkpoint
    model.load_checkpoint(checkpoint_path, device=device)
    model.eval()
    
    # Get num_thoughts from model configuration
    num_thoughts = model_config.get('num_thoughts', 32)
    print(f"Model configured for {num_thoughts} parallel CoT sequences")
    
    # Load tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print("Loaded GPT2 tokenizer")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None
    
    # Load some example data
    print(f"Loading example data from: {data_dir}")
    try:
        # Load a small subset of data for demonstration
        prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"))[:num_examples]
        cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"))[:num_examples]
        prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"))[:num_examples]
        cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"))[:num_examples]
        
        print(f"Loaded {len(prompt_sequences)} examples")
        
        # Check if data has enough CoT sequences
        data_num_thoughts = cot_sequences.shape[1]
        if data_num_thoughts < num_thoughts:
            print(f"Warning: Data only has {data_num_thoughts} CoT sequences, but model expects {num_thoughts}")
            print(f"Will use {data_num_thoughts} sequences for demonstration")
            num_thoughts = data_num_thoughts
        elif data_num_thoughts > num_thoughts:
            print(f"Data has {data_num_thoughts} CoT sequences, truncating to {num_thoughts} for model compatibility")
            cot_sequences = cot_sequences[:, :num_thoughts, :]
            cot_mask = cot_mask[:, :num_thoughts, :]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Function to decode tokens to text
    def decode_tokens(tokens, mask=None):
        if tokenizer is None:
            return f"[Tokens: {tokens.tolist()}]"
        
        if mask is not None:
            # Apply mask to remove padding
            tokens = tokens[mask.bool()]
        
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            return f"[Decode error: {e}, tokens: {tokens.tolist()}]"
    
    print("\n" + "="*80)
    print("GPT2VQVAE GENERATION DEMONSTRATION")
    print("="*80)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(prompt_sequences))):
            print(f"\n--- Example {i+1} ---")
            
            # Prepare single example
            prompt = prompt_sequences[i:i+1].to(device)  # [1, K]
            cot_gt = cot_sequences[i:i+1].to(device)     # [1, M, L]
            prompt_mask_ex = prompt_mask[i:i+1].to(device) if prompt_mask is not None else None
            cot_mask_ex = cot_mask[i:i+1].to(device) if cot_mask is not None else None
            
            # Decode ground truth
            prompt_text = decode_tokens(prompt[0], prompt_mask_ex[0] if prompt_mask_ex is not None else None)
            print(f"Prompt: {prompt_text}")
            
            # Show all ground truth CoT sequences
            print(f"\nGround Truth CoT Sequences ({num_thoughts} parallel chains):")
            for j in range(num_thoughts):
                cot_gt_text = decode_tokens(cot_gt[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                print(f"  CoT {j+1}: {cot_gt_text}")
            
            # Generate with teacher forcing (inference=False)
            print("\n--- Teacher Forcing Generation ---")
            try:
                _, output_logits_tf, vq_loss_tf, perplexity_tf = model(
                    prompt=prompt,
                    cot_sequences=cot_gt,
                    cot_mask=cot_mask_ex,
                    prompt_mask=prompt_mask_ex,
                    inference=False,  # Teacher forcing
                    quantize_cot_only=True
                )
                
                # Get predicted tokens from logits
                predicted_tokens = torch.argmax(output_logits_tf, dim=-1)  # [B, M, L]
                
                # Decode teacher forcing predictions from logits for all sequences
                print(f"Teacher Forcing CoT Sequences ({num_thoughts} parallel chains):")
                for j in range(num_thoughts):
                    cot_tf_text = decode_tokens(predicted_tokens[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                    print(f"  CoT {j+1}: {cot_tf_text}")
                print(f"VQ Loss: {vq_loss_tf.item():.4f}, Perplexity: {perplexity_tf.item():.2f}")
                
            except Exception as e:
                print(f"Teacher forcing generation failed: {e}")
            
            # Generate auto-regressively (inference=True)
            print("\n--- Auto-regressive Generation ---")
            try:
                output_sequences_ar, output_logits_ar, vq_loss_ar, perplexity_ar = model(
                    prompt=prompt,
                    cot_sequences=cot_gt,
                    cot_mask=cot_mask_ex,
                    prompt_mask=prompt_mask_ex,
                    inference=True,  # Auto-regressive
                    quantize_cot_only=True
                )
                
                # Decode auto-regressive output for all sequences
                print(f"Auto-regressive CoT Sequences ({num_thoughts} parallel chains):")
                for j in range(num_thoughts):
                    cot_ar_text = decode_tokens(output_sequences_ar[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                    print(f"  CoT {j+1}: {cot_ar_text}")
                print(f"VQ Loss: {vq_loss_ar.item():.4f}, Perplexity: {perplexity_ar.item():.2f}")
                
            except Exception as e:
                print(f"Auto-regressive generation failed: {e}")
            
            print("\n" + "-"*60)
    
    print("\nDemonstration completed!")

def main():
    """
    Main function for command-line training with memory optimizations.
    """
    parser = argparse.ArgumentParser(description='Train GPT2VQVAE model with memory optimizations')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--create-config', type=str, default=None,
                       help='Create a default configuration file at the specified path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to load (for debugging/memory constraints)')
    parser.add_argument('--memory-efficient', action='store_true', default=True,
                       help='Use memory-efficient data loading (default: True)')
    parser.add_argument('--monitor-gpu-memory', action='store_true', default=True,
                       help='Monitor GPU memory usage using nvidia-ml-py3 (default: True)')
    parser.add_argument('--num-thoughts', type=int, default=None,
                       help='Override num_thoughts parameter from config (truncates dataset if needed)')
    parser.add_argument('--demonstrate', type=str, default=None,
                       help='Demonstrate model generation from checkpoint (provide checkpoint path)')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of examples to generate in demonstration mode')
    
    args = parser.parse_args()
    
    # Check for nvidia-ml-py3 availability
    if args.monitor_gpu_memory and not NVML_AVAILABLE:
        print("Warning: nvidia-ml-py3 not available for GPU memory monitoring.")
        print("Install it with: pip install nvidia-ml-py3")
        print("Continuing without detailed GPU memory monitoring...")
    
    # Handle create-config option
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        model_config, training_config = load_config(args.config)
        
        # Override data directory if specified
        if args.data_dir:
            training_config['data_dir'] = args.data_dir
        
        # Override max samples if specified
        if args.max_samples:
            training_config['max_samples'] = args.max_samples
        
        # Set device
        if args.device:
            device = args.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Show GPU memory monitoring status
            if args.monitor_gpu_memory and NVML_AVAILABLE:
                print("GPU memory monitoring: Enabled (nvidia-ml-py3)")
            elif args.monitor_gpu_memory and not NVML_AVAILABLE:
                print("GPU memory monitoring: Disabled (nvidia-ml-py3 not available)")
            else:
                print("GPU memory monitoring: Disabled (--monitor-gpu-memory=False)")
        
        data_dir = training_config.get('data_dir', 'data/GSM8K')
        max_samples = training_config.get('max_samples', None)
        num_thoughts = model_config.get('num_thoughts', None)  # Extract from model config

        # Run demonstration if requested
        if args.demonstrate:
            print(f"\nRunning demonstration with checkpoint: {args.demonstrate}")
            demonstrate_model_from_checkpoint(
                checkpoint_path=args.demonstrate,
                model_config=model_config,
                data_dir=data_dir,
                num_examples=args.num_examples,
                device=device
            )
            return  # Exit after demonstration
        
        # Command-line argument takes precedence
        if args.num_thoughts is not None:
            num_thoughts = args.num_thoughts
            print(f"Overriding num_thoughts from config ({model_config.get('num_thoughts', 'not set')}) "
                  f"to command-line value: {num_thoughts}")
            # Update model config for consistency
            model_config['num_thoughts'] = num_thoughts
        
        print(f"Loading data from: {data_dir}")
        if num_thoughts is not None:
            print(f"Using num_thoughts: {num_thoughts}")
        
        # Initialize trainer first to use its memory-efficient loading method
        print("Initializing trainer...")
        trainer = GPT2VQVAETrainer(model_config, training_config, device=device)
        
        # Load data using memory-efficient method with num_thoughts truncation
        prompt_sequences, cot_sequences, prompt_mask, cot_mask = trainer.load_training_data_memory_efficient(
            data_dir, max_samples=max_samples, num_thoughts=num_thoughts
        )
        
        # Validate model-data compatibility
        validate_model_data_compatibility(model_config, prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        
        # Get validation split from config
        val_split = training_config.get('val_split', 0.1)
        
        # Start training
        print("Starting training...")
        trainer.train(
            prompt_sequences=prompt_sequences,
            cot_sequences=cot_sequences,
            prompt_mask=prompt_mask,
            cot_mask=cot_mask,
            val_split=val_split,
            resume_from=args.resume_from
        )
        
        # Save training history and memory usage
        checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
        history_path = os.path.join(checkpoint_dir, 'training_history.png')
        memory_path = os.path.join(checkpoint_dir, 'memory_usage.png')
        
        trainer.plot_training_history(history_path)
        trainer.plot_memory_usage(memory_path)
        
        print("Training completed successfully!")
        print(f"Best model saved to: {trainer.best_model_path}")
        print(f"Training history saved to: {history_path}")
        print(f"Memory usage plot saved to: {memory_path}")
        
        # Print final memory statistics
        if trainer.memory_stats:
            final_memory = trainer.memory_stats[-1]
            if 'pytorch_allocated_gb' in final_memory:
                print(f"Final memory usage: {final_memory['pytorch_allocated_gb']:.2f}GB allocated, {final_memory['pytorch_max_allocated_gb']:.2f}GB max")
            else:
                print(f"Final memory usage: {final_memory['allocated_gb']:.2f}GB allocated, {final_memory['max_allocated_gb']:.2f}GB max")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the configuration file and data files exist.")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run main function for command-line training
    main()