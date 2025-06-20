# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/20

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from transformers import GPT2Tokenizer
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb  # Optional: for experiment tracking

# Import the GPT2VQVAE model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vqvae_gpt2 import GPT2VQVAE

class GPT2VQVAETrainer:
    """
    Trainer class for GPT2VQVAE model with configurable hyperparameters.
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
        
        # Initialize model
        self.model = GPT2VQVAE(**model_config).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=training_config.get('pad_token_id', 0))
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.vq_losses = []
        self.perplexities = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
    def create_data_loader(self, 
                          prompt_sequences: torch.Tensor,
                          cot_sequences: torch.Tensor,
                          prompt_mask: torch.Tensor,
                          cot_mask: torch.Tensor,
                          batch_size: int,
                          shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for training/validation.
        
        Args:
            prompt_sequences: Tensor of shape [num_prompts, M, K]
            cot_sequences: Tensor of shape [num_prompts, M, L]
            prompt_mask: Tensor of shape [num_prompts, M, K]
            cot_mask: Tensor of shape [num_prompts, M, L]
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the dataset
        """
        dataset = TensorDataset(prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
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
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (prompts, cots, prompt_masks, cot_masks) in enumerate(progress_bar):
            # Move to device
            prompts = prompts.to(self.device)
            cots = cots.to(self.device)
            prompt_masks = prompt_masks.to(self.device)
            cot_masks = cot_masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output_sequences, output_logits, vq_loss, perplexity = self.model(
                prompt=prompts,
                cot_sequences=cots,
                cot_mask=cot_masks,
                prompt_mask=prompt_masks,
                inference=False,
                quantize_cot_only=self.training_config.get('quantize_cot_only', True)
            )
            
            # Calculate reconstruction loss
            # Reshape for cross entropy: [batch_size * M * L, vocab_size]
            logits_flat = output_logits.view(-1, output_logits.size(-1))
            targets_flat = cots.view(-1)
            
            # Apply mask to ignore padding tokens
            mask_flat = cot_masks.view(-1).bool()
            if mask_flat.sum() > 0:
                recon_loss = self.criterion(logits_flat[mask_flat], targets_flat[mask_flat])
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.training_config.get('gradient_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.training_config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'vq_loss': f"{vq_loss.item():.4f}",
                'perplexity': f"{perplexity.item():.2f}"
            })
        
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
        Validate the model.
        
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
                prompts = prompts.to(self.device)
                cots = cots.to(self.device)
                prompt_masks = prompt_masks.to(self.device)
                cot_masks = cot_masks.to(self.device)
                
                # Forward pass
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
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved with validation loss: {metrics['loss']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.vq_losses = checkpoint.get('vq_losses', [])
        self.perplexities = checkpoint.get('perplexities', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, 
              prompt_sequences: torch.Tensor,
              cot_sequences: torch.Tensor,
              prompt_mask: torch.Tensor,
              cot_mask: torch.Tensor,
              val_split: float = 0.1,
              resume_from: Optional[str] = None):
        """
        Train the model.
        
        Args:
            prompt_sequences: Training prompt sequences
            cot_sequences: Training CoT sequences
            prompt_mask: Training prompt masks
            cot_mask: Training CoT masks
            val_split: Fraction of data to use for validation
            resume_from: Path to checkpoint to resume from
        """
        # Split data into train and validation
        num_samples = len(prompt_sequences)
        val_size = int(num_samples * val_split)
        train_size = num_samples - val_size
        
        # Create train/val splits
        train_prompts = prompt_sequences[:train_size]
        train_cots = cot_sequences[:train_size]
        train_prompt_masks = prompt_mask[:train_size]
        train_cot_masks = cot_mask[:train_size]
        
        val_prompts = prompt_sequences[train_size:]
        val_cots = cot_sequences[train_size:]
        val_prompt_masks = prompt_mask[train_size:]
        val_cot_masks = cot_mask[train_size:]
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create data loaders
        train_loader = self.create_data_loader(
            train_prompts, train_cots, train_prompt_masks, train_cot_masks,
            batch_size=self.training_config['batch_size'], shuffle=True
        )
        
        val_loader = self.create_data_loader(
            val_prompts, val_cots, val_prompt_masks, val_cot_masks,
            batch_size=self.training_config['batch_size'], shuffle=False
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = len(self.train_losses) + 1
        
        # Training loop
        for epoch in range(start_epoch, self.training_config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.training_config['num_epochs']}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
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
            
            if (epoch + 1) % self.training_config.get('save_every', 5) == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_metrics, is_best)
        
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

def train_gpt2vqvae(prompt_sequences: torch.Tensor,
                    cot_sequences: torch.Tensor,
                    prompt_mask: torch.Tensor,
                    cot_mask: torch.Tensor,
                    model_config: Dict[str, Any],
                    training_config: Dict[str, Any],
                    val_split: float = 0.1,
                    resume_from: Optional[str] = None,
                    save_history: bool = True) -> GPT2VQVAETrainer:
    """
    Convenience function to train a GPT2VQVAE model.
    
    Args:
        prompt_sequences: Training prompt sequences
        cot_sequences: Training CoT sequences
        prompt_mask: Training prompt masks
        cot_mask: Training CoT masks
        model_config: Model hyperparameters
        training_config: Training hyperparameters
        val_split: Fraction of data to use for validation
        resume_from: Path to checkpoint to resume from
        save_history: Whether to save training history plot
        
    Returns:
        Trained GPT2VQVAETrainer instance
    """
    # Initialize trainer
    trainer = GPT2VQVAETrainer(model_config, training_config)
    
    # Train the model
    trainer.train(prompt_sequences, cot_sequences, prompt_mask, cot_mask, val_split, resume_from)
    
    # Save training history
    if save_history:
        history_path = os.path.join(
            training_config.get('checkpoint_dir', 'checkpoints'),
            'training_history.png'
        )
        trainer.plot_training_history(history_path)
    
    return trainer

# Example usage and configuration
def example_training():
    """
    Example of how to use the training functions.
    """
    # Load preprocessed data
    data_dir = "data/GSM8K"
    
    try:
        prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"))
        cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"))
        prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"))
        cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"))
        
        print(f"Loaded data shapes:")
        print(f"  prompt_sequences: {prompt_sequences.shape}")
        print(f"  cot_sequences: {cot_sequences.shape}")
        print(f"  prompt_mask: {prompt_mask.shape}")
        print(f"  cot_mask: {cot_mask.shape}")
        
        # Model configuration
        model_config = {
            'vocab_size': 50257,  # GPT2 vocabulary size
            'd_model': 768,       # GPT2 model dimension
            'num_embeddings': 512,  # VQ codebook size
            'commitment_cost': 0.25,  # VQ commitment cost
            'aggregation_hidden_dim': 1024,  # Aggregation MLP hidden dim
            'num_thoughts': 32,   # Number of parallel sequences
            'n_positions': 1024   # Maximum sequence length
        }
        
        # Training configuration
        training_config = {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'use_lr_scheduler': True,
            'min_lr': 1e-6,
            'num_epochs': 50,
            'batch_size': 4,  # Adjust based on your GPU memory
            'gradient_clip': 1.0,
            'vq_loss_weight': 1.0,
            'quantize_cot_only': True,
            'save_every': 5,
            'checkpoint_dir': 'checkpoints/gpt2vqvae',
            'pad_token_id': 0
        }
        
        # Train the model
        trainer = train_gpt2vqvae(
            prompt_sequences=prompt_sequences,
            cot_sequences=cot_sequences,
            prompt_mask=prompt_mask,
            cot_mask=cot_mask,
            model_config=model_config,
            training_config=training_config,
            val_split=0.1,
            save_history=True
        )
        
        print("Training completed successfully!")
        return trainer
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please run the data preprocessing first.")
        return None
    except Exception as e:
        print(f"Error during training: {e}")
        return None

if __name__ == "__main__":
    # Run example training
    trainer = example_training()
