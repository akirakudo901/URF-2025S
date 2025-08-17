# Author: Akira Kudo
# Created: 2025/01/27
# Last Updated: 2025/01/27

"""
Token Assorted Training Script

This script implements the Token Assorted approach from the paper, which trains a language model
to handle mixed text and latent token inputs. The process involves:

1. Loading a pre-trained language model (e.g., Llama3.2 1B)
2. Loading a pre-trained VQVAE checkpoint
3. For each training step, randomly replacing portions of text with latent tokens
4. Training the language model to handle these mixed inputs

The key innovation is the random partial replacement process that creates inputs combining
text tokens and latent tokens from the VQVAE codebook.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional, Dict, Any, List
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml
import gc
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import random
from datetime import datetime
import socket

# Import the VQVAE model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE
import training

# Import compatible functions from train_utils
from trainer.train_utils import load_config, validate_model_data_compatibility, load_training_data

class TokenAssortedTrainer:
    """
    Trainer for the Token Assorted approach that combines text and latent tokens.
    """
    
    def __init__(self, 
                 llm_model_name: str = "meta-llama/Llama-2-1b-hf",
                 vqvae_checkpoint_path: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 replacement_ratio: float = 0.3,
                 min_replacement_length: int = 2,
                 max_replacement_length: int = 10,
                 special_token_id: int = 50257,  # Custom token ID for latent tokens
                 pad_token_id: int = 50256):
        """
        Initialize the Token Assorted trainer.
        
        Args:
            llm_model_name: Name of the pre-trained language model to load
            vqvae_checkpoint_path: Path to the pre-trained VQVAE checkpoint
            device: Device to run training on
            replacement_ratio: Ratio of tokens to replace with latent tokens
            min_replacement_length: Minimum length of replacement segments
            max_replacement_length: Maximum length of replacement segments
            special_token_id: Token ID to use for latent tokens
            pad_token_id: Token ID for padding
        """
        self.device = device
        self.replacement_ratio = replacement_ratio
        self.min_replacement_length = min_replacement_length
        self.max_replacement_length = max_replacement_length
        self.special_token_id = special_token_id
        self.pad_token_id = pad_token_id
        
        # Load the language model
        print(f"Loading language model: {llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # Add special token for latent tokens if not present
        if self.special_token_id not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<LATENT>']})
            self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.llm.to(device)
        
        # Load the VQVAE model
        if vqvae_checkpoint_path:
            print(f"Loading VQVAE checkpoint: {vqvae_checkpoint_path}")
            self.vqvae = self._load_vqvae_checkpoint(vqvae_checkpoint_path)
            self.vqvae.to(device)
            self.vqvae.eval()  # Set to evaluation mode
        else:
            raise Exception("vqvae_checkpoint_path must be provided (not None).")
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'replacement_stats': []
        }
        
    def _load_vqvae_checkpoint(self, checkpoint_path: str) -> EnhancedGPT2VQVAE:
        """
        Load a VQVAE checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Loaded VQVAE model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # Default configuration if not found
            model_config = {
                'vocab_size': 50257,
                'd_model': 768,
                'num_embeddings': 512,
                'commitment_cost': 0.25,
                'num_thoughts': 32,
                'n_positions': 1024
            }
        
        # Create model instance
        vqvae = EnhancedGPT2VQVAE(**model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            vqvae.load_state_dict(checkpoint['model_state_dict'])
        else:
            vqvae.load_state_dict(checkpoint)
        
        return vqvae
    
    def _encode_text_to_latent(self, 
                              prompt_sequences: torch.Tensor,
                              cot_sequences: torch.Tensor, 
                              prompt_masks: torch.Tensor,
                              cot_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompt and COT sequences to latent tokens using the VQVAE.
        
        Args:
            prompt_sequences: Prompt token sequences [batch_size, prompt_len]
            cot_sequences: Chain-of-thought token sequences [batch_size, num_thoughts, cot_len]
            prompt_masks: Attention masks for prompt sequences [batch_size, prompt_len]
            cot_masks: Attention masks for COT sequences [batch_size, num_thoughts, cot_len]
            
        Returns:
            Tuple of (latent_indices, latent_masks) where latent_indices has shape [batch_size, num_thoughts, cot_len]
        """
        if self.vqvae is None:
            # If no VQVAE, return random latent tokens
            batch_size, num_thoughts, cot_len = cot_sequences.shape
            num_embeddings = 512  # Default codebook size
            latent_indices = torch.randint(0, num_embeddings, (batch_size, num_thoughts, cot_len), device=self.device)
            return latent_indices, cot_masks
        
        with torch.no_grad():
            # Encode using VQVAE with proper prompt and COT sequences
            _, _, _, indices = self.vqvae.encode(
                prompt_sequences, cot_sequences, prompt_masks, cot_masks,
                quantize_cot_only=True, no_vq=False
            )
            
            # The VQVAE encode method returns indices for the COT portion only when quantize_cot_only=True
            # indices shape should be [batch_size, cot_len] for each thought
            # We need to reshape it to [batch_size, num_thoughts, cot_len]
            batch_size, num_thoughts, cot_len = cot_sequences.shape
            
            if indices.dim() == 2:  # [batch_size * num_thoughts, cot_len]
                # Reshape to [batch_size, num_thoughts, cot_len]
                latent_indices = indices.view(batch_size, num_thoughts, cot_len)
            else:  # [batch_size, cot_len] - need to expand for num_thoughts
                latent_indices = indices.unsqueeze(1).expand(-1, num_thoughts, -1)
            
            return latent_indices, cot_masks
    
    def _create_mixed_sequence(self, 
                              text_sequences: torch.Tensor,
                              text_masks: torch.Tensor,
                              latent_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Create mixed text-latent sequences using random partial replacement.
        
        Args:
            text_sequences: Original text sequences [batch_size, seq_len]
            text_masks: Attention masks for text sequences [batch_size, seq_len]
            latent_indices: Latent token indices [batch_size, seq_len]
            
        Returns:
            Tuple of (mixed_sequences, mixed_masks, replacement_stats)
        """
        batch_size, seq_len = text_sequences.shape
        mixed_sequences = text_sequences.clone()
        mixed_masks = text_masks.clone()
        
        replacement_stats = {
            'total_replacements': 0,
            'total_replaced_tokens': 0,
            'replacement_lengths': []
        }
        
        for inbatch_idx in range(batch_size):
            # Get valid positions (non-padded)
            valid_positions = torch.where(text_masks[inbatch_idx] == 1)[0]
            if len(valid_positions) == 0:
                continue
            
            # Calculate number of replacements
            num_replacements = max(1, int(len(valid_positions) * self.replacement_ratio))
            
            # Randomly select positions to replace
            replacement_positions = torch.randperm(len(valid_positions))[:num_replacements]
            
            for pos_idx in replacement_positions:
                start_pos = valid_positions[pos_idx].item()
                
                # Determine replacement length
                max_possible_length = min(
                    self.max_replacement_length,
                    seq_len - start_pos,
                    len(valid_positions) - pos_idx
                )
                
                if max_possible_length < self.min_replacement_length:
                    continue
                
                replacement_length = random.randint(self.min_replacement_length, max_possible_length)
                
                # Replace text tokens with latent tokens
                for i in range(replacement_length):
                    if start_pos + i < seq_len:
                        # Use the latent index as the token ID (offset by special_token_id)
                        latent_idx = latent_indices[inbatch_idx, start_pos + i].item()
                        mixed_sequences[inbatch_idx, start_pos + i] = self.special_token_id + latent_idx
                
                replacement_stats['total_replacements'] += 1
                replacement_stats['total_replaced_tokens'] += replacement_length
                replacement_stats['replacement_lengths'].append(replacement_length)
        
        return mixed_sequences, mixed_masks, replacement_stats
    
    def _decode_latent_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Decode latent token IDs back to text tokens using the VQVAE.
        
        Args:
            token_ids: Token IDs including latent tokens [batch_size, seq_len]
            
        Returns:
            Decoded text sequences
        """
        if self.vqvae is None:
            return token_ids
        
        batch_size, seq_len = token_ids.shape
        decoded_sequences = token_ids.clone()
        
        # Find latent token positions
        latent_mask = (token_ids >= self.special_token_id) & (token_ids < self.special_token_id + 512)
        
        if latent_mask.any():
            with torch.no_grad():
                # Extract latent indices
                latent_indices = token_ids[latent_mask] - self.special_token_id
                
                # Create dummy sequences for VQVAE decode
                dummy_prompt = torch.zeros((batch_size, 1), device=self.device)
                dummy_cot = torch.zeros((batch_size, 1, seq_len), device=self.device)
                
                # Decode using VQVAE (this is a simplified approach)
                # In practice, you might want to implement a more sophisticated decoding
                decoded_sequences[latent_mask] = self.pad_token_id  # Placeholder
        
        return decoded_sequences
    
    def prepare_dataset(self, 
                       prompt_sequences: torch.Tensor,
                       cot_sequences: torch.Tensor,
                       prompt_mask: torch.Tensor,
                       cot_mask: torch.Tensor,
                       val_split: float = 0.1) -> Tuple[Dataset, Dataset]:
        """
        Prepare datasets for training with mixed text-latent sequences.
        
        Args:
            prompt_sequences: Prompt sequences
            cot_sequences: Chain-of-thought sequences
            prompt_mask: Prompt attention masks
            cot_mask: COT attention masks
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        batch_size, K = prompt_sequences.shape
        _, M, L = cot_sequences.shape
        
        # Encode prompt and COT sequences to latent tokens
        latent_indices, latent_masks = self._encode_text_to_latent(
            prompt_sequences, cot_sequences, prompt_mask, cot_mask
        )
        
        # Flatten COT sequences and latent indices for processing
        cot_flat = cot_sequences.view(batch_size * M, L)
        cot_mask_flat = cot_mask.view(batch_size * M, L)
        latent_indices_flat = latent_indices.view(batch_size * M, L)
        
        # Create mixed sequences
        mixed_sequences = []
        mixed_masks = []
        replacement_stats_list = []
        
        for i in range(batch_size * M):
            # Create mixed sequence for this sample
            mixed_seq, mixed_mask, stats = self._create_mixed_sequence(
                cot_flat[i:i+1], cot_mask_flat[i:i+1], latent_indices_flat[i:i+1]
            )
            
            mixed_sequences.append(mixed_seq[0])
            mixed_masks.append(mixed_mask[0])
            replacement_stats_list.append(stats)
        
        # Stack sequences
        mixed_sequences = torch.stack(mixed_sequences)
        mixed_masks = torch.stack(mixed_masks)
        
        # Create labels (original COT sequences)
        labels = cot_flat.clone()
        
        # Split into train/val
        total_samples = len(mixed_sequences)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        
        train_indices, val_indices = random_split(
            range(total_samples), [train_size, val_size]
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': mixed_sequences[train_indices.indices].tolist(),
            'attention_mask': mixed_masks[train_indices.indices].tolist(),
            'labels': labels[train_indices.indices].tolist()
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': mixed_sequences[val_indices.indices].tolist(),
            'attention_mask': mixed_masks[val_indices.indices].tolist(),
            'labels': labels[val_indices.indices].tolist()
        })
        
        return train_dataset, val_dataset
    
    def train(self,
              train_dataset: Dataset,
              val_dataset: Dataset,
              training_args: TrainingArguments,
              save_dir: str = "token_assorted_checkpoints") -> Trainer:
        """
        Train the language model on mixed text-latent sequences.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            training_args: Training arguments
            save_dir: Directory to save checkpoints
            
        Returns:
            Trained trainer instance
        """
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.llm,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        print("Starting Token Assorted training...")
        trainer.train()
        
        # Save the final model
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(os.path.join(save_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "final_model"))
        
        return trainer
    
    def generate_with_mixed_input(self, 
                                 prompt: str,
                                 max_length: int = 100,
                                 temperature: float = 0.7,
                                 do_sample: bool = True) -> str:
        """
        Generate text using the trained model with mixed text-latent input.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def save_training_stats(self, save_path: str):
        """
        Save training statistics and replacement information.
        
        Args:
            save_path: Path to save the statistics
        """
        stats = {
            'training_history': self.training_history,
            'replacement_ratio': self.replacement_ratio,
            'min_replacement_length': self.min_replacement_length,
            'max_replacement_length': self.max_replacement_length,
            'special_token_id': self.special_token_id,
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname()
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Training statistics saved to {save_path}")


def create_token_assorted_config(output_path: str = "token_assorted_config.yaml"):
    """
    Create a default configuration file for Token Assorted training.
    
    Args:
        output_path: Path to save the configuration file
    """
    config = {
        'model': {
            'llm_model_name': 'meta-llama/Llama-2-1b-hf',
            'vqvae_checkpoint_path': 'path/to/vqvae_checkpoint.pt',
            'replacement_ratio': 0.3,
            'min_replacement_length': 2,
            'max_replacement_length': 10,
            'special_token_id': 50257,
            'pad_token_id': 50256
        },
        'training': {
            'output_dir': './token_assorted_output',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'learning_rate': 5e-5,
            'warmup_steps': 100,
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 500,
            'save_total_limit': 2,
            'prediction_loss_only': True,
            'dataloader_pin_memory': False,
            'remove_unused_columns': False
        },
        'data': {
            'data_dir': 'path/to/your/data',
            'max_samples': None,
            'num_thoughts': 32,
            'val_split': 0.1
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to {output_path}")


def main():
    """
    Main function to run Token Assorted training.
    """
    parser = argparse.ArgumentParser(description="Token Assorted Training")
    parser.add_argument("--config", type=str, default="token_assorted_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--create_config", action="store_true",
                       help="Create a default configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_token_assorted_config(args.config)
        return
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = TokenAssortedTrainer(
        llm_model_name=config['model']['llm_model_name'],
        vqvae_checkpoint_path=config['model']['vqvae_checkpoint_path'],
        replacement_ratio=config['model']['replacement_ratio'],
        min_replacement_length=config['model']['min_replacement_length'],
        max_replacement_length=config['model']['max_replacement_length'],
        special_token_id=config['model']['special_token_id'],
        pad_token_id=config['model']['pad_token_id']
    )
    
    # Load training data
    print("Loading training data...")
    data_loaded = load_training_data(
        data_dir=config['data']['data_dir'],
        max_samples=config['data']['max_samples'],
        num_thoughts=config['data']['num_thoughts']
    )
    
    # Extract training data from the returned tuple
    # train_utils.load_training_data returns (train_data, test_data) or (train_data, test_data, backpointers)
    if len(data_loaded) == 8:
        # Without backpointers
        prompt_sequences, cot_sequences, prompt_mask, cot_mask, _, _, _, _ = data_loaded
    elif len(data_loaded) == 10:
        # With backpointers
        prompt_sequences, cot_sequences, prompt_mask, cot_mask, _, _, _, _, _, _ = data_loaded
    else:
        raise ValueError(f"Unexpected number of return values from load_training_data: {len(data_loaded)}")
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset = trainer.prepare_dataset(
        prompt_sequences, cot_sequences, prompt_mask, cot_mask,
        val_split=config['data']['val_split']
    )
    
    # Create training arguments
    training_args = TrainingArguments(**config['training'])
    
    # Train the model
    print("Starting training...")
    trained_trainer = trainer.train(train_dataset, val_dataset, training_args)
    
    # Save training statistics
    stats_path = os.path.join(config['training']['output_dir'], 'training_stats.json')
    trainer.save_training_stats(stats_path)
    
    print("Training completed!")


if __name__ == "__main__":
    main() 