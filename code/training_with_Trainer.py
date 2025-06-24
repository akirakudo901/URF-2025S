# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional, Dict, Any
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml
import gc
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import GPT2Tokenizer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils.import_utils import is_sagemaker_mp_enabled
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

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

class VQVAEDataset(torch.utils.data.Dataset):
    """
    Custom dataset for VQVAE training that handles prompt and CoT sequences.
    """
    
    def __init__(self, prompt_sequences, cot_sequences, prompt_mask, cot_mask):
        self.prompt_sequences = prompt_sequences
        self.cot_sequences = cot_sequences
        self.prompt_mask = prompt_mask
        self.cot_mask = cot_mask
        
    def __len__(self):
        return len(self.prompt_sequences)
    
    def __getitem__(self, idx):
        return {
            'prompt_sequences': self.prompt_sequences[idx],
            'cot_sequences': self.cot_sequences[idx],
            'prompt_mask': self.prompt_mask[idx],
            'cot_mask': self.cot_mask[idx]
        }

class VQVAEDataCollator:
    """
    Custom data collator for VQVAE training that handles batched prompt and CoT sequences.
    """
    
    def __init__(self, pad_token_id=50256):
        self.pad_token_id = pad_token_id
    
    def __call__(self, features):
        batch_size = len(features)
        
        # Extract sequences and masks
        prompt_sequences = [f['prompt_sequences'] for f in features]
        cot_sequences = [f['cot_sequences'] for f in features]
        prompt_masks = [f['prompt_mask'] for f in features]
        cot_masks = [f['cot_mask'] for f in features]
        
        # Stack into tensors
        prompt_sequences = torch.stack(prompt_sequences)
        cot_sequences = torch.stack(cot_sequences)
        prompt_masks = torch.stack(prompt_masks)
        cot_masks = torch.stack(cot_masks)
        
        return {
            'prompt_sequences': prompt_sequences,
            'cot_sequences': cot_sequences,
            'prompt_mask': prompt_masks,
            'cot_mask': cot_masks
        }

class VQVAETrainer(Trainer):
    """
    Custom Trainer class for VQVAE model that extends the transformers Trainer.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vq_losses = []
        self.perplexities = []
        self.detailed_metrics = []
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for VQVAE model.
        """
        # Extract inputs
        prompt_sequences = inputs['prompt_sequences']
        cot_sequences = inputs['cot_sequences']
        prompt_mask = inputs.get('prompt_mask', None)
        cot_mask = inputs.get('cot_mask', None)
        
        # Forward pass
        _, output_logits, vq_loss, perplexity = model(
            prompt=prompt_sequences,
            cot_sequences=cot_sequences,
            cot_mask=cot_mask,
            prompt_mask=prompt_mask,
            inference=False,
            quantize_cot_only=self.args.quantize_cot_only
        )
        
        # Compute reconstruction loss
        logits_flat = output_logits.reshape(-1, output_logits.size(-1))
        targets_flat = cot_sequences.view(-1)
        mask_flat = cot_mask.view(-1).bool() if cot_mask is not None else torch.ones_like(targets_flat, dtype=torch.bool)
        
        recon_loss = torch.tensor(0.0, device=model.device)
        if mask_flat.sum() > 0:
            recon_loss = nn.CrossEntropyLoss(ignore_index=self.args.pad_token_id)(
                logits_flat[mask_flat], targets_flat[mask_flat]
            )
        
        # Total loss
        total_loss = recon_loss + self.args.vq_loss_weight * vq_loss
        
        # Store metrics for logging
        self.vq_losses.append(vq_loss.item())
        self.perplexities.append(perplexity.item())
        
        if return_outputs:
            return total_loss, (output_logits, vq_loss, perplexity)
        return total_loss
    
    def log(self, logs):
        """
        Custom logging to include VQVAE-specific metrics.
        """
        # Add VQVAE metrics to logs
        if self.vq_losses:
            logs['vq_loss'] = sum(self.vq_losses) / len(self.vq_losses)
            logs['perplexity'] = sum(self.perplexities) / len(self.perplexities)
            # Clear for next logging step
            self.vq_losses.clear()
            self.perplexities.clear()
        
        super().log(logs)
    
    def create_optimizer(self):
        """
        Custom optimizer creation with quantization support.
        """
        if self.optimizer is not None:
            return
        
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        # Create optimizer grouped parameters
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Use regular AdamW for now (8-bit optimizer can be added later if needed)
        optimizer_cls = optim.AdamW
        
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Custom scheduler creation.
        """
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        
        if self.args.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        elif self.args.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        else:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        
        return self.lr_scheduler

class VQVAETrainingArguments(TrainingArguments):
    """
    Custom TrainingArguments class that includes VQVAE-specific parameters.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract VQVAE-specific parameters
        self.pad_token_id = kwargs.pop('pad_token_id', 50256)
        self.vq_loss_weight = kwargs.pop('vq_loss_weight', 1.0)
        self.quantize_cot_only = kwargs.pop('quantize_cot_only', True)
        
        super().__init__(*args, **kwargs)

def create_optimized_training_args(training_config: Dict[str, Any], output_dir: str) -> VQVAETrainingArguments:
    """
    Create optimized TrainingArguments with all requested optimizations.
    
    Args:
        training_config: Training configuration dictionary
        output_dir: Output directory for checkpoints and logs
        
    Returns:
        VQVAETrainingArguments object with optimizations enabled
    """
    return VQVAETrainingArguments(
        output_dir=output_dir,
        # Basic training settings
        num_train_epochs=training_config.get('num_epochs', 50),
        per_device_train_batch_size=training_config.get('batch_size', 2),
        per_device_eval_batch_size=training_config.get('batch_size', 2),
        learning_rate=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 0.01),
        
        # Gradient accumulation and checkpointing
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        gradient_checkpointing=training_config.get('use_gradient_checkpointing', True),
        
        # Mixed precision and optimization
        fp16=training_config.get('use_mixed_precision', True),
        bf16=training_config.get('use_bf16', False),  # Use bfloat16 if available
        dataloader_pin_memory=True,  # Data preloading
        dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
        
        # Optimizer settings
        adam_beta1=training_config.get('beta1', 0.9),
        adam_beta2=training_config.get('beta2', 0.999),
        adam_epsilon=training_config.get('adam_epsilon', 1e-8),
        
        # Learning rate scheduling
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_steps=training_config.get('warmup_steps', 0),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        
        # Evaluation and saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=training_config.get('logging_steps', 10),
        logging_first_step=True,
        report_to=None,  # Disable wandb/tensorboard for now
        
        # Compilation and optimization
        torch_compile=training_config.get('use_torch_compile', True),
        torch_compile_mode=training_config.get('torch_compile_mode', 'reduce-overhead'),
        
        # SDPA optimizations
        use_flash_attention_2=training_config.get('use_flash_attention_2', True),
        attn_implementation=training_config.get('attn_implementation', 'sdpa'),
        
        # Memory optimizations
        max_grad_norm=training_config.get('gradient_clip', 1.0),
        remove_unused_columns=False,  # Keep all columns for custom dataset
        
        # Custom settings for VQVAE
        pad_token_id=training_config.get('pad_token_id', 50256),
        vq_loss_weight=training_config.get('vq_loss_weight', 1.0),
        quantize_cot_only=training_config.get('quantize_cot_only', True),
        
        # Other optimizations
        ddp_find_unused_parameters=False,
        dataloader_prefetch_factor=training_config.get('dataloader_prefetch_factor', 2),
        group_by_length=training_config.get('group_by_length', False),
        length_column_name="length",
        ignore_data_skip=False,
        dataloader_drop_last=False,
        
        # Debugging
        seed=training_config.get('seed', 42),
        data_seed=training_config.get('data_seed', 42),
        deterministic=False,
        
        # Prediction
        predict_with_generate=False,
        generation_max_length=None,
        generation_num_beams=1,
        
        # Other
        label_names=None,
        push_to_hub=False,
        resume_from_checkpoint=None,
        hub_model_id=None,
        hub_strategy="every_save",
        hub_token=None,
        hub_private_repo=False,
        gradient_checkpointing_kwargs=None,
        include_inputs_for_metrics=False,
        fp16_full_eval=False,
        dataloader_pin_memory_device="",
        skip_memory_metrics=False,
        use_mps_device=False,
        torch_compile_backend="inductor",
        optim="adamw_torch",
        optim_args=None,
        adafactor=False,
        adafactor_kwargs=None,
        ddp_kwargs=None,
        dataloader_kwargs=None,
        remove_unused_columns=None,
        label_smoothing_factor=0.0,
        include_num_input_tokens_seen=False,
        neftune_noise_alpha=None,
    )

def load_training_data(data_dir: str, max_samples: Optional[int] = None, num_thoughts: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load training data with memory-efficient loading and truncate based on num_thoughts.
    
    Args:
        data_dir: Directory containing the preprocessed data files
        max_samples: Maximum number of samples to load (for debugging/memory constraints)
        num_thoughts: Number of parallel sequences to use (truncates dataset if needed)
        
    Returns:
        Tuple of (prompt_sequences, cot_sequences, prompt_mask, cot_mask)
    """
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
    
    # Load tensors
    prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"), map_location='cpu')
    cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"), map_location='cpu')
    prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"), map_location='cpu')
    cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"), map_location='cpu')
    
    print(f"Original data shapes:")
    print(f"  prompt_sequences: {prompt_sequences.shape}")
    print(f"  cot_sequences: {cot_sequences.shape}")
    print(f"  prompt_mask: {prompt_mask.shape}")
    print(f"  cot_mask: {cot_mask.shape}")
    
    # Validate and reorganize based on num_thoughts
    if num_thoughts is not None:
        current_num_thoughts = cot_sequences.shape[1]
        print(f"Current num_thoughts in dataset: {current_num_thoughts}")
        print(f"Requested num_thoughts: {num_thoughts}")
        
        if current_num_thoughts < num_thoughts:
            raise ValueError(f"Dataset only has {current_num_thoughts} parallel sequences, "
                           f"but model requires {num_thoughts}. Please regenerate dataset with more sequences.")
        
        if current_num_thoughts > num_thoughts:
            num_batches = current_num_thoughts // num_thoughts
            remainder = current_num_thoughts % num_thoughts
            
            if remainder > 0:
                print(f"Warning: {current_num_thoughts} is not perfectly divisible by {num_thoughts}")
                print(f"Will use {num_batches * num_thoughts} sequences (dropping {remainder} sequences)")
            
            print(f"Reorganizing dataset: {current_num_thoughts} sequences → {num_batches} batches of {num_thoughts} sequences each")
            
            original_batch_size = prompt_sequences.shape[0]
            new_batch_size = original_batch_size * num_batches
            usable_sequences = num_batches * num_thoughts
            
            def reorganize_sequence_pair(sequence_1d, sequence_2d):
                """Helper to reorganize a pair of prompt/cot sequences or masks"""
                sequence_1d = sequence_1d.unsqueeze(1).repeat(1, num_batches, 1).reshape(-1, sequence_1d.size(-1))
                sequence_2d = sequence_2d[:, :usable_sequences, :]
                sequence_2d = sequence_2d.view(original_batch_size, num_batches, num_thoughts, -1)
                sequence_2d = sequence_2d.transpose(1, 2).contiguous().view(new_batch_size, num_thoughts, -1)
                return sequence_1d, sequence_2d
            
            prompt_sequences, cot_sequences = reorganize_sequence_pair(prompt_sequences, cot_sequences)
            prompt_mask, cot_mask = reorganize_sequence_pair(prompt_mask, cot_mask)
            
            print(f"Reorganized data shapes:")
            print(f"  prompt_sequences: {prompt_sequences.shape}")
            print(f"  cot_sequences: {cot_sequences.shape}")
            print(f"  prompt_mask: {prompt_mask.shape}")
            print(f"  cot_mask: {cot_mask.shape}")
            print(f"  Dataset size increased from {original_batch_size} to {new_batch_size} samples")
    
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
    
    print("✅ Model-data compatibility validation complete!")

def load_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration from a YAML or JSON file.
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
    Create a default configuration file with all optimizations enabled.
    """
    default_config = {
        'model_config': {
            'vocab_size': 50257,
            'd_model': 768,
            'num_embeddings': 512,
            'commitment_cost': 0.25,
            'aggregation_hidden_dim': 1024,
            'num_thoughts': 40,
            'n_positions': 1024,
            'use_pretrained_encoder': True,
            'use_pretrained_decoder': True,
            'pretrained_model_name': 'gpt2'
        },
        'training_config': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'adam_epsilon': 1e-8,
            'num_epochs': 50,
            'batch_size': 2,
            'gradient_clip': 1.0,
            'vq_loss_weight': 1.0,
            'quantize_cot_only': True,
            'pad_token_id': 50256,
            'val_split': 0.1,
            
            # Memory optimizations
            'use_mixed_precision': True,
            'use_bf16': False,  # Set to True for newer GPUs
            'gradient_accumulation_steps': 4,
            'use_gradient_checkpointing': True,
            
            # Optimizer optimizations
            'use_8bit_optimizer': False,  # Set to True to use 8-bit AdamW
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            
            # Data loading optimizations
            'dataloader_num_workers': 4,
            'dataloader_prefetch_factor': 2,
            'group_by_length': False,
            
            # Compilation and attention optimizations
            'use_torch_compile': True,
            'torch_compile_mode': 'reduce-overhead',
            'use_flash_attention_2': True,
            'attn_implementation': 'sdpa',
            
            # Other settings
            'save_total_limit': 3,
            'logging_steps': 10,
            'seed': 42,
            'data_seed': 42,
            'max_samples': None
        },
        'data_config': {
            'data_dir': 'data/GSM8K'
        }
    }
    
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    else:
        output_path = output_path + '.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to: {output_path}")
    print("Optimizations included:")
    print("  - Mixed precision training (fp16/bf16)")
    print("  - Gradient accumulation (4 steps)")
    print("  - Gradient checkpointing")
    print("  - 8-bit optimizer (optional)")
    print("  - Data preloading with multiple workers")
    print("  - torch.compile for speedup")
    print("  - Flash Attention 2 and SDPA optimizations")
    print("  - Cosine learning rate scheduling")
    print("  - Memory-efficient data loading")

def main():
    """
    Main function for training with transformers Trainer.
    """
    parser = argparse.ArgumentParser(description='Train GPT2VQVAE model with transformers Trainer')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--output-dir', type=str, default='checkpoints/gpt2vqvae',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--create-config', type=str, default=None,
                       help='Create a default configuration file at the specified path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to load (for debugging/memory constraints)')
    parser.add_argument('--num-thoughts', type=int, default=None,
                       help='Override num_thoughts parameter from config')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Handle create-config option
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        model_config, training_config = load_config(args.config)
        
        # Override settings
        if args.data_dir:
            training_config['data_dir'] = args.data_dir
        if args.max_samples:
            training_config['max_samples'] = args.max_samples
        if args.num_thoughts:
            model_config['num_thoughts'] = args.num_thoughts
        
        # Set device
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if device == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Check for optimizations
            if training_config.get('use_bf16', False):
                if torch.cuda.is_bf16_supported():
                    print("bfloat16 training: Supported")
                else:
                    print("bfloat16 training: Not supported, falling back to fp16")
                    training_config['use_bf16'] = False
                    training_config['use_mixed_precision'] = True
            
            if training_config.get('use_flash_attention_2', True):
                try:
                    import flash_attn
                    print("Flash Attention 2: Available")
                except ImportError:
                    print("Flash Attention 2: Not available, using SDPA")
                    training_config['use_flash_attention_2'] = False
        
        # Load data
        data_dir = training_config.get('data_dir', 'data/GSM8K')
        max_samples = training_config.get('max_samples', None)
        num_thoughts = model_config.get('num_thoughts', None)
        
        print(f"Loading data from: {data_dir}")
        prompt_sequences, cot_sequences, prompt_mask, cot_mask = load_training_data(
            data_dir, max_samples=max_samples, num_thoughts=num_thoughts
        )
        
        # Validate model-data compatibility
        validate_model_data_compatibility(model_config, prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        
        # Create datasets
        dataset = VQVAEDataset(prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        
        # Split into train/validation
        val_split = training_config.get('val_split', 0.1)
        num_samples = len(dataset)
        val_size = int(num_samples * val_split)
        train_size = num_samples - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(training_config.get('seed', 42))
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Initialize model
        print("Initializing model...")
        model = GPT2VQVAE(**model_config).to(device)
        
        # Enable gradient checkpointing if specified
        if training_config.get('use_gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Create training arguments
        training_args = create_optimized_training_args(training_config, args.output_dir)
        
        # Override resume from checkpoint
        if args.resume_from:
            training_args.resume_from_checkpoint = args.resume_from
        
        # Create data collator
        data_collator = VQVAEDataCollator(pad_token_id=training_config.get('pad_token_id', 50256))
        
        # Create trainer
        trainer = VQVAETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("Starting training with transformers Trainer...")
        trainer.train(resume_from_checkpoint=args.resume_from)
        
        # Save final model
        trainer.save_model()
        print(f"Training completed! Model saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()