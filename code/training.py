# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import torch
import torch.optim as optim
from typing import Optional, Dict, Any
import os
import argparse
import gc
# import psutil
from torch.amp.autocast_mode import autocast
import traceback
import sys
import os

# Import the GPT2VQVAE model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from demonstrate import demonstrate_custom_prompt_cot, demonstrate_model_from_checkpoint

from trainer.auto_switching_trainer import AutoSwitchingTrainer
from trainer.enhanced_trainer import EnhancedGPT2VQVAETrainer
from trainer.gpt2_vqvae_trainer import GPT2VQVAETrainer, TrainingAbortedException, save_training_visualizations
from trainer.phased_trainer import PhasedEnhancedGPT2VQVAETrainer

from trainer.train_utils import (
    create_default_config, compute_reconstruction_loss, load_config, load_training_data, 
    validate_model_data_compatibility
    )

from vqvae_gpt2_simple import SimpleGPT2VQVAE

# GPU memory monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: nvidia-ml-py3 not available. Install with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False


def main():
    """
    Main function for command-line training with memory optimizations.
    """
    parser = argparse.ArgumentParser(description='Train GPT2VQVAE model with memory optimizations and perplexity threshold monitoring')
    parser.add_argument('--config', '-c', type=str, #not required anymore as it will be skipped when demonstrating
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
    parser.add_argument('--monitor-gpu-memory', action='store_true', default=True,
                       help='Monitor GPU memory usage using nvidia-ml-py3 (default: True)')
    parser.add_argument('--num-thoughts', type=int, default=None,
                       help='Override num_thoughts parameter from config (truncates dataset if needed)')
    parser.add_argument('--demonstrate', type=str, default=None,
                       help='Demonstrate model generation from checkpoint (provide checkpoint path)')
    parser.add_argument('--demonstrate-custom', type=str, default=None,
                       help='Demonstrate model generation from checkpoint using custom prompt-CoT files (provide checkpoint path)')
    parser.add_argument('--prompt-file', type=str, default='test/test_prompt.txt',
                       help='Path to file containing custom prompt (used with --demonstrate-custom)')
    parser.add_argument('--cot-file', type=str, default='test/test_cot.txt',
                       help='Path to file containing custom CoT (used with --demonstrate-custom)')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of examples to generate in demonstration mode')
    parser.add_argument('--demo-seed', type=int, default=42,
                       help='Random seed for demonstration sampling (default: 42)')
    parser.add_argument('--perplexity-threshold', type=float, default=None,
                       help='Training will abort when 20-step average perplexity goes below this value (default: 1.5)')
    parser.add_argument('--perplexity-window-size', type=int, default=None,
                       help='Number of steps to average for perplexity threshold check (default: 20)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Override checkpoint directory from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config')
    parser.add_argument('--min-lr', type=float, default=None,
                       help='Override minimum learning rate from config')
    parser.add_argument('--vq-loss-weight', type=float, default=None,
                       help='Override VQ loss weight from config')
    parser.add_argument('--minimum-batches-for-checkpoint', type=int, default=None,
                       help='Override minimum batches required to save aborted checkpoint (default: 200)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--simple', action='store_true', default=False,
                       help='Use SimpleGPT2VQVAETrainer and SimpleGPT2VQVAE model (default: False)')
    parser.add_argument('--enhanced', action='store_true', default=False,
                       help='Use EnhancedGPT2VQVAETrainer and EnhancedGPT2VQVAE model (default: False)')
    parser.add_argument('--phased', action='store_true', default=False,
                       help='Use PhasedEnhancedGPT2VQVAETrainer for phased training (requires --enhanced)')
    parser.add_argument('--auto', action='store_true', default=False,
                       help='Use AutoSwitchingTrainer for auto-switching phased training (requires --enhanced and --phased)')
    parser.add_argument('--use-vq', action='store_true', default=None,
                       help='Enable vector quantization (default: True for SimpleGPT2VQVAE, True for GPT2VQVAE)')
    parser.add_argument('--no-vq', action='store_true', default=False,
                       help='Disable vector quantization (passes encoder outputs directly to decoder)')
    
    # Enhanced VQ-VAE specific arguments
    parser.add_argument('--ema-decay', type=float, default=None,
                       help='Override EMA decay rate for enhanced VQ-VAE (default: 0.99)')
    parser.add_argument('--diversity-gamma', type=float, default=None,
                       help='Override diversity loss weight for enhanced VQ-VAE (default: 0.1)')
    parser.add_argument('--reset-threshold', type=float, default=None,
                       help='Override codebook reset threshold for enhanced VQ-VAE (default: 0.1)')
    parser.add_argument('--reset-frequency', type=int, default=None,
                       help='Override codebook reset frequency for enhanced VQ-VAE (default: 1000)')
    parser.add_argument('--disable-ema', action='store_true', default=False,
                       help='Disable EMA updates in enhanced VQ-VAE')
    parser.add_argument('--enhanced-codebook-tracking', action='store_true', default=None,
                       help='Enable enhanced codebook tracking (default: True for enhanced VQ-VAE)')
    parser.add_argument('--no-enhanced-codebook-tracking', action='store_true', default=False,
                       help='Disable enhanced codebook tracking')
    
    # Phased training specific arguments
    parser.add_argument('--initialization-steps', type=int, default=None,
                       help='Override initialization_steps for phased training (default: 1500)')
    parser.add_argument('--r-reestim', type=int, default=None,
                       help='Override r_reestim for phased training (default: 500)')
    parser.add_argument('--quantization-start', type=int, default=None,
                       help='Override quantization_start for phased training (default: 5000)')
    parser.add_argument('--codebook-lr-multiplier', type=float, default=None,
                       help='Override codebook_lr_multiplier for phased training (default: 1.0)')
    
    # Auto-switching specific arguments
    parser.add_argument('--auto-switch-patience', type=int, default=None,
                       help='Override auto_switch_patience for auto-switching training (default: 3)')
    parser.add_argument('--auto-switch-validation-checks', type=int, default=None,
                       help='Override auto_switch_validation_checks_per_epoch for auto-switching training (default: 5)')
    
    parser.add_argument('--do-figure-analyses', dest='do_figure_analyses', action='store_true', default=True,
                       help='Enable figure analyses and visualizations in demonstration mode (default: True)')
    parser.add_argument('--no-figure-analyses', dest='do_figure_analyses', action='store_false',
                       help='Disable figure analyses and visualizations in demonstration mode')
    
    args = parser.parse_args()
    
    # Validate phased training arguments
    if args.phased and not args.enhanced:
        print("Error: --phased requires --enhanced to be enabled")
        print("Please use both --enhanced and --phased flags together")
        return
    
    # Validate auto-switching arguments
    if args.auto and not (args.enhanced and args.phased):
        print("Error: --auto requires both --enhanced and --phased to be enabled")
        print("Please use --enhanced, --phased, and --auto flags together")
        return

    if not args.config and not (args.demonstrate or args.demonstrate_custom):
        print("Error: --config required unless --demonstrate or --demonstrate-custom is specified")
        return
    
    # Check for nvidia-ml-py3 availability
    if args.monitor_gpu_memory and not NVML_AVAILABLE:
        print("Warning: nvidia-ml-py3 not available for GPU memory monitoring.")
        print("Install it with: pip install nvidia-ml-py3")
        print("Continuing without detailed GPU memory monitoring...")
    
    # Handle create-config option
    if args.create_config:
        # Determine if enhanced VQ-VAE config is requested
        create_default_config(args.create_config, enhanced_vq=args.enhanced, phased_training=args.phased)
        return
    
    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            model_config, training_config = load_config(args.config)
            # Set run name to be used for push notifications
            run_name = os.path.basename(args.config)
        else:
            model_config, training_config = {}, {}
        
        # Override data directory if specified
        if args.data_dir:
            training_config['data_dir'] = args.data_dir
            print(f"Overriding data_dir from config to: {args.data_dir}")
        
        # Override max samples if specified
        if args.max_samples:
            training_config['max_samples'] = args.max_samples
            print(f"Overriding max_samples from config to: {args.max_samples}")
        
        # Override perplexity threshold if specified
        if args.perplexity_threshold:
            training_config['perplexity_threshold'] = args.perplexity_threshold
            print(f"Overriding perplexity_threshold from config to: {args.perplexity_threshold}")
        
        if args.perplexity_window_size:
            training_config['perplexity_window_size'] = args.perplexity_window_size
            print(f"Overriding perplexity_window_size from config to: {args.perplexity_window_size}")
        
        # Override checkpoint directory if specified
        if args.checkpoint_dir:
            training_config['checkpoint_dir'] = args.checkpoint_dir
            print(f"Overriding checkpoint_dir from config to: {args.checkpoint_dir}")
        
        # Override learning rate if specified
        if args.lr:
            training_config['learning_rate'] = args.lr
            print(f"Overriding learning_rate from config to: {args.lr}")
        
        # Override minimum learning rate if specified
        if args.min_lr:
            training_config['min_lr'] = args.min_lr
            print(f"Overriding min_lr from config to: {args.min_lr}")
        
        # Override VQ loss weight if specified
        if args.vq_loss_weight:
            training_config['vq_loss_weight'] = args.vq_loss_weight
            print(f"Overriding vq_loss_weight from config to: {args.vq_loss_weight}")
        
        # Override minimum batches for checkpoint if specified
        if args.minimum_batches_for_checkpoint:
            training_config['minimum_batches_for_checkpoint'] = args.minimum_batches_for_checkpoint
            print(f"Overriding minimum_batches_for_checkpoint from config to: {args.minimum_batches_for_checkpoint}")
        
        # Override batch size if specified
        if args.batch_size:
            training_config['batch_size'] = args.batch_size
            print(f"Overriding batch_size from config to: {args.batch_size}")
        
        # Handle use_vq argument
        if args.no_vq:
            training_config['use_vq'] = False
            print("Disabling vector quantization (--no-vq flag)")
        elif args.use_vq is not None:
            training_config['use_vq'] = args.use_vq
            print(f"Setting use_vq to: {args.use_vq}")
        else:
            # Default behavior: use_vq=True for both models
            training_config['use_vq'] = True
            print("Using default use_vq=True")
        
        # Handle enhanced VQ-VAE specific arguments
        if args.enhanced:
            print("Enhanced VQ-VAE mode enabled")
            
            # Override enhanced VQ-VAE parameters if specified
            if args.ema_decay is not None:
                model_config['ema_decay'] = args.ema_decay
                print(f"Overriding ema_decay to: {args.ema_decay}")
            
            if args.diversity_gamma is not None:
                model_config['diversity_gamma'] = args.diversity_gamma
                print(f"Overriding diversity_gamma to: {args.diversity_gamma}")
            
            if args.reset_threshold is not None:
                model_config['reset_threshold'] = args.reset_threshold
                print(f"Overriding reset_threshold to: {args.reset_threshold}")
            
            if args.reset_frequency is not None:
                model_config['reset_frequency'] = args.reset_frequency
                print(f"Overriding reset_frequency to: {args.reset_frequency}")
            
            if args.disable_ema:
                model_config['use_ema'] = False
                print("Disabling EMA updates (--disable-ema flag)")
            
            # Handle enhanced codebook tracking
            if args.no_enhanced_codebook_tracking:
                training_config['enhanced_codebook_tracking'] = False
                print("Disabling enhanced codebook tracking (--no-enhanced-codebook-tracking flag)")
            elif args.enhanced_codebook_tracking is not None:
                training_config['enhanced_codebook_tracking'] = args.enhanced_codebook_tracking
                print(f"Setting enhanced_codebook_tracking to: {args.enhanced_codebook_tracking}")
            else:
                # Default behavior: enable enhanced codebook tracking for enhanced VQ-VAE
                training_config['enhanced_codebook_tracking'] = True
                print("Using default enhanced_codebook_tracking=True for enhanced VQ-VAE")
            
            # Handle phased training parameters
            if args.phased:
                print("Phased training mode enabled")
                
                # Override phased training parameters if specified
                if args.initialization_steps is not None:
                    training_config['initialization_steps'] = args.initialization_steps
                    print(f"Overriding initialization_steps to: {args.initialization_steps}")
                
                if args.r_reestim is not None:
                    training_config['r_reestim'] = args.r_reestim
                    print(f"Overriding r_reestim to: {args.r_reestim}")
                
                if args.quantization_start is not None:
                    training_config['quantization_start'] = args.quantization_start
                    print(f"Overriding quantization_start to: {args.quantization_start}")
                
                if args.codebook_lr_multiplier is not None:
                    training_config['codebook_lr_multiplier'] = args.codebook_lr_multiplier
                    print(f"Overriding codebook_lr_multiplier to: {args.codebook_lr_multiplier}")
                
                # Handle auto-switching parameters
                if args.auto:
                    print("Auto-switching mode enabled")
                    
                    # Override auto-switching parameters if specified
                    if args.auto_switch_patience is not None:
                        training_config['auto_switch_patience'] = args.auto_switch_patience
                        print(f"Overriding auto_switch_patience to: {args.auto_switch_patience}")
                    
                    if args.auto_switch_validation_checks is not None:
                        training_config['auto_switch_validation_checks_per_epoch'] = args.auto_switch_validation_checks
                        print(f"Overriding auto_switch_validation_checks_per_epoch to: {args.auto_switch_validation_checks}")
        else:
            # For non-enhanced models, disable enhanced codebook tracking
            training_config['enhanced_codebook_tracking'] = False
        
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
            if args.auto or args.phased or args.enhanced:
                model_type = "EnhancedGPT2VQVAE"  # Auto-switching and phased trainers use EnhancedGPT2VQVAE model
            elif args.simple:
                model_type = "SimpleGPT2VQVAE"
            else:
                model_type = "GPT2VQVAE"
            print(f"\nRunning demonstration with checkpoint: {args.demonstrate}")
            print(f"Using model type: {model_type}")
            demonstrate_model_from_checkpoint(
                checkpoint_path=args.demonstrate,
                data_dir=data_dir,
                num_examples=args.num_examples,
                device=device,
                use_vq=training_config.get('use_vq', True),
                model_type=model_type,
                seed=args.demo_seed, 
                do_figure_analyses=args.do_figure_analyses,
                **model_config
            )
            return  # Exit after demonstration
        
        # Run custom demonstration if requested
        if args.demonstrate_custom:
            if args.auto or args.phased or args.enhanced:
                model_type = "EnhancedGPT2VQVAE"  # Auto-switching and phased trainers use EnhancedGPT2VQVAE model
            elif args.simple:
                model_type = "SimpleGPT2VQVAE"
            else:
                model_type = "GPT2VQVAE"
            print(f"\nRunning custom demonstration with checkpoint: {args.demonstrate_custom}")
            print(f"Using model type: {model_type}")
            print(f"Prompt file: {args.prompt_file}")
            print(f"CoT file: {args.cot_file}")
            demonstrate_custom_prompt_cot(
                checkpoint_path=args.demonstrate_custom,
                prompt_file=args.prompt_file,
                cot_file=args.cot_file,
                device=device,
                use_vq=training_config.get('use_vq', True),
                model_type=model_type,
                do_figure_analyses=args.do_figure_analyses,
                **model_config
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
        
        # If resume_from, initialize weights randomly
        if args.resume_from:
            model_config['use_pretrained_encoder'] = False
            model_config['use_pretrained_decoder'] = False
            if args.auto:
                trainer = AutoSwitchingTrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.phased:
                trainer = PhasedEnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.enhanced:
                trainer = EnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.simple:
                trainer = SimpleGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            else:
                trainer = GPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            print(f"Resuming from checkpoint: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        else:
            if args.auto:
                trainer = AutoSwitchingTrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.phased:
                trainer = PhasedEnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.enhanced:
                trainer = EnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.simple:
                trainer = SimpleGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            else:
                trainer = GPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)

        # Load training and test data using memory-efficient method with num_thoughts truncation
        train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
        test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = load_training_data(
            data_dir, max_samples=max_samples, num_thoughts=num_thoughts
        )
        
        # Validate model-data compatibility for both train and test sets
        print("Validating train data compatibility...")
        validate_model_data_compatibility(model_config, train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask)
        print("Validating test data compatibility...")
        validate_model_data_compatibility(model_config, test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
        
        # Start training
        print("Starting training...")
        print(f"Perplexity threshold monitoring: {training_config.get('perplexity_threshold', 1.5)} (window size: {training_config.get('perplexity_window_size', 20)})")
        print(f"Minimum batches for aborted checkpoint: {training_config.get('minimum_batches_for_checkpoint', 200)}")
        trainer.train(
            train_prompt_sequences=train_prompt_sequences,
            train_cot_sequences=train_cot_sequences,
            train_prompt_mask=train_prompt_mask,
            train_cot_mask=train_cot_mask,
            test_prompt_sequences=test_prompt_sequences,
            test_cot_sequences=test_cot_sequences,
            test_prompt_mask=test_prompt_mask,
            test_cot_mask=test_cot_mask,
            resume_from=args.resume_from,
            num_measurements_per_epoch=training_config.get('num_measurements_per_epoch', 20)
        )
        
        
        print(f"Best model saved to: {trainer.best_model_path}")
        
        # Print final memory statistics
        if trainer.memory_stats:
            final_memory = trainer.memory_stats[-1]
            if 'pytorch_allocated_gb' in final_memory:
                print(f"Final memory usage: {final_memory['pytorch_allocated_gb']:.2f}GB allocated, {final_memory['pytorch_max_allocated_gb']:.2f}GB max")
            else:
                print(f"Final memory usage: {final_memory['allocated_gb']:.2f}GB allocated, {final_memory['max_allocated_gb']:.2f}GB max")
        
    except TrainingAbortedException as e:
        # Handle training abortion gracefully
        print(f"\n✅ Training completed successfully (aborted due to {e.reason})")
        print(f"Training stopped at epoch {e.epoch} with final perplexity: {e.final_perplexity:.4f}")
        
        # Print final memory statistics for aborted training
        if trainer.memory_stats:
            final_memory = trainer.memory_stats[-1]
            if 'pytorch_allocated_gb' in final_memory:
                print(f"Final memory usage: {final_memory['pytorch_allocated_gb']:.2f}GB allocated, {final_memory['pytorch_max_allocated_gb']:.2f}GB max")
            else:
                print(f"Final memory usage: {final_memory['allocated_gb']:.2f}GB allocated, {final_memory['max_allocated_gb']:.2f}GB max")
    
    except torch.OutOfMemoryError as e:
        # Training failed due to OOM - clear any existing cache
        print("*"*60)
        print("Training halted due to OOM error: clearing cache & collecting garbage.")
        print("*"*60)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Save training visualizations before re-raising the exception
        save_training_visualizations(trainer, prefix="oom_error")
        raise
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the configuration file and data files exist.")
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        
        # Save training visualizations before exiting
        if 'trainer' in locals():
            print("Saving training results as figures.")
            save_training_visualizations(trainer, prefix="error")

class SimpleGPT2VQVAETrainer(GPT2VQVAETrainer):
    """
    Trainer for SimpleGPT2VQVAE, inherits from GPT2VQVAETrainer but uses SimpleGPT2VQVAE as the model.
    """
    def __init__(self, 
                 model_config: Dict[str, Any], 
                 training_config: Dict[str, Any], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 run_name : Optional[str] = "ANONYM_RUN"):
        super().__init__(model_config, training_config, device, run_name)
        self.ensure_numeric_types(model_config)
        self.ensure_numeric_types(training_config)
        # Replace the model with SimpleGPT2VQVAE
        self.model = SimpleGPT2VQVAE(**model_config).to(device)
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
            print("Gradient checkpointing enabled for SimpleGPT2VQVAE")
        print("SimpleGPT2VQVAE trainer initialized successfully.")
    
    def _forward_pass(self, prompts, cots, prompt_masks, cot_masks):
        """Helper function for forward pass and loss calculation with use_vq parameter"""
        use_vq = self.training_config.get('use_vq', True)
        if self.use_mixed_precision:
            with autocast('cuda'):
                _, output_logits, vq_loss, perplexity, indices = self.model(
                    prompt=prompts,
                    cot_sequences=cots,
                    cot_mask=cot_masks,
                    prompt_mask=prompt_masks,
                    inference=False,
                    quantize_cot_only=self.training_config.get('quantize_cot_only', True),
                    use_vq=use_vq
                )
                recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
                total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        else:
            _, output_logits, vq_loss, perplexity, indices = self.model(
                prompt=prompts,
                cot_sequences=cots,
                cot_mask=cot_masks,
                prompt_mask=prompt_masks,
                inference=False,
                quantize_cot_only=self.training_config.get('quantize_cot_only', True),
                use_vq=use_vq
            )
            recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
            total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        return total_loss_batch, recon_loss, vq_loss, perplexity, indices



if __name__ == "__main__":
    # Run main function for command-line training
    main()