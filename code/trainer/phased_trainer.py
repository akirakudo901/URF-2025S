# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import torch
import torch.optim as optim
from typing import Optional, Dict, Any
import os
from torch.amp.autocast_mode import autocast
import sys

# Add the current directory to the path to import dependencies
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules and functions
from phone_notification import send_notification
from train_utils import compute_reconstruction_loss

# Import the parent trainer class
from enhanced_trainer import EnhancedGPT2VQVAETrainer

# GPU memory monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: nvidia-ml-py3 not available. Install with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False

class PhasedEnhancedGPT2VQVAETrainer(EnhancedGPT2VQVAETrainer):
    """
    Specialized trainer for EnhancedGPT2VQVAE that implements a phased training approach:
    
    1. Initialization Phase (0 to initialization_steps):
       - Train using no-vq mode to learn good representations
       
    2. Reinitialization Phase (initialization_steps to quantization_start):
       - Every r_reestim steps, fully reinitialize codebook using KMeans on reservoir samples
       - Continue training with VQ enabled
       
    3. Normal Training Phase (after quantization_start):
       - Standard VQ-VAE training with all enhancements
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any], 
                 training_config: Dict[str, Any], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 run_name : Optional[str] = "ANONYM_RUN"):
        """
        Initialize the phased trainer.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary with phased training parameters:
                - initialization_steps: Number of steps to train in no-vq mode
                - r_reestim: Frequency of codebook reinitialization during reinitialization phase
                - quantization_start: Step at which to start normal VQ training
                - OR: reestimation_phase_length (alternative), taking precedence if both provided 
                      (e.g. with dynamic change of initialization phase)
            device: Device to train on
            run_name: Optional string for the name of the run, as texted via send_notification.
        """
        # --- Phase boundary logic ---
        # Require initialization_steps, then either reestimation_phase_length or quantization_start
        if 'initialization_steps' not in training_config:
            raise ValueError("'initialization_steps' must be specified in the training config.")
        self.initialization_steps = training_config['initialization_steps']

        has_reestimation_length = 'reestimation_phase_length' in training_config
        has_quantization_start = 'quantization_start' in training_config

        if has_reestimation_length and has_quantization_start:
            print("[PhasedTrainer] WARNING: Both reestimation_phase_length and quantization_start provided. Using reestimation_phase_length.")

        if has_quantization_start:
            self.quantization_start = training_config['quantization_start']
            self.reestimation_phase_length = self.quantization_start - self.initialization_steps
        elif has_reestimation_length:
            self.reestimation_phase_length = training_config['reestimation_phase_length']
            self.quantization_start = self.initialization_steps + self.reestimation_phase_length
        else:
            raise ValueError("Must specify either 'reestimation_phase_length' or 'quantization_start' in the training config.")

        self.r_reestim = training_config.get('r_reestim', 500)
        self.codebook_lr_multiplier = training_config.get('codebook_lr_multiplier', 1.0)
        
        # Validate parameters
        if self.initialization_steps >= self.quantization_start:
            raise ValueError("initialization_steps must be less than quantization_start")
        if self.r_reestim <= 0:
            raise ValueError("r_reestim must be positive")
        if self.codebook_lr_multiplier <= 0:
            raise ValueError("codebook_lr_multiplier must be positive")
        
        # Replace the track codebook usage function with the phased version (which ignores the initialization stage)
        phased_tracking_functions = {
            'track_codebook_usage': self._phased_track_codebook_usage
        }

        # Initialize parent trainer
        super().__init__(model_config, training_config, device, phased_tracking_functions, run_name)
        
        # Override optimizer with codebook-specific learning rates
        self._setup_codebook_optimizer()
        
        # Print phase info
        print(f"PhasedEnhancedGPT2VQVAETrainer initialized with:")
        print(f"  - Initialization phase: 0 to {self.initialization_steps} steps (no-vq mode)")
        print(f"  - Reinitialization phase: {self.initialization_steps} to {self.quantization_start} steps (length: {self.reestimation_phase_length})")
        print(f"  - Normal training phase: after {self.quantization_start} steps")
        print(f"  - Reinitialization frequency: every {self.r_reestim} steps")
        print(f"  - Codebook learning rate multiplier: {self.codebook_lr_multiplier}x")

        self._init_phase_best_tracking()
    
    def _setup_codebook_optimizer(self):
        """
        Set up optimizer with different learning rates for codebook parameters.
        Codebook parameters get learning_rate * codebook_lr_multiplier,
        while all other parameters get the normal learning_rate.
        """
        # if the multipler is 1.0, no change
        if self.codebook_lr_multiplier == 1.0:
            return
        
        # Get the base learning rate
        base_lr = self.training_config['learning_rate']
        codebook_lr = base_lr * self.codebook_lr_multiplier
        
        # Separate parameters into codebook and non-codebook groups
        codebook_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            # TODO: CONSIDER IF aggregation_mlp SHOULD GET HIGHER LR OR NOT
            # TBH, HARD TO SAY. WILL START WITHOUT.
            # if 'vector_quantizer' in name or 'aggregation_mlp' in name:
            if 'vector_quantizer' in name:
                codebook_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': self.training_config.get('weight_decay', 0.01),
                'betas': (self.training_config.get('beta1', 0.9), self.training_config.get('beta2', 0.999))
            },
            {
                'params': codebook_params,
                'lr': codebook_lr,
                'weight_decay': self.training_config.get('weight_decay', 0.01),
                'betas': (self.training_config.get('beta1', 0.9), self.training_config.get('beta2', 0.999))
            }
        ]
        
        # CRITICAL FIX: Clear old optimizer state before recreating
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            # Clear the old optimizer's state to prevent memory accumulation
            self.optimizer.state.clear()
            # Explicitly delete the old optimizer to free memory
            del self.optimizer
        
        # Recreate optimizer with parameter groups
        self.optimizer = optim.AdamW(param_groups)
        
        # CRITICAL FIX: Clear any existing scheduler state before recreating
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            del self.scheduler
        
        # Recreate scheduler if it exists
        if self.training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['num_epochs'],
                eta_min=self.training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        print(f"Optimizer reconfigured with separate learning rates:")
        print(f"  - Main parameters: {base_lr:.2e}")
        print(f"  - Codebook parameters: {codebook_lr:.2e} ({self.codebook_lr_multiplier}x multiplier)")
        print(f"  - Codebook parameters count: {len(codebook_params)}")
        print(f"  - Other parameters count: {len(other_params)}")
    
    def _determine_training_phase(self, current_step: int) -> str:
        """
        Determine the current training phase based on the step number.
        
        Args:
            current_step: Current training step
            
        Returns:
            str: Current phase name
        """
        if current_step < self.initialization_steps:
            return "initialization"
        elif current_step < self.quantization_start:
            return "reinitialization"
        else:
            return "normal"
    
    def _should_reinitialize_codebook(self, current_step: int) -> bool:
        """
        Check if codebook should be reinitialized at the current step.
        
        Args:
            current_step: Current training step
            
        Returns:
            bool: True if codebook should be reinitialized
        """
        if current_step < self.initialization_steps:
            return False
        
        if current_step >= self.quantization_start:
            return False
        
        # Check if we're in reinitialization phase using modulo
        if current_step >= self.initialization_steps:
            adjusted_step = current_step - self.initialization_steps
            return (adjusted_step + 1) % self.r_reestim == 0
        
        return False
    
    def _reinitialize_codebook(self, current_step: int):
        """
        Reinitialize the codebook using the EnhancedVectorQuantizer's _reset_codebook method.
        
        Args:
            current_step: Current training step for logging
        """
        print(f"\n=== Codebook Reinitialization at step {current_step} ===")
        
        # Get a dummy input tensor for the device reference (required by _reset_codebook)
        device = self.model.vector_quantizer.embedding.weight.device
        
        # Use the EnhancedVectorQuantizer's _reset_codebook method with 'full' strategy
        # This will perform K-means++ clustering on reservoir samples and reset the entire codebook
        self.model.vector_quantizer._reset_codebook(device, reset_strategy='full')
    
    def _forward_pass(self, prompts, cots, prompt_masks, cot_masks, backpointers=None, no_vq=False):
        """
        Enhanced forward pass that handles different training phases.
        
        Args:
            prompts: Prompt sequences
            cots: Chain-of-thought sequences
            prompt_masks: Prompt attention masks
            cot_masks: COT attention masks
            
        Returns:
            tuple: (total_loss_batch, vq_loss, perplexity, indices)
        """
        # Determine current phase
        current_phase = self._determine_training_phase(self.current_step)
        
        # Check if we need to reinitialize codebook
        if self._should_reinitialize_codebook(self.current_step):
            self._reinitialize_codebook(self.current_step)
        
        # Determine whether to use VQ based on current phase
        no_vq = current_phase == "initialization" or no_vq
        
        # Update phase tracking for logging
        last_step_phase = self._determine_training_phase(self.current_step-1)
        if current_phase != last_step_phase:
            print(f"\n=== Phase transition: {last_step_phase} -> {current_phase} at step {self.current_step-1} ===")
        
        # Perform forward pass with appropriate VQ setting and handle mixed precision
        return super()._forward_pass(prompts, cots, prompt_masks, cot_masks, backpointers, no_vq=no_vq)
    
    def train_epoch(self, train_loader, num_measurements_per_epoch, current_epoch=0, detailed_metrics_callback=None):
        """
        Enhanced train_epoch that tracks training steps and handles phase transitions.
        
        Args:
            train_loader: Training data loader
            num_measurements_per_epoch: Number of measurements per epoch
            current_epoch: Current epoch number
            
        Returns:
            dict: Training metrics for the epoch
        """
        # Set max_reset_steps on the first epoch if it's still None
        if current_epoch == 0 and self.model.vector_quantizer.max_reset_steps is None:
            # Set max_reset_steps to quantization_start to disable automatic resets during normal training
            self.model.vector_quantizer.max_reset_steps = self.quantization_start
        
        # Call parent train_epoch
        metrics = super().train_epoch(train_loader, num_measurements_per_epoch, current_epoch, detailed_metrics_callback)
        
        return metrics

    def _phased_track_codebook_usage(self, dataset: Any, measurement_point: int) -> None:
        """
        Enhanced codebook tracking with additional statistics.
        
        Args:
            dataset: Dataset to sample from
            measurement_point: Current measurement point index
        """
        if self._determine_training_phase(self.current_step) == "initialization": return
        # Only call the parent's _enhanced_track_codebook_usage if we are out of the warm up phase
        super()._enhanced_track_codebook_usage(dataset, measurement_point)
        
    
    def _init_phase_best_tracking(self):
        """
        Initialize the best validation loss tracking per phase and loss type.
        """
        self.best_val_loss_per_phase = {
            'initialization': {'total': float('inf'), 'recon': float('inf')},
            'reinitialization': {'total': float('inf'), 'recon': float('inf')},
            'normal': {'total': float('inf'), 'recon': float('inf')},
        }

    def is_new_best(self, val_loss: float, loss_type: str = 'total') -> bool:
        """
        Return True if val_loss is better than the current best loss for the specified type.
        
        Args:
            val_loss: The validation loss to check
            loss_type: Type of loss to check ('total' or 'recon')
            
        Returns:
            bool: True if this is a new best for the specified loss type
        """
        if loss_type not in ['total', 'recon']:
            raise ValueError(f"Unknown loss type: {loss_type}. Must be 'total' or 'recon'.")
        
        phase = self._determine_training_phase(self.current_step)
        return val_loss < self.best_val_loss_per_phase[phase][loss_type]

    def update_best(self, val_loss: float, loss_type: str = 'total'):
        """
        Update the best loss if val_loss is better for the specified type.
        
        Args:
            val_loss: The validation loss to update
            loss_type: Type of loss to update ('total' or 'recon')
        """
        if loss_type not in ['total', 'recon']:
            raise ValueError(f"Unknown loss type: {loss_type}. Must be 'total' or 'recon'.")
        
        phase = self._determine_training_phase(self.current_step)
        if self.is_new_best(val_loss, loss_type):
            self.best_val_loss_per_phase[phase][loss_type] = val_loss

    def get_best_val_loss(self, phase: str = None, loss_type: str = 'total'):
        """
        Get the best validation loss for a specific phase and loss type.
        
        Args:
            phase: Training phase (if None, uses current phase)
            loss_type: Type of loss ('total' or 'recon')
            
        Returns:
            float: Best validation loss for the specified phase and loss type
        """
        if loss_type not in ['total', 'recon']:
            raise ValueError(f"Unknown loss type: {loss_type}. Must be 'total' or 'recon'.")
        
        if phase is None:
            phase = self._determine_training_phase(self.current_step)
        return self.best_val_loss_per_phase[phase][loss_type]

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None, remove_other_best_models: bool = True, loss_type: str = 'total', **kwargs):
        """
        Save checkpoint with phase-aware and loss-type-aware best model logic.
        If is_best and remove_other_best_models is True, only remove best model checkpoints 
        from the same phase and loss type.
        """
        kwargs.update({'best_val_loss_per_phase': self.best_val_loss_per_phase})
        phase = self._determine_training_phase(self.current_step)
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        if is_best:
            if remove_other_best_models:
                # Remove only best model checkpoints from the same phase and loss type if folder exists
                if os.path.exists(checkpoint_dir):
                    for file in os.listdir(checkpoint_dir):
                        if file.startswith(f'best_model_{phase}_{loss_type}_') and file.endswith('.pt'):
                            os.remove(os.path.join(checkpoint_dir, file))
            if checkpoint_path is None:
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{phase}_{loss_type}_epoch_{epoch}.pt')
        super().save_checkpoint(epoch, metrics, is_best, checkpoint_path, remove_other_best_models=False, loss_type=loss_type, **kwargs)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = super().load_checkpoint(checkpoint_path)
        # Restore per-phase bests
        if 'best_val_loss_per_phase' in checkpoint:
            loaded_best_val_loss_per_phase = checkpoint['best_val_loss_per_phase']
            
            # Check if the loaded data has the new structure (dict with 'total' and 'recon')
            if isinstance(loaded_best_val_loss_per_phase, dict):
                # Check if it's the new structure (nested dict) or old structure (simple dict)
                first_phase = list(loaded_best_val_loss_per_phase.keys())[0]
                if isinstance(loaded_best_val_loss_per_phase[first_phase], dict):
                    # New structure: {'phase': {'total': val, 'recon': val}}
                    self.best_val_loss_per_phase = loaded_best_val_loss_per_phase
                else:
                    # Old structure: {'phase': val} - convert to new structure
                    self._init_phase_best_tracking()
                    for phase, val in loaded_best_val_loss_per_phase.items():
                        if phase in self.best_val_loss_per_phase:
                            self.best_val_loss_per_phase[phase]['total'] = val
                            self.best_val_loss_per_phase[phase]['recon'] = val
            else:
                # Fallback: initialize from best_val_loss if present
                best = checkpoint.get('best_val_loss', float('inf'))
                self._init_phase_best_tracking()
                for phase in self.best_val_loss_per_phase:
                    self.best_val_loss_per_phase[phase]['total'] = best
                    self.best_val_loss_per_phase[phase]['recon'] = best
        else:
            # Backward compatibility: initialize from best_val_loss if present
            best = checkpoint.get('best_val_loss', float('inf'))
            self._init_phase_best_tracking()
            for phase in self.best_val_loss_per_phase:
                self.best_val_loss_per_phase[phase]['total'] = best
                self.best_val_loss_per_phase[phase]['recon'] = best
        
        print(f"Current best val loss per phase: {self.best_val_loss_per_phase}")
        return checkpoint

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
            
            metric_dict = self._get_training_completion_metric_dict(final_metrics, training_duration)

            # Show per-phase best val loss if available, otherwise fallback to overall best
            if hasattr(self, "best_val_loss_per_phase") and isinstance(self.best_val_loss_per_phase, dict):
                del metric_dict["Best Val Loss"]
                
                formatted_phase_val_loss = ""
                
                for phase in ["initialization", "reinitialization", "normal"]:
                    phase_data = self.best_val_loss_per_phase.get(phase, {})
                    if isinstance(phase_data, dict):
                        # New structure with separate total and recon losses
                        total_val = phase_data.get('total', float('inf'))
                        recon_val = phase_data.get('recon', float('inf'))
                        if total_val == float('inf') and recon_val == float('inf'):
                            formatted_phase_val_loss += f"  {phase}: N/A\n"
                        else:
                            formatted_phase_val_loss += f"  {phase}: total={total_val:.4f}, recon={recon_val:.4f}\n"
                    else:
                        # Old structure (backward compatibility)
                        val = phase_data if isinstance(phase_data, (int, float)) else float('inf')
                        if val == float('inf'):
                            formatted_phase_val_loss += f"  {phase}: N/A\n"
                        else:
                            formatted_phase_val_loss += f"  {phase}: {val:.4f}\n"
                
                metric_dict["Best Val Loss (per phase)"] = formatted_phase_val_loss

            message += self._get_training_completion_message_from_dict(metric_dict)
            
            # Send the phone notification
            return send_notification(message)
            
        except Exception as e:
            print(f"Error sending training completion phone notification: {e}")
            return False