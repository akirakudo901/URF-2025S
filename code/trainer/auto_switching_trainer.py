import os
import sys
from typing import Optional, Dict, Any

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trainer.phased_trainer import PhasedEnhancedGPT2VQVAETrainer

class AutoSwitchingTrainer(PhasedEnhancedGPT2VQVAETrainer):
    """
    Trainer that automatically switches from initialization to reestimation phase
    based on validation reconstruction loss trend or a step threshold.
    
    Two switching mechanisms are supported:
    1. Patience-based: Switch after consecutive validation loss increases (default)
    2. Threshold-based: Switch when validation loss drops below a threshold
    
    Inherits from PhasedEnhancedGPT2VQVAETrainer.
    """
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any], device: str = "cuda" if torch.cuda.is_available() else "cpu", run_name: Optional[str] = "ANONYM_RUN"):
        # Extract auto-switching parameters from training_config
        self.patience = training_config.pop('auto_switch_patience', 3)
        self.validation_checks_per_epoch = training_config.pop('auto_switch_validation_checks_per_epoch', 5)
        self.threshold = training_config.pop('auto_switch_threshold', None)  # New threshold parameter
        
        # Note: If threshold is set, it takes precedence over patience mechanism
        # If threshold is None, patience-based switching is used
        
        # Initialize parent with cleaned configs
        super().__init__(model_config, training_config, device, run_name=run_name)
        
        # Initialize auto-switching state
        self._val_recon_loss_history = []
        self._val_loss_increase_count = 0
        self._force_reestimation_phase = False
        
        # Print auto-switching configuration
        if self.threshold is not None:
            print(f"[AutoSwitch] Threshold-based switching enabled: {self.threshold:.4f}")
        else:
            print(f"[AutoSwitch] Patience-based switching enabled: {self.patience}")

    def train(self, 
              train_prompt_sequences: torch.Tensor,
              train_cot_sequences: torch.Tensor,
              train_prompt_mask: torch.Tensor,
              train_cot_mask: torch.Tensor,
              test_prompt_sequences: torch.Tensor,
              test_cot_sequences: torch.Tensor,
              test_prompt_mask: torch.Tensor,
              test_cot_mask: torch.Tensor,
              train_backpointers: Optional[torch.Tensor] = None,
              test_backpointers: Optional[torch.Tensor] = None,
              resume_from: Optional[str] = None,
              num_measurements_per_epoch: Optional[int] = None,
              seed: int = 42):
        """
        Override train method to set up auto-switching validation before calling the helper function.
        """
        # Create train and test datasets
        train_elems = [train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask]
        test_elems = [test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask]
        if self.model.compress_beam_search:
            if (train_backpointers is not None) and (test_backpointers is not None):
                train_elems += [train_backpointers]
                test_elems += [test_backpointers]
            else:
                raise Exception("If model is set to compress_beam_search mode, both train_backpointers and test_backpointers must be passed.")
        train_dataset = torch.utils.data.TensorDataset(*train_elems)
        test_dataset = torch.utils.data.TensorDataset(*test_elems)
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, batch_size=self.training_config['batch_size'], shuffle=True)
        val_loader = self.create_data_loader(test_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
        
        # Set up auto-switching validation
        total_batches = len(train_loader)
        steps_per_epoch = max(1, total_batches // self.gradient_accumulation_steps)
        self._val_check_interval_steps = max(1, steps_per_epoch // self.validation_checks_per_epoch)
        
        self._val_loader = val_loader
        
        print(f"[AutoSwitch] Validation checks every {self._val_check_interval_steps} steps ({self.validation_checks_per_epoch} times per epoch)")
        
        # Call the helper function with the created loaders
        return self._train_with_loaders(
            train_loader=train_loader,
            test_loader=val_loader,
            resume_from=resume_from,
            num_measurements_per_epoch=num_measurements_per_epoch,
            seed=seed
        )

    def _update_weights(self):
        super()._update_weights()
        # Only do auto-switching validation in initialization phase
        if (self._determine_training_phase(self.current_step) == "initialization" and
            hasattr(self, '_val_check_interval_steps') and self._val_check_interval_steps > 0 and
            hasattr(self, '_val_loader') and self._val_loader is not None and
            self.current_step % self._val_check_interval_steps == 0):
            
            val_metrics = self.validate(self._val_loader)
            val_recon_loss = val_metrics.get('recon_loss', None)
            
            # Check for consecutive increases (patience-based mechanism)
            if len(self._val_recon_loss_history) > 0:
                if val_recon_loss > self._val_recon_loss_history[-1]:
                    self._val_loss_increase_count += 1
                else:
                    self._val_loss_increase_count = 0
            self._val_recon_loss_history.append(val_recon_loss)
            
            # Determine switching reason
            switch_reason = None
            if self.threshold is not None and val_recon_loss < self.threshold:
                switch_reason = f"threshold ({val_recon_loss:.4f} < {self.threshold:.4f})"
            elif self.threshold is None and self._val_loss_increase_count >= self.patience:
                switch_reason = f"patience ({self._val_loss_increase_count}/{self.patience})"
            
            # Print current status
            if self.threshold is not None:
                print(f"\n[AutoSwitch] Step {self.current_step}, Val Recon Loss: {val_recon_loss:.4f}, Threshold: {self.threshold:.4f}")
            else:
                print(f"\n[AutoSwitch] Step {self.current_step}, Val Recon Loss: {val_recon_loss:.4f}, Increases: {self._val_loss_increase_count}/{self.patience}")
            
            # Switch to reestimation phase if criteria are met
            if switch_reason is not None:
                print(f"\n[AutoSwitch] Switching to reestimation phase at step {self.current_step} ({switch_reason})")
                self._force_reestimation_phase = True
                # If using phase lengths, dynamically set phase boundaries
                if self.reestimation_phase_length is not None:
                    self.initialization_steps = self.current_step
                    self.quantization_start = self.current_step + self.reestimation_phase_length
                    print(f"[AutoSwitch] Dynamic phase boundaries set: initialization_steps={self.initialization_steps}, quantization_start={self.quantization_start}")

    def _determine_training_phase(self, current_step: int) -> str:
        """
        Override parent's phase determination to check for forced reestimation phase.
        If the auto-switching criteria are met, force reestimation phase even if we're
        still within initialization_steps.
        """
        if self._force_reestimation_phase and current_step < self.quantization_start:
            return "reinitialization"
        else:
            # Call parent's method for normal phase determination
            return super()._determine_training_phase(current_step)
    
    def _reset_auto_switch_state(self):
        """
        Reset the auto-switching state. Useful when loading from checkpoints
        or when you want to reset the forced phase flag.
        """
        self._force_reestimation_phase = False
        self._val_recon_loss_history = []
        self._val_loss_increase_count = 0
        # Note: threshold is not reset as it's a configuration parameter

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None, remove_other_best_models: bool = True, loss_type: str = 'total', **kwargs):
        """
        Override save_checkpoint to include auto-switching state.
        """
        # Add auto-switching state to kwargs
        auto_switch_state = {
            'patience': self.patience,
            'validation_checks_per_epoch': self.validation_checks_per_epoch,
            'threshold': self.threshold,  # Save threshold parameter
            'val_recon_loss_history': self._val_recon_loss_history,
            'val_loss_increase_count': self._val_loss_increase_count,
            'force_reestimation_phase': self._force_reestimation_phase,
            'val_check_interval_steps': getattr(self, '_val_check_interval_steps', None),
            # Save dynamic phase boundaries if changed
            'initialization_steps': self.initialization_steps,
            'quantization_start': self.quantization_start,
            'reestimation_phase_length': getattr(self, 'reestimation_phase_length', None),
        }
        kwargs.update(auto_switch_state)
        
        # Call parent save_checkpoint
        return super().save_checkpoint(epoch, metrics, is_best, checkpoint_path, remove_other_best_models, loss_type, **kwargs)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Override load_checkpoint to restore auto-switching state.
        """
        # Call parent load_checkpoint
        checkpoint = super().load_checkpoint(checkpoint_path)
        
        # Restore auto-switching state if available
        if 'patience' in checkpoint:
            self.patience = checkpoint['patience']
            print(f"Restored patience: {self.patience}")
        
        if 'threshold' in checkpoint:
            self.threshold = checkpoint['threshold']
            print(f"Restored threshold: {self.threshold}")
        
        if 'validation_checks_per_epoch' in checkpoint:
            self.validation_checks_per_epoch = checkpoint['validation_checks_per_epoch']
            print(f"Restored validation_checks_per_epoch: {self.validation_checks_per_epoch}")
        
        if 'val_recon_loss_history' in checkpoint:
            self._val_recon_loss_history = checkpoint['val_recon_loss_history']
            print(f"Restored validation loss history with {len(self._val_recon_loss_history)} entries")
        
        if 'val_loss_increase_count' in checkpoint:
            self._val_loss_increase_count = checkpoint['val_loss_increase_count']
            print(f"Restored validation loss increase count: {self._val_loss_increase_count}")
        
        if 'force_reestimation_phase' in checkpoint:
            self._force_reestimation_phase = checkpoint['force_reestimation_phase']
            print(f"Restored force_reestimation_phase: {self._force_reestimation_phase}")
        
        if 'val_check_interval_steps' in checkpoint:
            self._val_check_interval_steps = checkpoint['val_check_interval_steps']
            print(f"Restored val_check_interval_steps: {self._val_check_interval_steps}")
        
        # Restore dynamic phase boundaries if present
        if 'initialization_steps' in checkpoint:
            self.initialization_steps = checkpoint['initialization_steps']
            print(f"Restored initialization_steps: {self.initialization_steps}")
        if 'quantization_start' in checkpoint:
            self.quantization_start = checkpoint['quantization_start']
            print(f"Restored quantization_start: {self.quantization_start}")
        if 'reestimation_phase_length' in checkpoint:
            self.reestimation_phase_length = checkpoint['reestimation_phase_length']
            print(f"Restored reestimation_phase_length: {self.reestimation_phase_length}")
        
        return checkpoint
