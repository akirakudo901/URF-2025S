import torch
from typing import Optional, Dict, Any
from training import PhasedEnhancedGPT2VQVAETrainer

class AutoSwitchingTrainer(PhasedEnhancedGPT2VQVAETrainer):
    """
    Trainer that automatically switches from initialization to reestimation phase
    based on validation reconstruction loss trend or a step threshold.
    Inherits from PhasedEnhancedGPT2VQVAETrainer.
    """
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any], device: str = "cuda" if torch.cuda.is_available() else "cpu", run_name: Optional[str] = "ANONYM_RUN"):
        # Extract auto-switching parameters from training_config
        self.patience = training_config.pop('auto_switch_patience', 3)
        self.validation_checks_per_epoch = training_config.pop('auto_switch_validation_checks_per_epoch', 5)
        
        # Initialize parent with cleaned configs
        super().__init__(model_config, training_config, device, run_name=run_name)
        
        # Initialize auto-switching state
        self._val_recon_loss_history = []
        self._val_loss_increase_count = 0
        self._force_reestimation_phase = False

    def set_val_loader(self, val_loader):
        """Set the validation loader to be used for auto-switching validation checks."""
        self._val_loader = val_loader

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
        Override train method to set up auto-switching validation before calling parent.
        """
        # Create train and test datasets
        train_dataset = torch.utils.data.TensorDataset(train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask)
        test_dataset = torch.utils.data.TensorDataset(test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
        
        # Create data loaders
        train_loader = self.create_data_loader(train_dataset, batch_size=self.training_config['batch_size'], shuffle=True)
        val_loader = self.create_data_loader(test_dataset, batch_size=self.training_config['batch_size'], shuffle=False)
        
        # Set up auto-switching validation
        total_batches = len(train_loader)
        steps_per_epoch = max(1, total_batches // self.gradient_accumulation_steps)
        self._val_check_interval_steps = max(1, steps_per_epoch // self.validation_checks_per_epoch)
        self.set_val_loader(val_loader)
        
        print(f"[AutoSwitch] Validation checks every {self._val_check_interval_steps} steps ({self.validation_checks_per_epoch} times per epoch)")
        
        # Call parent train method
        return super().train(
            train_prompt_sequences=train_prompt_sequences,
            train_cot_sequences=train_cot_sequences,
            train_prompt_mask=train_prompt_mask,
            train_cot_mask=train_cot_mask,
            test_prompt_sequences=test_prompt_sequences,
            test_cot_sequences=test_cot_sequences,
            test_prompt_mask=test_prompt_mask,
            test_cot_mask=test_cot_mask,
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
            
            # Check for consecutive increases
            if len(self._val_recon_loss_history) > 0:
                if val_recon_loss > self._val_recon_loss_history[-1]:
                    self._val_loss_increase_count += 1
                else:
                    self._val_loss_increase_count = 0
            self._val_recon_loss_history.append(val_recon_loss)
            
            print(f"[AutoSwitch] Step {self.current_step}, Val Recon Loss: {val_recon_loss:.4f}, Increases: {self._val_loss_increase_count}/{self.patience}")
            if self._val_loss_increase_count >= self.patience:
                print(f"[AutoSwitch] Switching to reestimation phase at step {self.current_step} (patience={self.patience})")
                self._force_reestimation_phase = True

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

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None, **kwargs):
        """
        Override save_checkpoint to include auto-switching state.
        """
        # Add auto-switching state to kwargs
        auto_switch_state = {
            'patience': self.patience,
            'validation_checks_per_epoch': self.validation_checks_per_epoch,
            'val_recon_loss_history': self._val_recon_loss_history,
            'val_loss_increase_count': self._val_loss_increase_count,
            'force_reestimation_phase': self._force_reestimation_phase,
            'val_check_interval_steps': getattr(self, '_val_check_interval_steps', None)
        }
        kwargs.update(auto_switch_state)
        
        # Call parent save_checkpoint
        return super().save_checkpoint(epoch, metrics, is_best, checkpoint_path, **kwargs)

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
        
        return checkpoint
