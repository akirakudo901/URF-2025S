# Author: Akira Kudo
# Created: 2025/06/23
# Last Updated: 2025/06/23
# 
# Training Queue System
# - Executes multiple training configurations sequentially
# - Handles training abortion gracefully
# - Provides progress tracking and logging

import os
import sys
import argparse
import gc
import time
import itertools
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add the current directory to the path to import training module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from training import main as training_main, TrainingAbortedException

def generate_parameter_combinations(parameter_tuples: List[Tuple[str, List[Any]]]) -> List[List[str]]:
    """
    Generate all possible combinations of parameter values.
    
    Args:
        parameter_tuples: List of tuples where each tuple is (parameter_name, parameter_values)
                         - parameter_name: name of the parameter
                         - parameter_values: list of possible values for that parameter
    
    Returns:
        List of parameter combinations, where each combination is a list of strings
        in the format ["--param_name", "param_value", "--param_name2", "param_value2", ...]
    
    Example:
        >>> params = [
        ...     ("lr", [1e-4, 5e-5]),
        ...     ("vq-loss-weight", [0.5, 1.0])
        ... ]
        >>> combinations = generate_parameter_combinations(params)
        >>> for combo in combinations:
        ...     print(combo)
        ['--lr', '0.0001', '--vq-loss-weight', '0.5']
        ['--lr', '0.0001', '--vq-loss-weight', '1.0']
        ['--lr', '5e-05', '--vq-loss-weight', '0.5']
        ['--lr', '5e-05', '--vq-loss-weight', '1.0']
    """
    if not parameter_tuples:
        return []
    
    # Extract parameter names and values
    param_names = [param[0] for param in parameter_tuples]
    param_values_lists = [param[1] for param in parameter_tuples]
    
    # Generate all combinations using itertools.product
    value_combinations = list(itertools.product(*param_values_lists))
    
    # Convert each combination to the required format
    result = []
    for value_combo in value_combinations:
        param_combo = []
        for param_name, param_value in zip(param_names, value_combo):
            param_combo.extend([f"--{param_name}", str(param_value)])
        result.append(param_combo)
    
    return result

def create_parameter_sweep_queue(parameter_tuples: List[Tuple[str, List[Any]]], 
                                base_config: str,
                                output_dir: Optional[str] = None):
    """
    Create a training queue for parameter sweeping.
    
    Args:
        parameter_tuples: List of tuples (parameter_name, parameter_values)
        base_config: Base configuration file to use for all runs
        output_dir: Output directory for the parameter sweep
    
    Returns:
        List of argument lists for each parameter combination
    
    Example:
        >>> params = [
        ...     ("lr", [1e-4, 5e-5]),
        ...     ("vq-loss-weight", [0.5, 1.0])
        ... ]
        >>> queue_args = create_parameter_sweep_queue(params, "configs/default.yaml")
        >>> print(f"Generated {len(queue_args)} parameter combinations")
    """
    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations(parameter_tuples)
    
    # Create argument lists for each combination
    specifics_args = []
    for i, param_combo in enumerate(param_combinations):
        if output_dir:
            args = [
                "--checkpoint-dir", f"{output_dir}/run_{i:03d}"
            ]
        else:
            args = []
        args.extend(param_combo)
        specifics_args.append(args)
    
    configs = [base_config for _ in range(len(param_combinations))]
    return configs, specifics_args

class TrainingQueue:
    """
    Queue system for executing multiple training configurations sequentially.
    """
    
    def __init__(self, 
                 config_files: List[str],
                 base_args: Optional[Dict[str, Any]] = None,
                 run_specific_args: Optional[List[List[str]]] = None,
                 output_dir: str = "queue_outputs",
                 resume_from_last: bool = False,
                 halt_on_oom: bool = True):
        """
        Initialize the training queue.
        
        Args:
            config_files: List of configuration file paths
            base_args: Base arguments to apply to all training runs
            run_specific_args: List of argument lists for each run (optional)
            output_dir: Directory to store queue outputs
            resume_from_last: Whether to resume from the last successful checkpoint
            halt_on_oom: Whether to halt the entire queue when encountering OOM error
        """
        self.config_files = config_files
        self.base_args = base_args or {}
        self.run_specific_args = run_specific_args or []
        self.output_dir = Path(output_dir)
        self.resume_from_last = resume_from_last
        self.halt_on_oom = halt_on_oom
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Queue state
        self.completed_runs = []
        self.failed_runs = []
        self.current_run = 0
        self.start_time = None
        self.oom_encountered = False
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the queue system."""
        import logging
        
        # Create logger
        self.logger = logging.getLogger('TrainingQueue')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        log_file = self.output_dir / 'queue.log'
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def validate_config_files(self) -> List[str]:
        """
        Validate that all config files exist and are accessible.
        
        Returns:
            List of valid config file paths
        """
        valid_configs = []
        
        for config_file in self.config_files:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.error(f"Config file not found: {config_file}")
                continue
            
            if not config_path.is_file():
                self.logger.error(f"Config path is not a file: {config_file}")
                continue
            
            # Check file extension
            if config_path.suffix not in ['.yaml', '.yml', '.json']:
                self.logger.error(f"Unsupported config file format: {config_file}")
                continue
            
            valid_configs.append(str(config_path))
            self.logger.info(f"Validated config file: {config_file}")
        
        return valid_configs
    
    def get_checkpoint_for_resume(self, config_file: str) -> Optional[str]:
        """
        Find the latest checkpoint for resuming training.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Path to the latest checkpoint, or None if not found
        """
        try:
            # Load config to get checkpoint directory
            import yaml
            import json
            
            config_path = Path(config_file)
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            training_config = config.get('training_config', {})
            checkpoint_dir = Path(training_config.get('checkpoint_dir', 'checkpoints'))
            
            if not checkpoint_dir.exists():
                return None
            
            # Find the latest checkpoint
            checkpoint_files = list(checkpoint_dir.glob('*.pt'))
            if not checkpoint_files:
                return None
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Prefer best model checkpoints
            best_models = [f for f in checkpoint_files if 'best_model' in f.name]
            if best_models:
                return str(best_models[0])
            
            # Fall back to regular checkpoints
            return str(checkpoint_files[0])
            
        except Exception as e:
            self.logger.warning(f"Could not find checkpoint for {config_file}: {e}")
            return None
    
    def prepare_args_for_config(self, config_file: str, run_index: int) -> List[str]:
        """
        Prepare command line arguments for a specific config file.
        
        Args:
            config_file: Path to the configuration file
            run_index: Index of the current run
            
        Returns:
            List of command line arguments
        """
        args = ['--config', config_file]
        
        # Add base arguments
        for key, value in self.base_args.items():
            if key.startswith('--'):
                args.append(key)
                if value is not None and value is not True:
                    args.append(str(value))
            else:
                args.append(f'--{key}')
                if value is not None and value is not True:
                    args.append(str(value))
        
        # Add run-specific arguments if available
        if run_index < len(self.run_specific_args):
            run_args = self.run_specific_args[run_index]
            args.extend(run_args)
            self.logger.info(f"Added run-specific arguments for run {run_index + 1}: {' '.join(run_args)}")
        
        # Add run-specific arguments
        args.extend(['--checkpoint-dir', str(self.output_dir / f'run_{run_index}')])
        
        # Handle resume from last checkpoint
        if self.resume_from_last and run_index > 0:
            checkpoint = self.get_checkpoint_for_resume(config_file)
            if checkpoint:
                args.extend(['--resume-from', checkpoint])
                self.logger.info(f"Resuming from checkpoint: {checkpoint}")
        
        return args
    
    def execute_training_run(self, config_file: str, run_index: int) -> Dict[str, Any]:
        """
        Execute a single training run.
        
        Args:
            config_file: Path to the configuration file
            run_index: Index of the current run
            
        Returns:
            Dictionary with run results
        """
        run_start_time = time.time()
        run_result = {
            'config_file': config_file,
            'run_index': run_index,
            'start_time': run_start_time,
            'end_time': None,
            'duration': None,
            'status': 'unknown',
            'error': None,
            'aborted': False,
            'abort_reason': None
        }
        
        try:
            self.logger.info(f"Starting training run {run_index + 1}/{len(self.config_files)}")
            self.logger.info(f"Config file: {config_file}")
            
            # Prepare arguments
            args = self.prepare_args_for_config(config_file, run_index)
            self.logger.info(f"Arguments: {' '.join(args)}")
            
            # Execute training
            # Save original sys.argv and set it to our arguments
            original_argv = sys.argv
            sys.argv = ['training_queue.py'] + args
            
            try:
                training_main()
            except torch.OutOfMemoryError as e:
                # Handle OOM error based on configuration
                self.oom_encountered = True
                
                if self.halt_on_oom:
                    self.logger.error(f"Out of memory error on run {run_index + 1}, halting queue")
                    run_result['status'] = 'failed'
                    run_result['error'] = f"Out of memory error"
                    run_result['end_time'] = time.time()
                    run_result['duration'] = run_result['end_time'] - run_result['start_time']
                    return run_result
                else:
                    self.logger.warning(f"Out of memory error on run {run_index + 1}, continuing to next run")
                    run_result['status'] = 'failed'
                    run_result['error'] = f"Out of memory error"
                    run_result['end_time'] = time.time()
                    run_result['duration'] = run_result['end_time'] - run_result['start_time']
                    return run_result
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
                # Close any figure if they're open
                plt.close()
            
            # Training completed successfully
            run_result['status'] = 'completed'
            run_result['end_time'] = time.time()
            run_result['duration'] = run_result['end_time'] - run_result['start_time']
            
            self.logger.info(f"Training run {run_index + 1} completed successfully in {run_result['duration']:.2f} seconds")
            
        except TrainingAbortedException as e:
            # Training was aborted (this is expected behavior)
            run_result['status'] = 'aborted'
            run_result['aborted'] = True
            run_result['abort_reason'] = e.reason
            run_result['end_time'] = time.time()
            run_result['duration'] = run_result['end_time'] - run_result['start_time']
            
            self.logger.info(f"Training run {run_index + 1} aborted: {e.reason}")
            self.logger.info(f"Duration: {run_result['duration']:.2f} seconds")
        
        except Exception as e:
            # Training failed with an error (not OOM)
            run_result['status'] = 'failed'
            run_result['error'] = str(e)
            run_result['end_time'] = time.time()
            run_result['duration'] = run_result['end_time'] - run_result['start_time']
            
            self.logger.error(f"Training run {run_index + 1} failed: {e}")
            self.logger.error(f"Duration: {run_result['duration']:.2f} seconds")
        
        return run_result
    
    def run_queue(self, start_run_id: int = 0) -> Dict[str, Any]:
        """
        Execute all training configurations in the queue.
        
        Args:
            start_run_id: Run ID to start from (skips earlier configs). Defaults to 0.
        
        Returns:
            Dictionary with queue execution results
        """
        self.start_time = time.time()
        
        # Validate config files
        valid_configs = self.validate_config_files()
        if not valid_configs:
            self.logger.error("No valid configuration files found. Exiting.")
            return {
                'status': 'failed',
                'error': 'No valid configuration files',
                'completed_runs': [],
                'failed_runs': [],
                'total_duration': 0
            }
        
        # Skip configs before start_run_id
        if start_run_id > 0:
            valid_configs = valid_configs[start_run_id:]
            self.logger.info(f"Starting from run ID {start_run_id}")
            
        self.logger.info(f"Starting training queue with {len(valid_configs)} configurations")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Execute each configuration
        for i, config_file in enumerate(valid_configs):
            self.current_run = i + start_run_id
            run_result = self.execute_training_run(config_file, self.current_run)
            
            if run_result['status'] == 'completed':
                self.completed_runs.append(run_result)
            elif run_result['status'] == 'aborted':
                # Aborted runs are considered successful for queue purposes
                self.completed_runs.append(run_result)
            else:
                self.failed_runs.append(run_result)
                # Check if we should halt due to OOM
                if self.halt_on_oom and 'Out of memory error' in run_result.get('error', ''):
                    self.logger.error(f"Halting queue due to OOM error on run {self.current_run + 1}")
                    break
            
            # Add a small delay between runs
            if i < len(valid_configs) - 1:
                time.sleep(2)
        
        # Generate final report
        total_duration = time.time() - self.start_time
        queue_result = {
            'status': 'completed' if not self.failed_runs else 'partial_failure',
            'total_runs': len(valid_configs),
            'completed_runs': len(self.completed_runs),
            'failed_runs': len(self.failed_runs),
            'total_duration': total_duration,
            'completed_runs_details': self.completed_runs,
            'failed_runs_details': self.failed_runs
        }
        
        self.logger.info(f"Queue execution completed")
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        self.logger.info(f"Completed runs: {len(self.completed_runs)}")
        self.logger.info(f"Failed runs: {len(self.failed_runs)}")
        
        return queue_result
    
    def print_summary(self, queue_result: Dict[str, Any]):
        """
        Print a summary of the queue execution results.
        
        Args:
            queue_result: Results from queue execution
        """
        print("\n" + "="*80)
        print("TRAINING QUEUE SUMMARY")
        print("="*80)
        
        print(f"Total runs: {queue_result['total_runs']}")
        print(f"Completed runs: {queue_result['completed_runs']}")
        print(f"Failed runs: {queue_result['failed_runs']}")
        print(f"Total duration: {queue_result['total_duration']:.2f} seconds")
        print(f"Status: {queue_result['status']}")
        
        if queue_result['completed_runs_details']:
            print(f"\nCompleted Runs:")
            for run in queue_result['completed_runs_details']:
                status = "ABORTED" if run['aborted'] else "COMPLETED"
                print(f"  - {run['config_file']} ({status}) - {run['duration']:.2f}s")
                if run['aborted']:
                    print(f"    Abort reason: {run['abort_reason']}")
        
        if queue_result['failed_runs_details']:
            print(f"\nFailed Runs:")
            for run in queue_result['failed_runs_details']:
                print(f"  - {run['config_file']} - {run['duration']:.2f}s")
                print(f"    Error: {run['error']}")
        
        print("="*80)

def main():
    """
    Main function for command-line queue execution.
    """
    parser = argparse.ArgumentParser(description='Execute multiple training configurations in sequence')
    parser.add_argument('--configs', '-c', nargs='+', required=True,
                       help='List of configuration files to execute')
    parser.add_argument('--checkpoint-dir', '-o', type=str, default='queue_outputs',
                       help='Output directory for queue results')
    parser.add_argument('--resume-from-last', action='store_true',
                       help='Resume each run from the last checkpoint of the previous run')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory for all configs')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device for all configs')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Override max samples for all configs')
    parser.add_argument('--num-thoughts', type=int, default=None,
                       help='Override num_thoughts for all configs')
    parser.add_argument('--perplexity-threshold', type=float, default=None,
                       help='Override perplexity threshold for all configs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate for all configs')
    parser.add_argument('--min-lr', type=float, default=None,
                       help='Override minimum learning rate for all configs')
    parser.add_argument('--vq-loss-weight', type=float, default=None,
                       help='Override VQ loss weight for all configs')
    parser.add_argument('--run-args', nargs='+', action='append', default=[],
                       help='Run-specific arguments. Use --run-args for each run. Example: --run-args --lr 1e-4 --run-args --lr 5e-5')
    parser.add_argument('--halt-on-oom', action='store_true', default=True,
                       help='Halt the entire queue when encountering out of memory error (default: True)')
    parser.add_argument('--continue-on-oom', action='store_true',
                       help='Continue to next run when encountering out of memory error (overrides --halt-on-oom)')
    
    args = parser.parse_args()
    
    # Handle OOM behavior flags
    halt_on_oom = args.halt_on_oom
    if args.continue_on_oom:
        halt_on_oom = False
    
    # Prepare base arguments
    base_args = {}
    if args.data_dir:
        base_args['data_dir'] = args.data_dir
    if args.device:
        base_args['device'] = args.device
    if args.max_samples:
        base_args['max_samples'] = args.max_samples
    if args.num_thoughts:
        base_args['num_thoughts'] = args.num_thoughts
    if args.perplexity_threshold:
        base_args['perplexity_threshold'] = args.perplexity_threshold
    if args.lr:
        base_args['lr'] = args.lr
    if args.min_lr:
        base_args['min_lr'] = args.min_lr
    if args.vq_loss_weight:
        base_args['vq_loss_weight'] = args.vq_loss_weight
    
    # Process run-specific arguments
    run_specific_args = []
    for run_arg_list in args.run_args:
        run_specific_args.append(run_arg_list)
    
    # Validate that we have the right number of run-specific argument lists
    if run_specific_args and len(run_specific_args) != len(args.configs):
        print(f"Warning: Number of --run-args lists ({len(run_specific_args)}) doesn't match number of configs ({len(args.configs)})")
        print("Extra run-specific arguments will be ignored, missing ones will use base arguments only")
    
    # Create and run queue
    queue = TrainingQueue(
        config_files=args.configs,
        base_args=base_args,
        run_specific_args=run_specific_args,
        output_dir=args.checkpoint_dir,
        resume_from_last=args.resume_from_last,
        halt_on_oom=halt_on_oom
    )
    
    try:
        result = queue.run_queue()
        queue.print_summary(result)
        
        # Exit with appropriate code
        if result['status'] == 'completed':
            sys.exit(0)
        elif result['status'] == 'partial_failure':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\nQueue execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Queue execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
