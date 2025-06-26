#!/usr/bin/env python3

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_queue import TrainingQueue, create_parameter_sweep_queue

def training_trial():
    
    # List of configuration files with its corresponding extra arguments
    config_files = [
        "configs/test_config.yaml",
        "configs/test_config2.yaml"
    ]
    
    # Base arguments to override for all runs
    base_args = {
        'data-dir': 'data/GSM8K',
        'max-samples': 10,  # Limit samples for testing
        'perplexity-threshold': 0  # Lower threshold for faster completion
    }
    
    # Create queue with overrides
    queue = TrainingQueue(
        config_files=config_files,
        base_args=base_args,
        output_dir="June25",
        resume_from_last=False
    )
    
    # Run the queue
    result = queue.run_queue()
    
    # Print summary
    queue.print_summary(result)
    
    return result

def adjust_min_lr_based_on_lr(specific_args : list, ratio_to_lr : float=0.01):
    updated_args = []
    for arg in specific_args:
        if "--lr" in arg:
            lr = float(arg[arg.index("--lr")+1])
            min_lr = str(lr * ratio_to_lr)
            arg.extend(["--min-lr", min_lr])
        updated_args.append(arg)
    
    return updated_args


def training(base_config : str, parameter_tuples, output_dir : str, start_from : int=0):
    config_files, specific_args = create_parameter_sweep_queue(
        parameter_tuples=parameter_tuples,
        base_config=base_config
        )
    
    # Base arguments to override for all runs
    base_args = {
        # 'perplexity-threshold': 5  # Lower threshold for faster completion
    }

    # Adjust min_lr based on the value of lr
    specific_args = adjust_min_lr_based_on_lr(specific_args, ratio_to_lr=0.01)

    # Create queue with overrides
    queue = TrainingQueue(
        config_files=config_files,
        base_args=base_args,
        output_dir=output_dir,
        run_specific_args=specific_args,
        resume_from_last=False
    )
    
    # Run the queue
    result = queue.run_queue(start_run_id=start_from)
    
    # Print summary
    queue.print_summary(result)
    
    return result


def sweep_max_batch_size(base_config, sweep_start : int, interval : int=8):
    config_files, specific_args = create_parameter_sweep_queue(
        parameter_tuples=[
            ("batch-size", [i for i in range(sweep_start, sweep_start*16, interval) if i > 0])
        ],
        base_config=base_config
        )
    
    # Base arguments to override for all runs
    base_args = {
        "perplexity-threshold" : float("inf"),
        "perplexity-window-size" : 15
    }

    # Create queue with overrides
    queue = TrainingQueue(
        config_files=config_files,
        base_args=base_args,
        output_dir=f"MemorySweep_{os.path.basename(base_config).replace('.yaml', '')}",
        run_specific_args=specific_args,
        resume_from_last=False,
        halt_on_oom=True
    )
    
    # Run the queue
    result = queue.run_queue()
    
    # Print summary
    queue.print_summary(result)
    
    return result

if __name__ == "__main__":
    # sweep_max_batch_size("configs/small_enc_GPT2_dec_one_thought.yaml", sweep_start=144, interval=8)
    
    # training(base_config="configs/two_thoughts_config.yaml", output_dir="")
    
    training(base_config="configs/small_enc_GPT2_dec_one_thought.yaml", 
             parameter_tuples=[
                ("lr", ["1e-6", "3e-7", "1e-7", "3e-8"]),
                ("vq-loss-weight", ["3e-2", "3e-1", "1.0", "3.0", "10", "100"])
             ],
             output_dir="checkpoints/small_enc_GPT2_dec/one_thought")
    
    # training(base_config="configs/four_thoughts_config.yaml", output_dir="")