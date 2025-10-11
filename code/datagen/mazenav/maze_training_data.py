#!/usr/bin/env python3
"""
Functions to convert mazes into training data format for VQ-VAE training.
This module handles the conversion from maze data to tokenized sequences that can be used
as prompts in the same manner as GSM8K problems.

Based on the specification in todo.txt:
- Maze format: "bos start x y goal x y wall x1 y1 wall x2 y2 ... wall xn yn eos"
- Path format: "bos plan x1 y1 plan x2 y2 ... plan xm ym eos"
"""

import json
import os
import random
import torch
import logging
from typing import List, Tuple, Optional, Dict, Any
from transformers import GPT2TokenizerFast

# Import maze generation functions
from generate_maze import (
    generate_maze_recursive_backtracking, 
    generate_maze_random,
    Maze
)

# Import beam search functions
from astar_beam_search import AStarBeamSearch

# Import logging utilities
from logging_utils import (
    ProgressTracker, 
    log_execution_time, 
    SearchResult, 
    MazeSearchStats,
    log_search_result,
    log_final_summary,
    setup_logging
)

def load_and_tokenize_mazes(maze_strings_file: str,
                           tokenizer_name: str = "gpt2",
                           max_length: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load maze strings from file and tokenize them.
    
    Args:
        maze_strings_file: Path to JSON file containing maze strings
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (tokenized_sequences, attention_masks)
        - tokenized_sequences: torch.Tensor of shape (num_mazes, max_length)
        - attention_masks: torch.Tensor of shape (num_mazes, max_length)
    """
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load maze data
    # TODO FIX THIS ONCE IMPLEMENTATION OF MAZE SAVING IS DONE
    maze_data = load_maze_to_be_implemented(maze_strings_file)
    
    # Extract maze strings
    maze_strings = []
    for maze_dict in maze_data:
        maze_string = maze_dict['maze_string']
        # Add bos and eos tokens
        maze_string = f"bos {maze_string} eos"
        maze_strings.append(maze_string)
    # TODO END FIX UP UNTIL HERE
    
    # Tokenize
    tokenized = tokenizer(
        maze_strings,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized['input_ids'], tokenized['attention_mask']


def save_tokenized_maze_data(maze_strings_file: str,
                            output_dir: str,
                            tokenizer_name: str = "gpt2",
                            pad_token_id: int = 50256,
                            max_length: int = 1024,
                            test_split_ratio: float = 0.1,
                            random_seed: int = 42) -> None:
    """
    Load maze strings, tokenize them, and save as torch files compatible with training.py.
    
    This creates two files, prompt_sequences.pt and prompt_mask.pt (padding masks).
    
    Args:
        maze_strings_file: Path to JSON file containing maze strings
        output_dir: Directory to save the torch files
        tokenizer_name: Name of the tokenizer to use
        pad_token_id: Token ID to use for padding
        max_length: Maximum sequence length
        test_split_ratio: Ratio of data to use for testing
        random_seed: Random seed for train/test split
    """
    random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Tokenize mazes
    print(f"Loading and tokenizing mazes from {maze_strings_file}...")
    tokenized_sequences, attention_masks = load_and_tokenize_mazes(
        maze_strings_file, tokenizer_name, max_length
    )
    
    # Split into train/test
    num_mazes = tokenized_sequences.shape[0]
    indices = list(range(num_mazes))
    random.shuffle(indices)
    
    split_idx = int(num_mazes * (1 - test_split_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Create train/test directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save train data
    train_sequences = tokenized_sequences[train_indices]
    train_masks = attention_masks[train_indices]
    
    torch.save(train_sequences, os.path.join(train_dir, "prompt_sequences.pt"))
    torch.save(train_masks, os.path.join(train_dir, "prompt_mask.pt"))
    
    # Save test data
    test_sequences = tokenized_sequences[test_indices]
    test_masks = attention_masks[test_indices]
    
    torch.save(test_sequences, os.path.join(test_dir, "prompt_sequences.pt"))
    torch.save(test_masks, os.path.join(test_dir, "prompt_mask.pt"))
    
    # Save metadata
    metadata = {
        'num_train_mazes': len(train_indices),
        'num_test_mazes': len(test_indices),
        'max_length': max_length,
        'pad_token_id': pad_token_id,
        'tokenizer_name': tokenizer_name,
        'shapes': {
            'train_sequences': list(train_sequences.shape),
            'test_sequences': list(test_sequences.shape)
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved tokenized maze data to {output_dir}")
    print(f"  Train: {len(train_indices)} mazes")
    print(f"  Test: {len(test_indices)} mazes")
    print(f"  Sequence shape: {train_sequences.shape}")


# TODO IF NEEDED, GIVE CHOICE TO GENERATE MAZE WITH RANDOM OR BACKTRACKING
def generate_and_save_maze_training_data(num_mazes: int = 1000,
                                        maze_size: int = 10,
                                        output_dir: str = "data/mazes",
                                        beam_search_params: Optional[Dict[str, Any]] = None,
                                        tokenizer_name: str = "gpt2",
                                        test_split_ratio: float = 0.1) -> None:
    """
    Complete pipeline: generate mazes, save as npz, convert to strings, and save as tokenized training data.
    
    Args:
        num_mazes: Number of mazes to generate
        maze_size: Size of each maze (maze_size x maze_size)
        output_dir: Directory to save all files
        beam_search_params: Parameters for beam search if including paths
        tokenizer_name: Name of the tokenizer to use
        test_split_ratio: Ratio of data to use for testing
    """
    print(f"Generating {num_mazes} mazes of size {maze_size}x{maze_size}...")
    
    # Generate mazes
    mazes = []
    for i in range(num_mazes):
        maze = generate_maze_recursive_backtracking(size=maze_size)
        mazes.append(maze)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_mazes} mazes")
    
    # Save mazes as strings
    maze_strings_file = os.path.join(output_dir, "maze_strings.json")
    os.makedirs(output_dir, exist_ok=True)
    
    save_mazes_as_strings(
        mazes, 
        maze_strings_file, 
        include_paths=beam_search_params is not None,
        beam_search_params=beam_search_params
    )
    
    # Save tokenized data
    save_tokenized_maze_data(
        maze_strings_file,
        output_dir,
        tokenizer_name=tokenizer_name,
        test_split_ratio=test_split_ratio
    )
    
    print(f"Complete pipeline finished. Data saved to {output_dir}")


def _run_single_maze_search(maze: Maze, beam_size: int, min_length: int, max_steps: int) -> SearchResult:
    """
    Run A* and beam search on a single maze.
    
    Args:
        maze: The maze to search
        beam_size: Beam width for beam search
        min_length: Minimum length of A* solution required
        max_steps: Maximum steps before giving up
    
    Returns:
        SearchResult containing all search outcomes
    """
    # Create beam search instance using the maze's navigation methods
    beam_search = AStarBeamSearch(
        next_states=maze.get_neighbors,
        heuristic_fn=maze.manhattan_distance,
        is_goal_fn=maze.is_goal,
        cost_fn=None  # Use additive costs (g + step_cost)
    )
    
    # Run A* search first to check solution length
    astar_solution, astar_cost = beam_search.solve_normal_astar(
        start_state=maze.start_pos,
        max_steps=max_steps
    )
    
    # Check if maze meets minimum length requirement
    if astar_solution is None or len(astar_solution) < min_length:
        return SearchResult(
            is_accepted=False,
            beam_solution=None,
            beam_cost=None,
            astar_solution=astar_solution,
            astar_cost=astar_cost,
            maze_data=None
        )
    
    # Run beam search
    beam_solution, state_matrix, bp_matrix, beam_cost = beam_search.solve(
        start_state=maze.start_pos,
        beam_size=beam_size,
        max_steps=max_steps
    )
    
    # Check acceptance criteria: beam search must find a solution
    if beam_solution is not None:
        # Store maze data for saving
        maze_data = {
            'maze': maze,
            'beam_solution': beam_solution,
            'state_matrix': state_matrix, 
            'bp_matrix': bp_matrix,
            'beam_cost': beam_cost,
            'astar_cost': astar_cost
        }
        
        return SearchResult(
            is_accepted=True,
            beam_solution=beam_solution,
            beam_cost=beam_cost,
            astar_solution=astar_solution,
            astar_cost=astar_cost,
            maze_data=maze_data
        )
    else:
        return SearchResult(
            is_accepted=False,
            beam_solution=None,
            beam_cost=None,
            astar_solution=astar_solution,
            astar_cost=astar_cost,
            maze_data=None
        )


def _save_search_results(stats: MazeSearchStats, maze_size: int, beam_size: int, 
                        backtrack_gen: bool, max_steps: int, logger: logging.Logger):
    """Save search results to files."""
    if not stats.accepted_mazes:
        logger.warning("No accepted mazes to save.")
        return
    
    logger.info(f"Saving results for {len(stats.accepted_mazes)} accepted mazes...")
    
    # Extract maze data for saving
    mazes = [maze_data['maze'] for maze_data in stats.accepted_mazes]
    
    # Save mazes using the multi-maze storage function
    # TODO: Implement save_multiple_mazes_compressed function
    maze_filename = f"beam_search_mazes_size{maze_size}_beam{beam_size}_count{len(stats.accepted_mazes)}.txt"
    logger.info(f"Would save {len(mazes)} mazes to '{maze_filename}' (function not implemented)")
    
    # Save search results data
    results_filename = f"beam_search_results_size{maze_size}_beam{beam_size}_count{len(stats.accepted_mazes)}.txt"
    
    summary = stats.get_summary()
    
    with open(results_filename, 'w') as f:
        f.write("Beam Search Experiment Results\n")
        f.write("=" * 40 + "\n\n")
        
        # Write experiment parameters
        f.write("Experiment Parameters:\n")
        f.write(f"  Maze size: {maze_size}x{maze_size}\n")
        f.write(f"  Beam size: {beam_size}\n")
        f.write(f"  Generation method: {'recursive_backtracking' if backtrack_gen else 'random'}\n")
        f.write(f"  Max steps: {max_steps}\n")
        f.write(f"  Total generation trials: {summary['total_generation_trials']}\n")
        f.write(f"  Accepted mazes: {summary['accepted_mazes']}\n\n")
        
        # Write statistics summary
        f.write("Statistics Summary:\n")
        f.write(f"  Acceptance rate: {summary['acceptance_rate']:.1f}%\n")
        f.write(f"  A* found solution but beam didn't: {summary['astar_only_solutions']}\n")
        f.write(f"  A* success rate: {summary['astar_success_rate']:.1f}%\n")
        f.write(f"  Beam success rate: {summary['beam_success_rate']:.1f}%\n")
        f.write(f"  Both found same solution: {summary['both_same_solution']} ({100*summary['both_same_solution']/summary['accepted_mazes']:.1f}%)\n")
        f.write(f"  Both found different solutions: {summary['both_different_solution']} ({100*summary['both_different_solution']/summary['accepted_mazes']:.1f}%)\n")
        
        # Write detailed maze results
        f.write("Detailed Maze Results, first five mazes:\n")
        f.write("=" * 20 + "\n\n")
        
        for i, maze_data in enumerate(stats.accepted_mazes[:5]):
            maze = maze_data['maze']
            f.write(f"Maze {i+1}:\n")
            f.write(f"  Start: {maze.start_pos}, End: {maze.end_pos}, Size: {maze.size}x{maze.size}\n")
            f.write(f"  Walls: {len(maze.walls)} walls\n")
            
            beam_solution = maze_data['beam_solution']
            beam_cost = maze_data['beam_cost']
            astar_cost = maze_data['astar_cost']
            
            f.write(f"  Beam solution: {len(beam_solution)} steps, cost {beam_cost:.2f}\n")
            f.write(f"  A* solution cost: {astar_cost:.2f}\n")
            
            # Create maze visualizations with paths
            f.write(f"\n  Maze visualization with beam search path:\n")
            beam_visualization = maze.visualize(beam_solution)
            f.write("  ")
            f.write(beam_visualization.replace('\n', '\n  '))
            f.write("\n")
            
            f.write(f"\n  Legend: S=Start, E=End, #=Wall, arrows show beam search path direction\n")
            f.write("\n")
    
    logger.info(f"Saved detailed results to '{results_filename}'")


def batch_maze_search(num_trials: int = 10, maze_size: int = 10, beam_size: int = 4, 
                      backtrack_gen: bool = True, min_length: int = 5, max_steps: int = 1000,
                      save_results: bool = False, verbose: bool = False):
    """
    Run multiple maze searches and track statistics comparing A* vs Beam Search.
    Only accepts mazes where beam search found a solution, and A*'s solution is at least min_length.
    
    Args:
        num_trials: Number of accepted maze instances to test
        maze_size: Size of each maze (maze_size x maze_size)
        beam_size: Beam width for beam search
        backtrack_gen: Whether to generate mazes using backtracking (true), or randomly (false).
        min_length: Minimum length of A* solution required to accept a maze.
        max_steps: Maximum steps before giving up for A* and beam search.
        save_results: Whether to save the generated mazes & search results.
        verbose: Whether to use verbose logging (DEBUG level)
    
    Returns:
        MazeSearchStats object containing all statistics
    """
    
    # Setup logging
    setup_logging(log_level="DEBUG" if verbose else "INFO")
    logger = logging.getLogger(__name__)
    
    with log_execution_time(f"Batch maze search ({num_trials} trials)"):
        # Log configuration
        logger.info(f"Starting batch maze search:")
        logger.info(f"  Trials: {num_trials}")
        logger.info(f"  Maze size: {maze_size}x{maze_size}")
        logger.info(f"  Beam size: {beam_size}")
        logger.info(f"  Min length: {min_length}")
        logger.info(f"  Generation method: {'recursive_backtracking' if backtrack_gen else 'random'}")
        
        # Initialize tracking objects
        stats = MazeSearchStats()
        progress = ProgressTracker(num_trials, "Maze Trials", log_every=10)
        
        while len(stats.accepted_mazes) < num_trials:
            # Generate maze
            if backtrack_gen:
                maze = generate_maze_recursive_backtracking(size=maze_size)
            else:
                maze = generate_maze_random(size=maze_size, wall_density=0.4)
            
            # Run search comparison
            search_result = _run_single_maze_search(maze, beam_size, min_length, max_steps)
            
            # Update statistics
            stats.update(search_result)
            
            # Log result details
            if search_result.is_accepted:
                progress.update(message=f"Accepted: {search_result.summary()}")
                logger.debug(f"Trial {len(stats.accepted_mazes)}/{num_trials}: {search_result.summary()}")
            else:
                if search_result.astar_solution is None or len(search_result.astar_solution) < min_length:
                    logger.debug(f"Trial {stats.total_generation_trials}: Rejected - A* solution length {len(search_result.astar_solution) if search_result.astar_solution else 'None'} < {min_length}")
                else:
                    logger.debug(f"Trial {stats.total_generation_trials}: Rejected - beam search failed")
        
        # Log final summary
        log_final_summary(logger, stats)
        
        # Save results if requested
        if save_results:
            _save_search_results(stats, maze_size, beam_size, backtrack_gen, max_steps, logger)
        else:
            logger.info("Not saving results (save_results=False)")
    
    return stats


def main():
    """Example usage of the maze training data generation pipeline."""
    
    # print("Maze Training Data Generation Pipeline")
    # print("=" * 50)
    
    # # Example: Generate a small dataset
    # beam_search_params = {
    #     'beam_size': 4,
    #     'max_steps': 1000
    # }
    
    # generate_and_save_maze_training_data(
    #     num_mazes=100,
    #     maze_size=10,
    #     output_dir="data/mazes_example",
    #     beam_search_params=beam_search_params,
    #     tokenizer_name="gpt2",
    #     test_split_ratio=0.1
    # )
    
    # print("\nExample completed successfully!")

    # print("\n" + "="*80)
    # print("BATCH TESTING")
    # print("="*80)
    
    # Run batch testing with verbose logging
    stats = batch_maze_search(
        num_trials=100, 
        maze_size=10, 
        beam_size=4, 
        backtrack_gen=False, 
        min_length=10, 
        save_results=True,
        verbose=True
    )
    
    # Example of accessing statistics programmatically
    summary = stats.get_summary()
    print(f"\nProgrammatic access to results:")
    print(f"  Acceptance rate: {summary['acceptance_rate']:.1f}%")
    print(f"  Total trials: {summary['total_generation_trials']}")

if __name__ == "__main__":
    main()
