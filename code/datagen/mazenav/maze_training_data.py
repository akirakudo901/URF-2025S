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
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from transformers import GPT2TokenizerFast

# Import maze generation functions
from generate_maze import (
    Maze,
    generate_maze_recursive_backtracking, 
    generate_maze_random,
    mazes_to_npz,
    npz_to_mazes,
    path_to_string_format
)

# Import beam search functions
from astar_beam_search import AStarBeamSearch, StateT

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

# Import data generation utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.datagen import save_generated_datasets

def load_and_tokenize_mazes(maze_npz: str,
                           tokenizer=None) -> torch.Tensor:
    """
    Load maze npz from file and tokenize them.
    
    Args:
        maze_npz: Path to NPZ file containing mazes
        tokenizer: Tokenizer to use (optional, defaults to GPT2TokenizerFast)
        
    Returns:
        tokenized_sequences: torch.Tensor of shape (num_mazes, max_length)
    """
    # Load tokenizer
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load maze data
    maze_data = npz_to_mazes(maze_npz)
    # Extract maze strings
    maze_strings = [m.to_string_format(add_special_tokens=False) for m in maze_data]
    
    # Tokenize
    tokenized = tokenizer(
        maze_strings,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors="pt"
    )
    
    return tokenized['input_ids']


def save_tokenized_maze_data(maze_npz: str,
                            output_dir: str,
                            tokenizer_name: str = "gpt2",
                            pad_token_id: int = 50256,
                            max_length: int = 1024,
                            test_split_ratio: float = 0.1,
                            random_seed: int = 42) -> None:
    """
    Load mazes, tokenize them, and save as torch files compatible with training.py.
    
    This creates two files, prompt_sequences.pt and prompt_mask.pt (padding masks).
    
    Args:
        maze_npz: Path to JSON file containing maze strings
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
    print(f"Loading and tokenizing mazes from {maze_npz}...")
    tokenized_sequences = load_and_tokenize_mazes(
        maze_npz, tokenizer
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


def _state_matrix_to_string_format(state_matrix: List[List[Optional[Tuple[int, int]]]]) -> List[List[str]]:
    """
    Convert state matrix to string format for tokenization.
    
    Args:
        state_matrix: List[List[Optional[Tuple[int, int]]]] of shape [K, L] where K=beam_size, L=iterations
        Each entry is either a (x, y) position tuple or None
        
    Returns:
        List[List[str]] of shape [K, 3L] where each (x, y) becomes ["beam", str(x), str(y)]
        and None entries become ["beam", "None", "None"]
    """
    result = []
    for beam_idx, beam_row in enumerate(state_matrix):
        string_row = []
        for state in beam_row:
            if state is not None:
                x, y = state
                string_row.extend(["beam", str(x), str(y)])
            else:
                string_row.extend(["beam", "None", "None"])
        result.append(string_row)
    return result


def _extend_backpointer_matrix(bp_matrix: List[List[Optional[int]]]) -> List[List[str]]:
    """
    Extend backpointer matrix to align with tokenized state matrix.
    
    Args:
        bp_matrix: List[List[Optional[int]]] of shape [K, L] where entries are parent rank indices or None
        
    Returns:
        List[List[str]] of shape [K, 3L] where each bp entry becomes [str(bp), str(rank), str(rank)]
        and None entries become ["None", "None", "None"]
    """
    result = []
    for rank_idx, bp_row in enumerate(bp_matrix):
        extended_row = []
        for bp_val in bp_row:
            if bp_val is not None:
                extended_row.extend([str(bp_val), str(rank_idx), str(rank_idx)])
            else:
                extended_row.extend(["None", "None", "None"])
        result.append(extended_row)
    return result


def load_and_tokenize_beam_sols(beam_sols_npz: str, tokenizer):
    """
    Load beam search solutions from NPZ and tokenize them for training.
    
    Args:
        beam_sols_npz: Path to NPZ file containing beam search data
        tokenizer: Tokenizer to use for converting strings to token IDs
        
    Returns:
        Dict containing tokenized data as PyTorch tensors ready for training
    """
    # First load the data
    state_matrices, bp_matrices, beam_sols, beam_costs = _npz_to_maze_beam_sols(beam_sols_npz)
    
    # Convert each state matrix to string format and tokenize
    tokenized_state_matrices = []
    for state_matrix in state_matrices:
        string_matrix = _state_matrix_to_string_format(state_matrix)
        
        # Tokenize each row in the matrix
        tokenized_rows = []
        for row in string_matrix:
            tokenized_row = tokenizer.encode(" ".join(row))  # Shape: [3L] where L=iterations
            tokenized_rows.append(tokenized_row)
        tokenized_state_matrices.append(tokenized_rows)
    
    # Extend backpointer matrices to align with tokenized state matrices
    extended_bp_matrices = []
    for bp_matrix in bp_matrices:
        extended_bp = _extend_backpointer_matrix(bp_matrix)
        # Tokenize each row in the extended backpointer matrix
        tokenized_rows = []
        for row in extended_bp:
            tokenized_row = tokenizer.encode(" ".join(row))  # Shape: [3L] where L=iterations
            tokenized_rows.append(tokenized_row)
        extended_bp_matrices.append(tokenized_rows)

    # Convert beam solutions to string format using path_to_string_format
    tokenized_beam_sols = []
    for beam_sol in beam_sols:
        path_string = path_to_string_format(beam_sol, add_special_tokens=True)
        tokenized_path = tokenizer.encode(path_string)  # Shape: [variable_length]
        tokenized_beam_sols.append(tokenized_path)
    
    # Convert to PyTorch tensors with proper padding
    num_mazes = len(state_matrices)
    
    # Find maximum dimensions for padding
    max_beam_size = max(len(matrix) for matrix in tokenized_state_matrices) if tokenized_state_matrices else 0
    max_sequence_length = max(max(len(row) for row in matrix) for matrix in tokenized_state_matrices) if tokenized_state_matrices else 0
    max_solution_length = max(len(sol) for sol in tokenized_beam_sols) if tokenized_beam_sols else 0
    
    # Create padded tensors
    state_matrices_tensor = torch.full((num_mazes, max_beam_size, max_sequence_length), tokenizer.pad_token_id, dtype=torch.long)  # Shape: [B, max_beam_size, max_sequence_length]
    bp_matrices_tensor    = torch.full((num_mazes, max_beam_size, max_sequence_length), tokenizer.pad_token_id, dtype=torch.long)  # Shape: [B, max_beam_size, max_sequence_length]
    beam_sols_tensor      = torch.full((num_mazes, max_solution_length), tokenizer.pad_token_id, dtype=torch.long)  # Shape: [B, max_solution_length]
    beam_costs_tensor     = torch.tensor(beam_costs, dtype=torch.float32)  # Shape: [B]
    
    # Fill tensors with actual data
    for maze_idx in range(num_mazes):
        # Fill state matrix tensor
        state_matrix = tokenized_state_matrices[maze_idx]
        for beam_idx in range(len(state_matrix)):
            sequence = state_matrix[beam_idx]
            state_matrices_tensor[maze_idx, beam_idx, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        
        # Fill backpointer matrix tensor
        bp_matrix = extended_bp_matrices[maze_idx]
        for beam_idx in range(len(bp_matrix)):
            sequence = bp_matrix[beam_idx]
            bp_matrices_tensor[maze_idx, beam_idx, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        
        # Fill beam solution tensor
        beam_sol = tokenized_beam_sols[maze_idx]
        beam_sols_tensor[maze_idx, :len(beam_sol)] = torch.tensor(beam_sol, dtype=torch.long)
    
    # Create result dictionary with tensors
    result = {
        'state_matrices': state_matrices_tensor,
        'bp_matrices': bp_matrices_tensor, 
        'beam_sols': beam_sols_tensor,
        'beam_costs': beam_costs_tensor
    }
    
    print(f"Loaded and tokenized beam search data from {beam_sols_npz}")
    print(f"  {num_mazes} mazes processed")
    print(f"  State matrices tensor shape: {state_matrices_tensor.shape}")
    print(f"  Backpointer matrices tensor shape: {bp_matrices_tensor.shape}")
    print(f"  Beam solutions tensor shape: {beam_sols_tensor.shape}")
    print(f"  Beam costs tensor shape: {beam_costs_tensor.shape}")
    
    return result

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

def _maze_beam_sols_to_npz(state_matrices : List[List[List[Tuple[int, int]]]], 
                           bp_matrices : List[List[List[int]]],
                           beam_sols : List[List[Tuple[int, int]]], 
                           beam_costs : List[float], 
                           filename : str):
    """
    Save beam search data to NPZ format with proper padding.
    
    Args:
        state_matrices: List of state matrices, each of shape [beam_size, max_steps] with Tuple[int, int] states
        bp_matrices: List of backpointer matrices, each of shape [beam_size, max_steps] with int indices
        beam_sols: List of solution paths, each of variable length with Tuple[int, int] positions
        beam_costs: List of solution costs as floats
        filename: Path to store beam search data to
    """
    num_mazes = len(state_matrices)
    
    # Find maximum dimensions across all mazes for proper padding
    max_beam_size = max(len(matrix) for matrix in state_matrices) if state_matrices else 0
    max_steps = max(max(len(row) for row in matrix) for matrix in state_matrices) if state_matrices else 0
    max_solution_length = max(len(sol) for sol in beam_sols) if beam_sols else 0
    
    # Initialize arrays with proper paddin
    padded_state_matrices = np.full((num_mazes, max_beam_size, max_steps, 2), -1, dtype=np.int8)  # Shape: [B, max_beam_size, max_steps, 2]    
    padded_bp_matrices = np.full((num_mazes, max_beam_size, max_steps), -1, dtype=np.int8)  # Shape: [B, max_beam_size, max_steps]
    padded_beam_sols = np.full((num_mazes, max_solution_length, 2), -1, dtype=np.int8)  # Shape: [B, max_solution_length, 2]
    beam_costs_array = np.array(beam_costs, dtype=np.float32)  # Shape: [B]
    
    # Fill arrays with actual data
    for maze_idx in range(num_mazes):
        # Fill state matrix
        state_matrix = state_matrices[maze_idx]
        for beam_idx in range(len(state_matrix)):
            for step_idx in range(len(state_matrix[beam_idx])):
                if state_matrix[beam_idx][step_idx] is not None:
                    x, y = state_matrix[beam_idx][step_idx]
                    padded_state_matrices[maze_idx, beam_idx, step_idx] = [x, y]
        
        # Fill backpointer matrix
        bp_matrix = bp_matrices[maze_idx]
        for beam_idx in range(len(bp_matrix)):
            for step_idx in range(len(bp_matrix[beam_idx])):
                if bp_matrix[beam_idx][step_idx] is not None:
                    padded_bp_matrices[maze_idx, beam_idx, step_idx] = bp_matrix[beam_idx][step_idx]
        
        # Fill solution path
        beam_sol = beam_sols[maze_idx]
        for pos_idx in range(len(beam_sol)):
            x, y = beam_sol[pos_idx]
            padded_beam_sols[maze_idx, pos_idx] = [x, y]
    
    # Save to NPZ file
    np.savez_compressed(filename,
                       state_matrices=padded_state_matrices,
                       bp_matrices=padded_bp_matrices, 
                       beam_sols=padded_beam_sols,
                       beam_costs=beam_costs_array)
    
    print(f"Saved beam search data to {filename}")
    print(f"  {num_mazes} mazes, max beam size: {max_beam_size}, max steps: {max_steps}, max solution length: {max_solution_length}")


def _npz_to_maze_beam_sols(filename: str):
    """
    Load beam search data from NPZ format and restore original data structures.
    
    Args:
        filename: Path to the NPZ file containing beam search data
        
    Returns:
        Tuple of (state_matrices, bp_matrices, beam_sols, beam_costs) in original format
    """
    # Load NPZ file
    data = np.load(filename)
    
    state_matrices_padded = data['state_matrices']  # [num_mazes, max_beam_size, max_steps, 2]
    bp_matrices_padded = data['bp_matrices']        # [num_mazes, max_beam_size, max_steps]
    beam_sols_padded = data['beam_sols']            # [num_mazes, max_solution_length, 2]
    beam_costs_array = data['beam_costs']           # [num_mazes]
    
    num_mazes = state_matrices_padded.shape[0]
    
    # Restore original data structures
    state_matrices = []
    bp_matrices = []
    beam_sols = []
    beam_costs = []
    
    for maze_idx in range(num_mazes):
        # Restore state matrix - convert from padded array back to list of lists with None values
        state_matrix = []
        for beam_idx in range(state_matrices_padded.shape[1]):
            beam_states = []
            for step_idx in range(state_matrices_padded.shape[2]):
                state_coord = state_matrices_padded[maze_idx, beam_idx, step_idx]
                if state_coord[0] != -1 and state_coord[1] != -1:  # Not padding
                    beam_states.append((int(state_coord[0]), int(state_coord[1])))
                else:
                    beam_states.append(None)
            state_matrix.append(beam_states)
        state_matrices.append(state_matrix)
        
        # Restore backpointer matrix - convert from padded array back to list of lists with None values
        bp_matrix = []
        for beam_idx in range(bp_matrices_padded.shape[1]):
            bp_indices = []
            for step_idx in range(bp_matrices_padded.shape[2]):
                bp_val = bp_matrices_padded[maze_idx, beam_idx, step_idx]
                if bp_val != -1:  # Not padding
                    bp_indices.append(int(bp_val))
                else:
                    bp_indices.append(None)
            bp_matrix.append(bp_indices)
        bp_matrices.append(bp_matrix)
        
        # Restore solution path - convert from padded array back to list of tuples
        beam_sol = []
        for pos_idx in range(beam_sols_padded.shape[1]):
            pos_coord = beam_sols_padded[maze_idx, pos_idx]
            if pos_coord[0] != -1 and pos_coord[1] != -1:  # Not padding
                beam_sol.append((int(pos_coord[0]), int(pos_coord[1])))
        beam_sols.append(beam_sol)
        
        # Restore cost
        beam_costs.append(float(beam_costs_array[maze_idx]))
    
    print(f"Loaded beam search data from {filename}")
    print(f"  {num_mazes} mazes restored to original format")
    
    return state_matrices, bp_matrices, beam_sols, beam_costs


def _save_search_results(stats: MazeSearchStats, maze_size: int, beam_size: int, 
                        backtrack_gen: bool, max_steps: int, logger: logging.Logger,
                        save_path: Optional[str] = None, tokenizer=None, train_test_split: float = 0.8, 
                        seed: Optional[int] = None):
    """Save search results to files."""
    if not stats.accepted_mazes:
        logger.warning("No accepted mazes to save.")
        return
    
    logger.info(f"Saving results for {len(stats.accepted_mazes)} accepted mazes...")
    
    # Extract maze data for saving
    mazes          = [maze_data['maze']          for maze_data in stats.accepted_mazes]
    state_matrices = [maze_data['state_matrix']  for maze_data in stats.accepted_mazes]
    bp_matrices    = [maze_data['bp_matrix']     for maze_data in stats.accepted_mazes]
    beam_sols      = [maze_data['beam_solution'] for maze_data in stats.accepted_mazes]
    beam_costs     = [maze_data['beam_cost']     for maze_data in stats.accepted_mazes]
    
    # Split data into train/test sets using randomized indexing
    indices = list(range(len(mazes)))
    
    # Set random seed for deterministic shuffling if provided
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using deterministic seed {seed} for train/test split")
    
    random.shuffle(indices)  # Randomize the order
    
    train_size = int(len(mazes) * train_test_split)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    
    # Split all data arrays
    train_mazes = [mazes[i] for i in train_indices]
    train_state_matrices = [state_matrices[i] for i in train_indices]
    train_bp_matrices = [bp_matrices[i] for i in train_indices]
    train_beam_sols = [beam_sols[i] for i in train_indices]
    train_beam_costs = [beam_costs[i] for i in train_indices]
    
    test_mazes = [mazes[i] for i in test_indices]
    test_state_matrices = [state_matrices[i] for i in test_indices]
    test_bp_matrices = [bp_matrices[i] for i in test_indices]
    test_beam_sols = [beam_sols[i] for i in test_indices]
    test_beam_costs = [beam_costs[i] for i in test_indices]
    
    logger.info(f"Split {len(mazes)} mazes into {len(train_mazes)} training and {len(test_mazes)} testing samples")
    
    # Save in npz format - separate train and test files
    basename = f"{maze_size}x{maze_size}_{beam_size}beam_{'backtrack' if backtrack_gen else 'random'}"
    
    # Save training data
    train_maze_path = f"{save_path}/{basename}_train_mazes.npz"
    train_beamsol_path = f"{save_path}/{basename}_train_beam_sols.npz"
    mazes_to_npz(train_mazes, train_maze_path)
    _maze_beam_sols_to_npz(train_state_matrices, train_bp_matrices, train_beam_sols, train_beam_costs, train_beamsol_path)
    
    # Save testing data
    test_maze_path = f"{save_path}/{basename}_test_mazes.npz"
    test_beamsol_path = f"{save_path}/{basename}_test_beam_sols.npz"
    mazes_to_npz(test_mazes, test_maze_path)
    _maze_beam_sols_to_npz(test_state_matrices, test_bp_matrices, test_beam_sols, test_beam_costs, test_beamsol_path)
    
    # Load and tokenize training data
    tokenized_train_mazes = load_and_tokenize_mazes(train_maze_path, tokenizer)  # Shape: [B_train, max_length]
    train_result = load_and_tokenize_beam_sols(train_beamsol_path, tokenizer)
    train_state_matrices = train_result['state_matrices']  # Shape: [B_train, max_beam_size, max_sequence_length]
    train_bp_matrices = train_result['bp_matrices']        # Shape: [B_train, max_beam_size, max_sequence_length]
    train_beam_sols = train_result['beam_sols']            # Shape: [B_train, max_solution_length]
    train_beam_costs = train_result['beam_costs']          # Shape: [B_train]
    
    # Load and tokenize testing data
    tokenized_test_mazes = load_and_tokenize_mazes(test_maze_path, tokenizer)  # Shape: [B_test, max_length]
    test_result = load_and_tokenize_beam_sols(test_beamsol_path, tokenizer)
    test_state_matrices = test_result['state_matrices']  # Shape: [B_test, max_beam_size, max_sequence_length]
    test_bp_matrices = test_result['bp_matrices']        # Shape: [B_test, max_beam_size, max_sequence_length]
    test_beam_sols = test_result['beam_sols']            # Shape: [B_test, max_solution_length]
    test_beam_costs = test_result['beam_costs']          # Shape: [B_test]
    
    logger.info(f"Saved training data: {train_maze_path}, {train_beamsol_path}")
    logger.info(f"Saved testing data: {test_maze_path}, {test_beamsol_path}")
    
    # Prepare data in the format expected by save_generated_datasets
    datasets = {
        f'beam_width_{beam_size}': {
            'train': {
                'prompt_sequences': tokenized_train_mazes,  # Shape: [B_train, max_length]
                'cot_sequences': train_state_matrices,       # Shape: [B_train, max_beam_size, max_sequence_length]
                'prompt_mask': torch.ones_like(tokenized_train_mazes, dtype=torch.bool),  # All tokens are valid
                'cot_mask': torch.ones_like(train_state_matrices, dtype=torch.bool),      # All tokens are valid
                'backpointers': train_bp_matrices,           # Shape: [B_train, max_beam_size, max_sequence_length]
                'beam_solutions': train_beam_sols,            # Shape: [B_train, max_solution_length]
                'beam_costs': train_beam_costs,              # Shape: [B_train]
                'metadata': {
                    'num_prompts': len(train_mazes),
                    'beam_width': beam_size,
                    'max_prompt_len': tokenized_train_mazes.shape[1],
                    'max_cot_len': train_state_matrices.shape[2],
                    'pad_token_id': tokenizer.pad_token_id,
                }
            },
            'test': {
                'prompt_sequences': tokenized_test_mazes,     # Shape: [B_test, max_length]
                'cot_sequences': test_state_matrices,         # Shape: [B_test, max_beam_size, max_sequence_length]
                'prompt_mask': torch.ones_like(tokenized_test_mazes, dtype=torch.bool),  # All tokens are valid
                'cot_mask': torch.ones_like(test_state_matrices, dtype=torch.bool),       # All tokens are valid
                'backpointers': test_bp_matrices,             # Shape: [B_test, max_beam_size, max_sequence_length]
                'beam_solutions': test_beam_sols,             # Shape: [B_test, max_solution_length]
                'beam_costs': test_beam_costs,                # Shape: [B_test]
                'metadata': {
                    'num_prompts': len(test_mazes),
                    'beam_width': beam_size,
                    'max_prompt_len': tokenized_test_mazes.shape[1],
                    'max_cot_len': test_state_matrices.shape[2],
                    'pad_token_id': tokenizer.pad_token_id,
                }
            }
        }
    }
    
    # Save using the modified save_generated_datasets function
    save_generated_datasets(datasets, save_path, data_type="maze")
    

    """
    TODO REMOVE
    DONE -> First, turn list of mazes into npz
    DONE -> Then, also format them into string using maze_to_string from generate_maze.py, followed by tokenization with NEW FUNCTION 1

    DONE -> Also, save the state and back pointer matrices as well as beam path in compact format (npz, implement NEW FUNCTION 2)
    Finally, tokenize those:
    DONE -> state matrix will first be converted into appropriate string via a variant of path_to_string function from generate_maze.py
    DONE -> then, the state & backpointers will be adjusted to accommodate for example 3 tokens for each 'step'
      in beam search (e.g. 'plan' '3' '4'), using NEW FUNCTION 3
    DONE ->  then, state matrix will be tokenized accordingly using a function (WANNA REUSE NEW FUNCTION 1 IF POSSIBLE)
    DONE ->  solution path is also tokenized and stored in its own arrays (NEW FUNCTION 4)

    We then finally divide the set into training and testing sets, and then we save the result.
    TODO REMOVE END
    """

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
                      save_path: Optional[str] = None, tokenizer=None, verbose: bool = False,
                      train_test_split: float = 0.8, seed: Optional[int] = None):
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
        save_path: Path to save generated mazes & search results as npz, if provided.
        tokenizer: Tokenizer used for both maze and paths.
        verbose: Whether to use verbose logging (DEBUG level)
        train_test_split: Proportion of data to use for training (default 0.8, meaning 80% train, 20% test)
        seed: Random seed for deterministic train/test split. If None, uses random seed each time.
    
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
        
        # Track unique mazes to avoid duplicates
        seen_mazes = set()  # Set of maze hashes we've already processed
        duplicate_count = 0
        
        while len(stats.accepted_mazes) < num_trials:
            # Generate maze
            if backtrack_gen:
                maze = generate_maze_recursive_backtracking(size=maze_size)
            else:
                maze = generate_maze_random(size=maze_size, wall_density=0.4)
            
            # Check for duplicates
            maze_hash = hash(maze)
            if maze_hash in seen_mazes:
                duplicate_count += 1
                logger.debug(f"Trial {stats.total_generation_trials + 1}: Duplicate maze detected, skipping")
                continue
            
            # Add to seen mazes
            seen_mazes.add(maze_hash)
            
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
        
        # Log duplicate detection statistics
        logger.info(f"Duplicate detection summary:")
        logger.info(f"  Total unique mazes generated: {len(seen_mazes)}")
        logger.info(f"  Duplicates detected and skipped: {duplicate_count}")
        logger.info(f"  Duplicate rate: {duplicate_count / (len(seen_mazes) + duplicate_count) * 100:.2f}%")
        
        # Save results if requested
        if save_path:
            _save_search_results(stats, maze_size, beam_size, backtrack_gen, max_steps, logger, save_path, tokenizer, train_test_split, seed)
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
