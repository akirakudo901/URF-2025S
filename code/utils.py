# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/20

import torch
import pandas as pd
from transformers import GPT2Tokenizer
import numpy as np
from typing import List, Tuple, Optional
import os

def read_parquet_to_tensors(file_path: str, 
                           tokenizer: Optional[GPT2Tokenizer] = None,
                           max_prompts: Optional[int] = None,
                           random_seed: int = 42) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """
    Read a parquet file and convert prompts and CoTs into token tensors.
    
    Args:
        file_path (str): Path to the parquet file
        tokenizer (GPT2Tokenizer, optional): Tokenizer to use. If None, will load GPT2 tokenizer
        max_prompts (int, optional): Maximum number of prompts to process. If None, processes all
        random_seed (int): Random seed for sampling if max_prompts is specified
    
    Returns:
        tuple: (prompts, cot_sequences) where:
            - prompts: List of P prompts, each a list of token IDs
            - cot_sequences: List of P lists, each containing M CoT sequences as token IDs
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Read parquet file
    print(f"Reading parquet file: {file_path}")
    df = pd.read_parquet(file_path)
    
    # Validate columns
    required_columns = ['query', 'response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Found {len(df):,} rows in dataset")
    
    # Sample data if max_prompts is specified
    if max_prompts and max_prompts < len(df):
        np.random.seed(random_seed)
        df = df.sample(n=max_prompts, random_state=random_seed).reset_index(drop=True)
        print(f"Sampled {max_prompts:,} prompts for processing")
    
    # Group by query to get multiple responses per prompt
    print("Grouping prompts and responses...")
    grouped = df.groupby('query')['response'].apply(list).reset_index()
    
    print(f"Found {len(grouped):,} unique prompts")
    print(f"Average responses per prompt: {grouped['response'].apply(len).mean():.1f}")
    
    # Convert to tensors efficiently
    prompts = []
    cot_sequences = []
    
    print("Converting to tokens...")
    for idx, (query, responses) in enumerate(zip(grouped['query'], grouped['response'])):
        if idx % 1000 == 0:
            print(f"  Processed {idx:,} prompts...")
        
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(query, add_special_tokens=True)
        prompts.append(prompt_tokens)
        
        # Tokenize all responses for this prompt
        cot_tokens_list = []
        for response in responses:
            if pd.isna(response) or response == "":
                continue
            cot_tokens = tokenizer.encode(response, add_special_tokens=True)
            cot_tokens_list.append(cot_tokens)
        
        cot_sequences.append(cot_tokens_list)
    
    print(f"Conversion complete!")
    print(f"Total prompts: {len(prompts):,}")
    print(f"Total CoT sequences: {sum(len(cots) for cots in cot_sequences):,}")
    
    return prompts, cot_sequences

def batch_parquet_files(file_paths: List[str], 
                       tokenizer: Optional[GPT2Tokenizer] = None,
                       max_prompts_per_file: Optional[int] = None,
                       random_seed: int = 42) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """
    Read multiple parquet files and combine them into a single dataset.
    
    Args:
        file_paths (List[str]): List of parquet file paths
        tokenizer (GPT2Tokenizer, optional): Tokenizer to use
        max_prompts_per_file (int, optional): Maximum prompts per file
        random_seed (int): Random seed for sampling
    
    Returns:
        tuple: Combined (prompts, cot_sequences) from all files
    """
    all_prompts = []
    all_cot_sequences = []
    
    for i, file_path in enumerate(file_paths):
        print(f"\nProcessing file {i+1}/{len(file_paths)}: {file_path}")
        
        prompts, cot_sequences = read_parquet_to_tensors(
            file_path, tokenizer, max_prompts_per_file, random_seed + i
        )
        
        all_prompts.extend(prompts)
        all_cot_sequences.extend(cot_sequences)
    
    print(f"\nCombined dataset:")
    print(f"Total prompts: {len(all_prompts):,}")
    print(f"Total CoT sequences: {sum(len(cots) for cots in all_cot_sequences):,}")
    
    return all_prompts, all_cot_sequences

def get_dataset_stats(prompts: List[List[int]], 
                     cot_sequences: List[List[List[int]]]) -> dict:
    """
    Get statistics about the processed dataset.
    
    Args:
        prompts: List of prompt token sequences
        cot_sequences: List of CoT token sequences
    
    Returns:
        dict: Dataset statistics
    """
    prompt_lengths = [len(p) for p in prompts]
    cot_lengths = [len(cot) for cots in cot_sequences for cot in cots]
    cots_per_prompt = [len(cots) for cots in cot_sequences]
    
    stats = {
        'num_prompts': len(prompts),
        'total_cot_sequences': sum(cots_per_prompt),
        'avg_cots_per_prompt': np.mean(cots_per_prompt),
        'prompt_lengths': {
            'min': min(prompt_lengths),
            'max': max(prompt_lengths),
            'mean': np.mean(prompt_lengths),
            'median': np.median(prompt_lengths),
            'std': np.std(prompt_lengths)
        },
        'cot_lengths': {
            'min': min(cot_lengths) if cot_lengths else 0,
            'max': max(cot_lengths) if cot_lengths else 0,
            'mean': np.mean(cot_lengths) if cot_lengths else 0,
            'median': np.median(cot_lengths) if cot_lengths else 0,
            'std': np.std(cot_lengths) if cot_lengths else 0
        }
    }
    
    return stats

def process_dataset(prompts, cot_sequences, pad_token_id=0):
    """
    Process a dataset and create batches for training.
    
    Args:
        prompts (list): List of P prompts, where each prompt is a list of token IDs
        cot_sequences (list): List of P lists, where each inner list contains M CoT sequences
        pad_token_id (int): Token ID to use for padding, defaults to 0
    
    Returns:
        tuple: (prompt_sequences, cot_sequences, prompt_mask, cot_mask)
    """
    P = len(prompts)
    
    # Validate input
    if len(cot_sequences) != P:
        raise ValueError(f"Number of prompt groups ({P}) must match number of CoT groups ({len(cot_sequences)})")
    
    # Find dimensions
    M = max(len(cots) for cots in cot_sequences)
    max_prompt_len = max(len(prompt) for prompt in prompts)
    max_cot_len = max(len(cot) for cots in cot_sequences for cot in cots)
    
    # Pre-allocate tensors
    prompt_sequences = torch.full((P, M, max_prompt_len), pad_token_id, dtype=torch.long)
    cot_sequences_tensor = torch.full((P, M, max_cot_len), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((P, M, max_prompt_len), dtype=torch.float)
    cot_mask = torch.zeros((P, M, max_cot_len), dtype=torch.float)
    
    # Process prompts
    for p_idx, prompt in enumerate(prompts):
        prompt_tensor = torch.tensor(prompt, dtype=torch.long)
        prompt_len = len(prompt)
        
        # Fill all M copies at once using broadcasting
        prompt_sequences[p_idx, :, :prompt_len] = prompt_tensor.unsqueeze(0).expand(M, -1)
        prompt_mask[p_idx, :, :prompt_len] = 1.0
    
    # Process cots
    for p_idx, cots in enumerate(cot_sequences):
        if len(cots) == 0:
            continue
            
        # Convert all CoTs to tensors at once
        cot_tensors = [torch.tensor(cot, dtype=torch.long) for cot in cots]
        cot_lengths = torch.tensor([len(cot) for cot in cots], dtype=torch.long)
        
        # Use scatter_ for efficient CoT filling
        for m_idx, (cot_tensor, cot_len) in enumerate(zip(cot_tensors, cot_lengths)):
            cot_sequences_tensor[p_idx, m_idx, :cot_len] = cot_tensor
            cot_mask[p_idx, m_idx, :cot_len] = 1.0
        
    return prompt_sequences, cot_sequences_tensor, prompt_mask, cot_mask

def save_tensors_in_usable_form(file_path : str):
    """
    Example usage of the parquet reading and tensor conversion functions.
    """
    try:
        # Read and convert to tensors
        prompts, cot_sequences = read_parquet_to_tensors(
            file_path, 
            max_prompts=None
        )
        
        # Get dataset statistics
        stats = get_dataset_stats(prompts, cot_sequences)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Process into training format
        prompt_sequences, cot_sequences_tensor, prompt_mask, cot_mask = process_dataset(
            prompts, cot_sequences
        )
        
        print(f"\nTraining tensors shape:")
        print(f"  prompt_sequences: {prompt_sequences.shape}")
        print(f"  cot_sequences_tensor: {cot_sequences_tensor.shape}")
        print(f"  prompt_mask: {prompt_mask.shape}")
        print(f"  cot_mask: {cot_mask.shape}")
        
        # Create output directory
        output_dir = "data/GSM8K"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tensors
        print(f"\nSaving tensors to {output_dir}...")
        torch.save(prompt_sequences, os.path.join(output_dir, "prompt_sequences.pt"))
        torch.save(cot_sequences_tensor, os.path.join(output_dir, "cot_sequences_tensor.pt"))
        torch.save(prompt_mask, os.path.join(output_dir, "prompt_mask.pt"))
        torch.save(cot_mask, os.path.join(output_dir, "cot_mask.pt"))
        
        # Save raw prompts and cot_sequences for reference
        torch.save(prompts, os.path.join(output_dir, "prompts_raw.pt"))
        torch.save(cot_sequences, os.path.join(output_dir, "cot_sequences_raw.pt"))
        
        print(f"Tensors saved successfully!")
        
        # Save statistics to text file
        stats_file = os.path.join(output_dir, "dataset_stats.txt")
        print(f"Saving statistics to {stats_file}...")
        
        with open(stats_file, 'w') as f:
            f.write("Dataset Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Source file: {file_path}\n")
            f.write(f"Processing date: {pd.Timestamp.now()}\n\n")
            
            f.write("Basic Information:\n")
            f.write(f"  Number of prompts: {stats['num_prompts']:,}\n")
            f.write(f"  Total CoT sequences: {stats['total_cot_sequences']:,}\n")
            f.write(f"  Average CoTs per prompt: {stats['avg_cots_per_prompt']:.2f}\n\n")
            
            f.write("Prompt Length Statistics:\n")
            f.write(f"  Minimum: {stats['prompt_lengths']['min']}\n")
            f.write(f"  Maximum: {stats['prompt_lengths']['max']}\n")
            f.write(f"  Mean: {stats['prompt_lengths']['mean']:.2f}\n")
            f.write(f"  Median: {stats['prompt_lengths']['median']:.2f}\n")
            f.write(f"  Standard deviation: {stats['prompt_lengths']['std']:.2f}\n\n")
            
            f.write("CoT Length Statistics:\n")
            f.write(f"  Minimum: {stats['cot_lengths']['min']}\n")
            f.write(f"  Maximum: {stats['cot_lengths']['max']}\n")
            f.write(f"  Mean: {stats['cot_lengths']['mean']:.2f}\n")
            f.write(f"  Median: {stats['cot_lengths']['median']:.2f}\n")
            f.write(f"  Standard deviation: {stats['cot_lengths']['std']:.2f}\n\n")
            
            f.write("Tensor Shapes:\n")
            f.write(f"  prompt_sequences: {prompt_sequences.shape}\n")
            f.write(f"  cot_sequences_tensor: {cot_sequences_tensor.shape}\n")
            f.write(f"  prompt_mask: {prompt_mask.shape}\n")
            f.write(f"  cot_mask: {cot_mask.shape}\n\n")
            
            f.write("File Information:\n")
            f.write(f"  prompt_sequences.pt: Training-ready prompt sequences\n")
            f.write(f"  cot_sequences_tensor.pt: Training-ready CoT sequences\n")
            f.write(f"  prompt_mask.pt: Attention mask for prompts\n")
            f.write(f"  cot_mask.pt: Attention mask for CoT sequences\n")
            f.write(f"  prompts_raw.pt: Original prompt token lists\n")
            f.write(f"  cot_sequences_raw.pt: Original CoT token lists\n")
        
        print(f"Statistics saved successfully!")
        
        # Print summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        print(f"  - prompt_sequences.pt ({prompt_sequences.shape})")
        print(f"  - cot_sequences_tensor.pt ({cot_sequences_tensor.shape})")
        print(f"  - prompt_mask.pt ({prompt_mask.shape})")
        print(f"  - cot_mask.pt ({cot_mask.shape})")
        print(f"  - prompts_raw.pt")
        print(f"  - cot_sequences_raw.pt")
        print(f"  - dataset_stats.txt")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    FILE_PATH = "dart-math-uniform/data/train-00001-of-00002.parquet"

    # Run example
    save_tensors_in_usable_form(FILE_PATH)