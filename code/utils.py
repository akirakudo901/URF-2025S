# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/21

import torch
import pandas as pd
from transformers import GPT2TokenizerFast
import numpy as np
from typing import List, Tuple, Optional
import os

def tokenize_prompts_and_responses(queries: List[str], 
                                  responses_groups: List[List[str]], 
                                  tokenizer: GPT2TokenizerFast) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """
    Tokenize prompts and their corresponding response groups efficiently using batch processing.
    
    Args:
        queries (List[str]): List of prompt strings
        responses_groups (List[List[str]]): List of response groups, where each group contains responses for one prompt
        tokenizer (GPT2TokenizerFast): Tokenizer to use for encoding
    
    Returns:
        tuple: (prompts, cot_sequences) where:
            - prompts: List of tokenized prompts
            - cot_sequences: List of response groups, where each group contains tokenized responses
    """
    # Batch tokenize all prompts at once
    prompts = tokenizer(queries, 
                       add_special_tokens=True,
                       return_attention_mask=False)['input_ids']
    
    # First collect all valid responses and track indices
    all_valid_responses = []
    cots_per_prompt = []
    
    for responses in responses_groups:
        valid_responses = [r for r in responses if pd.notna(r) and r != ""]
        all_valid_responses.extend(valid_responses)
        cots_per_prompt.append(len(valid_responses))
    
    # Batch tokenize all valid responses at once
    all_cot_tokens = tokenizer(all_valid_responses,
                              add_special_tokens=True,
                              return_attention_mask=False)['input_ids']
    
    # Append EOS token to each response
    for i in range(len(all_cot_tokens)):
        all_cot_tokens[i].append(tokenizer.eos_token_id)
    
    # Regroup CoT tokens by prompt using the tracked indices
    cot_sequences = []
    start_idx = 0
    for num_cots in cots_per_prompt:
        end_idx = start_idx + num_cots
        cot_sequences.append(all_cot_tokens[start_idx:end_idx])
        start_idx = end_idx
    
    return prompts, cot_sequences

def read_parquet_to_tensors(file_path: str, 
                           tokenizer: Optional[GPT2TokenizerFast] = None,
                           max_prompts: Optional[int] = None,
                           random_seed: int = 42) -> Tuple[List[List[int]], List[List[List[int]]], str]:
    """
    Read a parquet file and convert prompts and CoTs into token tensors. Use the eos_token for padding.
    
    Args:
        file_path (str): Path to the parquet file
        tokenizer (GPT2TokenizerFast, optional): Tokenizer to use. If None, will load GPT2 tokenizer
        max_prompts (int, optional): Maximum number of prompts to process. If None, processes all
        random_seed (int): Random seed for sampling if max_prompts is specified
    
    Returns:
        tuple: (prompts, cot_sequences, eos_token) where:
            - prompts: List of P prompts, each a list of token IDs
            - cot_sequences: List of P lists, each containing M CoT sequences as token IDs
            - eos_token: The EOS token used for padding
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
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
    
    # Use the helper function for tokenization
    print("Tokenizing prompts and responses...")
    prompts, cot_sequences = tokenize_prompts_and_responses(
        grouped['query'].tolist(), 
        grouped['response'].tolist(), 
        tokenizer
    )
    
    print(f"Conversion complete!")
    print(f"Total prompts: {len(prompts):,}")
    print(f"Total CoT sequences: {sum(len(cots) for cots in cot_sequences):,}")
    
    return prompts, cot_sequences, tokenizer.eos_token


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

def process_dataset(prompts, cot_sequences, pad_token_id):
    """
    Process a dataset and create batches for training.
    
    Args:
        prompts (list): List of P prompts, where each prompt is a list of token IDs
        cot_sequences (list): List of P lists, where each inner list contains M CoT sequences
        pad_token_id (int): Token ID to use for padding
    
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
    prompt_sequences = torch.full((P, max_prompt_len), pad_token_id, dtype=torch.long)
    cot_sequences_tensor = torch.full((P, M, max_cot_len), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((P, max_prompt_len), dtype=torch.float)
    cot_mask = torch.zeros((P, M, max_cot_len), dtype=torch.float)
    
    # Process prompts
    for p_idx, prompt in enumerate(prompts):
        prompt_tensor = torch.tensor(prompt, dtype=torch.long)
        prompt_len = len(prompt)
        prompt_sequences[p_idx, :prompt_len] = prompt_tensor
        prompt_mask[p_idx, :prompt_len] = 1.0
    
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

def save_tensors_in_usable_form(file_path: str, 
                               max_prompt_length: Optional[int] = None,
                               max_cot_length: Optional[int] = None):
    """
    Example usage of the parquet reading and tensor conversion functions.
    
    Args:
        file_path (str): Path to the parquet file
        max_prompt_length (int, optional): Maximum allowed prompt length. If None, uses longest prompt
        max_cot_length (int, optional): Maximum allowed CoT length. If None, uses (1024 - max_prompt_length)
    """
    try:
        # Read and convert to tensors
        prompts, cot_sequences, eos_token = read_parquet_to_tensors(
            file_path, 
            max_prompts=None
        )
        
        # Get initial dataset statistics
        initial_stats = get_dataset_stats(prompts, cot_sequences)
        print("\nInitial Dataset Statistics:")
        for key, value in initial_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Determine max lengths
        if max_prompt_length is None:
            max_prompt_length = max(len(prompt) for prompt in prompts)
        
        if max_cot_length is None:
            max_cot_length = 1024 - max_prompt_length
        
        print(f"\nUsing max_prompt_length: {max_prompt_length}")
        print(f"Using max_cot_length: {max_cot_length}")
        
        # Filter prompts and track exclusions
        filtered_prompts = []
        filtered_cot_sequences = []
        excluded_prompts = []
        excluded_cots_info = []
        
        for i, (prompt, cots) in enumerate(zip(prompts, cot_sequences)):
            # Check if prompt exceeds max length
            if len(prompt) > max_prompt_length:
                excluded_prompts.append({
                    'index': i,
                    'prompt_length': len(prompt),
                    'max_allowed': max_prompt_length,
                    'prompt_tokens': prompt[:50] + ['...'] if len(prompt) > 50 else prompt  # First 50 tokens for reference
                })
                continue
            
            # Filter CoTs for this prompt
            valid_cots = []
            excluded_cots_for_prompt = []
            
            for j, cot in enumerate(cots):
                if len(cot) > max_cot_length:
                    excluded_cots_for_prompt.append({
                        'cot_index': j,
                        'cot_length': len(cot),
                        'max_allowed': max_cot_length,
                        'cot_tokens': cot[:50] + ['...'] if len(cot) > 50 else cot  # First 50 tokens for reference
                    })
                else:
                    valid_cots.append(cot)
            
            # Only include prompt if it has at least one valid CoT
            if valid_cots:
                filtered_prompts.append(prompt)
                filtered_cot_sequences.append(valid_cots)
                
                # Track excluded CoTs for this prompt
                if excluded_cots_for_prompt:
                    excluded_cots_info.append({
                        'prompt_index': i,
                        'prompt_length': len(prompt),
                        'excluded_cots': excluded_cots_for_prompt,
                        'remaining_cots': len(valid_cots)
                    })
        
        # Get filtered dataset statistics
        filtered_stats = get_dataset_stats(filtered_prompts, filtered_cot_sequences)
        
        print(f"\nFiltering Results:")
        print(f"  Original prompts: {len(prompts):,}")
        print(f"  Filtered prompts: {len(filtered_prompts):,}")
        print(f"  Excluded prompts: {len(excluded_prompts):,}")
        print(f"  Prompts with excluded CoTs: {len(excluded_cots_info):,}")
        
        print(f"\nFiltered Dataset Statistics:")
        for key, value in filtered_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Process into training format
        prompt_sequences, cot_sequences_tensor, prompt_mask, cot_mask = process_dataset(
            filtered_prompts, filtered_cot_sequences, pad_token_id=eos_token
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
        
        print(f"Tensors saved successfully!")
        
        # Save statistics to text file
        stats_file = os.path.join(output_dir, "dataset_stats.txt")
        print(f"Saving statistics to {stats_file}...")
        
        with open(stats_file, 'w') as f:
            f.write("Dataset Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Source file: {file_path}\n")
            f.write(f"Processing date: {pd.Timestamp.now()}\n")
            f.write(f"Max prompt length: {max_prompt_length}\n")
            f.write(f"Max CoT length: {max_cot_length}\n\n")
            
            f.write("Filtering Summary:\n")
            f.write(f"  Original prompts: {len(prompts):,}\n")
            f.write(f"  Filtered prompts: {len(filtered_prompts):,}\n")
            f.write(f"  Excluded prompts: {len(excluded_prompts):,}\n")
            f.write(f"  Prompts with excluded CoTs: {len(excluded_cots_info):,}\n\n")
            
            f.write("Basic Information:\n")
            f.write(f"  Number of prompts: {filtered_stats['num_prompts']:,}\n")
            f.write(f"  Total CoT sequences: {filtered_stats['total_cot_sequences']:,}\n")
            f.write(f"  Average CoTs per prompt: {filtered_stats['avg_cots_per_prompt']:.2f}\n\n")
            
            f.write("Prompt Length Statistics:\n")
            f.write(f"  Minimum: {filtered_stats['prompt_lengths']['min']}\n")
            f.write(f"  Maximum: {filtered_stats['prompt_lengths']['max']}\n")
            f.write(f"  Mean: {filtered_stats['prompt_lengths']['mean']:.2f}\n")
            f.write(f"  Median: {filtered_stats['prompt_lengths']['median']:.2f}\n")
            f.write(f"  Standard deviation: {filtered_stats['prompt_lengths']['std']:.2f}\n\n")
            
            f.write("CoT Length Statistics:\n")
            f.write(f"  Minimum: {filtered_stats['cot_lengths']['min']}\n")
            f.write(f"  Maximum: {filtered_stats['cot_lengths']['max']}\n")
            f.write(f"  Mean: {filtered_stats['cot_lengths']['mean']:.2f}\n")
            f.write(f"  Median: {filtered_stats['cot_lengths']['median']:.2f}\n")
            f.write(f"  Standard deviation: {filtered_stats['cot_lengths']['std']:.2f}\n\n")
            
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
            f.write(f"  excluded_data.json: Details of excluded prompts and CoTs\n")
        
        # Save excluded data details
        excluded_data = {
            'filtering_parameters': {
                'max_prompt_length': max_prompt_length,
                'max_cot_length': max_cot_length
            },
            'summary': {
                'original_prompts': len(prompts),
                'filtered_prompts': len(filtered_prompts),
                'excluded_prompts': len(excluded_prompts),
                'prompts_with_excluded_cots': len(excluded_cots_info)
            },
            'excluded_prompts': excluded_prompts,
            'excluded_cots_info': excluded_cots_info
        }
        
        import json
        excluded_file = os.path.join(output_dir, "excluded_data.json")
        print(f"Saving excluded data details to {excluded_file}...")
        
        # Convert token lists to strings for JSON serialization
        json_excluded_data = {
            'filtering_parameters': excluded_data['filtering_parameters'],
            'summary': excluded_data['summary'],
            'excluded_prompts': [
                {
                    'index': item['index'],
                    'prompt_length': item['prompt_length'],
                    'max_allowed': item['max_allowed'],
                    'prompt_tokens': [str(token) for token in item['prompt_tokens']]
                }
                for item in excluded_data['excluded_prompts']
            ],
            'excluded_cots_info': [
                {
                    'prompt_index': item['prompt_index'],
                    'prompt_length': item['prompt_length'],
                    'remaining_cots': item['remaining_cots'],
                    'excluded_cots': [
                        {
                            'cot_index': cot['cot_index'],
                            'cot_length': cot['cot_length'],
                            'max_allowed': cot['max_allowed'],
                            'cot_tokens': [str(token) for token in cot['cot_tokens']]
                        }
                        for cot in item['excluded_cots']
                    ]
                }
                for item in excluded_data['excluded_cots_info']
            ]
        }
        
        with open(excluded_file, 'w') as f:
            json.dump(json_excluded_data, f, indent=2)
        
        print(f"Statistics and excluded data saved successfully!")
        
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
        print(f"  - excluded_data.json")
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def analyze_length_distribution(file_path: str, 
                              tokenizer: Optional[GPT2TokenizerFast] = None,
                              max_prompts: Optional[int] = None,
                              random_seed: int = 42,
                              save_results: bool = True) -> dict:
    """
    Analyze the distribution of prompt and CoT lengths relative to powers of 2.
    
    Args:
        file_path (str): Path to the parquet file
        tokenizer (GPT2TokenizerFast, optional): Tokenizer to use. If None, will load GPT2 tokenizer
        max_prompts (int, optional): Maximum number of prompts to process. If None, processes all
        random_seed (int): Random seed for sampling if max_prompts is specified
        save_results (bool): Whether to save results to data/GSM8K directory
    
    Returns:
        dict: Analysis results with percentages for different length thresholds
    """
    # Use read_parquet_to_tensors to get tokenized data
    print(f"Reading and tokenizing data from: {file_path}")
    prompts, cot_sequences, eos_token = read_parquet_to_tensors(
        file_path, 
        tokenizer=tokenizer,
        max_prompts=max_prompts,
        random_seed=random_seed
    )
    
    # Calculate lengths from tokenized data
    print("Calculating token lengths...")
    prompt_lengths = [len(prompt) for prompt in prompts]
    cot_lengths = [len(cot) for cot_group in cot_sequences for cot in cot_group]
    
    print(f"Length calculation complete!")
    print(f"Total prompts analyzed: {len(prompt_lengths):,}")
    print(f"Total CoT sequences analyzed: {len(cot_lengths):,}")
    
    # Define powers of 2 thresholds (>= 64)
    thresholds = [2**i for i in range(6, 16)]  # 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    
    # Calculate percentages for prompts
    prompt_analysis = {}
    for threshold in thresholds:
        count = sum(1 for length in prompt_lengths if length > threshold)
        percentage = (count / len(prompt_lengths)) * 100 if prompt_lengths else 0
        prompt_analysis[f"prompts_longer_than_{threshold}"] = {
            'count': count,
            'percentage': percentage,
            'total': len(prompt_lengths)
        }
    
    # Calculate percentages for CoTs
    cot_analysis = {}
    for threshold in thresholds:
        count = sum(1 for length in cot_lengths if length > threshold)
        percentage = (count / len(cot_lengths)) * 100 if cot_lengths else 0
        cot_analysis[f"cots_longer_than_{threshold}"] = {
            'count': count,
            'percentage': percentage,
            'total': len(cot_lengths)
        }
    
    # Print results
    print(f"\n{'='*80}")
    print("LENGTH DISTRIBUTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Source file: {file_path}")
    print(f"Analysis date: {pd.Timestamp.now()}")
    print(f"\nPROMPT LENGTH ANALYSIS:")
    print(f"{'Threshold':<12} {'Count':<8} {'Percentage':<12} {'Total':<8}")
    print(f"{'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    
    for threshold in thresholds:
        key = f"prompts_longer_than_{threshold}"
        data = prompt_analysis[key]
        print(f"{threshold:<12} {data['count']:<8} {data['percentage']:<12.2f}% {data['total']:<8}")
    
    print(f"\nCOT LENGTH ANALYSIS:")
    print(f"{'Threshold':<12} {'Count':<8} {'Percentage':<12} {'Total':<8}")
    print(f"{'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    
    for threshold in thresholds:
        key = f"cots_longer_than_{threshold}"
        data = cot_analysis[key]
        print(f"{threshold:<12} {data['count']:<8} {data['percentage']:<12.2f}% {data['total']:<8}")
    
    # Additional statistics
    print(f"\nADDITIONAL STATISTICS:")
    print(f"Prompt lengths - Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, Mean: {np.mean(prompt_lengths):.1f}, Median: {np.median(prompt_lengths):.1f}")
    print(f"CoT lengths - Min: {min(cot_lengths)}, Max: {max(cot_lengths)}, Mean: {np.mean(cot_lengths):.1f}, Median: {np.median(cot_lengths):.1f}")
    
    # Return combined results
    results = {
        'file_path': file_path,
        'analysis_date': str(pd.Timestamp.now()),
        'prompt_analysis': prompt_analysis,
        'cot_analysis': cot_analysis,
        'summary_stats': {
            'prompts': {
                'total': len(prompt_lengths),
                'min': min(prompt_lengths),
                'max': max(prompt_lengths),
                'mean': np.mean(prompt_lengths),
                'median': np.median(prompt_lengths)
            },
            'cots': {
                'total': len(cot_lengths),
                'min': min(cot_lengths),
                'max': max(cot_lengths),
                'mean': np.mean(cot_lengths),
                'median': np.median(cot_lengths)
            }
        }
    }
    
    # Save results if requested
    if save_results:
        # Create output directory
        output_dir = "data/GSM8K"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        import json
        json_file = os.path.join(output_dir, "length_distribution_analysis.json")
        print(f"\nSaving JSON results to {json_file}...")
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = {
            'file_path': results['file_path'],
            'analysis_date': results['analysis_date'],
            'prompt_analysis': results['prompt_analysis'],
            'cot_analysis': results['cot_analysis'],
            'summary_stats': {
                'prompts': {
                    'total': int(results['summary_stats']['prompts']['total']),
                    'min': int(results['summary_stats']['prompts']['min']),
                    'max': int(results['summary_stats']['prompts']['max']),
                    'mean': float(results['summary_stats']['prompts']['mean']),
                    'median': float(results['summary_stats']['prompts']['median'])
                },
                'cots': {
                    'total': int(results['summary_stats']['cots']['total']),
                    'min': int(results['summary_stats']['cots']['min']),
                    'max': int(results['summary_stats']['cots']['max']),
                    'mean': float(results['summary_stats']['cots']['mean']),
                    'median': float(results['summary_stats']['cots']['median'])
                }
            }
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save formatted text report
        text_file = os.path.join(output_dir, "length_distribution_report.txt")
        print(f"Saving text report to {text_file}...")
        
        with open(text_file, 'w') as f:
            f.write("LENGTH DISTRIBUTION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Source file: {file_path}\n")
            f.write(f"Analysis date: {pd.Timestamp.now()}\n")
            f.write(f"Total prompts analyzed: {len(prompt_lengths):,}\n")
            f.write(f"Total CoT sequences analyzed: {len(cot_lengths):,}\n\n")
            
            f.write("PROMPT LENGTH ANALYSIS:\n")
            f.write(f"{'Threshold':<12} {'Count':<8} {'Percentage':<12} {'Total':<8}\n")
            f.write(f"{'-'*12} {'-'*8} {'-'*12} {'-'*8}\n")
            
            for threshold in thresholds:
                key = f"prompts_longer_than_{threshold}"
                data = prompt_analysis[key]
                f.write(f"{threshold:<12} {data['count']:<8} {data['percentage']:<12.2f}% {data['total']:<8}\n")
            
            f.write(f"\nCOT LENGTH ANALYSIS:\n")
            f.write(f"{'Threshold':<12} {'Count':<8} {'Percentage':<12} {'Total':<8}\n")
            f.write(f"{'-'*12} {'-'*8} {'-'*12} {'-'*8}\n")
            
            for threshold in thresholds:
                key = f"cots_longer_than_{threshold}"
                data = cot_analysis[key]
                f.write(f"{threshold:<12} {data['count']:<8} {data['percentage']:<12.2f}% {data['total']:<8}\n")
            
            f.write(f"\nADDITIONAL STATISTICS:\n")
            f.write(f"Prompt lengths - Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, Mean: {np.mean(prompt_lengths):.1f}, Median: {np.median(prompt_lengths):.1f}\n")
            f.write(f"CoT lengths - Min: {min(cot_lengths)}, Max: {max(cot_lengths)}, Mean: {np.mean(cot_lengths):.1f}, Median: {np.median(cot_lengths):.1f}\n\n")
            
            f.write("ANALYSIS NOTES:\n")
            f.write("- This analysis shows the distribution of token lengths relative to powers of 2\n")
            f.write("- Thresholds start at 64 (2^6) and go up to 32768 (2^15)\n")
            f.write("- Percentages indicate what portion of sequences exceed each threshold\n")
            f.write("- This information is useful for determining model architecture and padding strategies\n")
        
        print(f"Results saved successfully!")
        print(f"Files created:")
        print(f"  - {json_file}")
        print(f"  - {text_file}")
    
    return results

if __name__ == "__main__":
    # GSM8K dataset
    FILE_PATH = "dart-math-uniform/data/train-00001-of-00002.parquet"

    save_tensors_in_usable_form(FILE_PATH)

    # analyze_length_distribution(file_path=FILE_PATH, 
    #                             save_results=True)