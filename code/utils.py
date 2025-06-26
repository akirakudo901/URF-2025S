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
    
    return prompts, cot_sequences, tokenizer.eos_token_id


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

def process_dataset(prompts, cot_sequences, pad_token_id, cots_per_batch: Optional[int] = None):
    """
    Process a dataset and create batches for training.
    
    Args:
        prompts (list): List of P prompts, where each prompt is a list of token IDs
        cot_sequences (list): List of P lists, where each inner list contains M CoT sequences
        pad_token_id (int): Token ID to use for padding
        cots_per_batch (int, optional): Number of CoTs to aggregate together per prompt. 
                                       If None, uses all available CoTs. If specified, creates
                                       multiple batches per prompt and discards incomplete batches.
    
    Returns:
        tuple: (prompt_sequences, cot_sequences, prompt_mask, cot_mask)
    """
    P = len(prompts)
    
    # Validate input
    if len(cot_sequences) != P:
        raise ValueError(f"Number of prompt groups ({P}) must match number of CoT groups ({len(cot_sequences)})")
    
    if cots_per_batch is not None and cots_per_batch <= 0:
        raise ValueError(f"cots_per_batch must be positive, got {cots_per_batch}")
    
    # Process data with batching if specified
    if cots_per_batch is not None:
        return _process_dataset_with_batching(prompts, cot_sequences, pad_token_id, cots_per_batch)
    else:
        return _process_dataset_standard(prompts, cot_sequences, pad_token_id)

def _process_dataset_standard(prompts, cot_sequences, pad_token_id):
    """
    Process dataset using the original method (all CoTs per prompt).
    Due to the nature of padding, could be quite wasteful in space.
    """
    P = len(prompts)
    
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

def _process_dataset_with_batching(prompts, cot_sequences, pad_token_id, cots_per_batch):
    """
    Process dataset with batching - create multiple batches per prompt.
    """
    # Create batched data
    batched_prompts = []
    batched_cot_sequences = []
    
    for prompt, cots in zip(prompts, cot_sequences):
        # Calculate number of complete batches we can make
        num_complete_batches = len(cots) // cots_per_batch
        
        # Create complete batches
        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * cots_per_batch
            end_idx = start_idx + cots_per_batch
            batch_cots = cots[start_idx:end_idx]
            
            batched_prompts.append(prompt)
            batched_cot_sequences.append(batch_cots)
    
    # Now process the batched data using standard method
    return _process_dataset_standard(batched_prompts, batched_cot_sequences, pad_token_id)

def save_tensors_in_usable_form(file_path: str, 
                               output_dir: str,
                               max_prompt_length: Optional[int] = None,
                               max_cot_length: Optional[int] = None,
                               min_qualifying_cots: int = 8):
    """
    Example usage of the parquet reading and tensor conversion functions.
    
    Args:
        file_path (str): Path to the parquet file
        output_dir (str): Path to save generated dataset, created if non-existent
        max_prompt_length (int, optional): Maximum allowed prompt length. If None, uses longest prompt
        max_cot_length (int, optional): Maximum allowed CoT length. If None, uses (1024 - max_prompt_length)
        min_qualifying_cots (int): Minimum number of qualifying CoTs required per prompt. Defaults to 8
    """
    try:
        # Read and convert to tensors
        prompts, cot_sequences, eos_token_id = read_parquet_to_tensors(
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
        print(f"Using min_qualifying_cots: {min_qualifying_cots}")
        
        # Filter prompts and track exclusions
        filtered_prompts = []
        filtered_cot_sequences = []
        excluded_prompts = []
        excluded_cots_info = []
        excluded_insufficient_cots = []
        
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
            
            # Check if we have enough qualifying CoTs
            if len(valid_cots) < min_qualifying_cots:
                excluded_insufficient_cots.append({
                    'index': i,
                    'prompt_length': len(prompt),
                    'valid_cots_count': len(valid_cots),
                    'min_required': min_qualifying_cots,
                    'prompt_tokens': prompt[:50] + ['...'] if len(prompt) > 50 else prompt  # First 50 tokens for reference
                })
                continue
            
            # Include prompt if it has enough valid CoTs
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
        print(f"  Excluded prompts (length): {len(excluded_prompts):,}")
        print(f"  Excluded prompts (insufficient CoTs): {len(excluded_insufficient_cots):,}")
        print(f"  Prompts with excluded CoTs: {len(excluded_cots_info):,}")
        
        print(f"\nFiltered Dataset Statistics:")
        for key, value in filtered_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Define different batch sizes to create
        cots_per_batch_values = [1, 2, 4, 8, 16]
        
        print(f"\nCreating datasets with different batch sizes: {cots_per_batch_values}")
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process and save datasets for each batch size
        batch_results = {}
        
        for cots_per_batch in cots_per_batch_values:
            print(f"\n{'='*60}")
            print(f"PROCESSING: cots_per_batch = {cots_per_batch}")
            print(f"{'='*60}")
            
            # Create subdirectory for this batch size
            batch_dir = os.path.join(output_dir, f"batch_{cots_per_batch}")
            os.makedirs(batch_dir, exist_ok=True)
            
            # Process into training format with batching
            prompt_sequences, cot_sequences_tensor, prompt_mask, cot_mask = process_dataset(
                filtered_prompts, filtered_cot_sequences, pad_token_id=eos_token_id, cots_per_batch=cots_per_batch
            )
            
            print(f"Training tensors shape (batch_size={cots_per_batch}):")
            print(f"  prompt_sequences: {prompt_sequences.shape}")
            print(f"  cot_sequences_tensor: {cot_sequences_tensor.shape}")
            print(f"  prompt_mask: {prompt_mask.shape}")
            print(f"  cot_mask: {cot_mask.shape}")
            
            # Save tensors
            print(f"Saving tensors to {batch_dir}...")
            torch.save(prompt_sequences, os.path.join(batch_dir, "prompt_sequences.pt"))
            torch.save(cot_sequences_tensor, os.path.join(batch_dir, "cot_sequences_tensor.pt"))
            torch.save(prompt_mask, os.path.join(batch_dir, "prompt_mask.pt"))
            torch.save(cot_mask, os.path.join(batch_dir, "cot_mask.pt"))
            
            # Calculate batch statistics
            num_batches = prompt_sequences.shape[0]
            total_cot_sequences = cot_sequences_tensor.shape[0] * cot_sequences_tensor.shape[1]
            
            batch_stats = {
                'cots_per_batch': cots_per_batch,
                'num_batches': num_batches,
                'total_cot_sequences': total_cot_sequences,
                'tensor_shapes': {
                    'prompt_sequences': prompt_sequences.shape,
                    'cot_sequences_tensor': cot_sequences_tensor.shape,
                    'prompt_mask': prompt_mask.shape,
                    'cot_mask': cot_mask.shape
                }
            }
            
            batch_results[f"batch_{cots_per_batch}"] = batch_stats
            
            # Save batch-specific statistics
            batch_stats_file = os.path.join(batch_dir, "batch_stats.json")
            import json
            with open(batch_stats_file, 'w') as f:
                json.dump(batch_stats, f, indent=2)
            
            # Calculate padding ratio for this batch
            print(f"Calculating padding ratio for batch {cots_per_batch}...")
            try:
                padding_results = calculate_padding_ratio(batch_dir)
                batch_stats['padding_ratio'] = padding_results['total_padding_ratio']
                batch_stats['prompt_padding_ratio'] = padding_results['prompt_padding_ratio']
                batch_stats['cot_padding_ratio'] = padding_results['cot_padding_ratio']
                
                # Update the batch stats file with padding information
                with open(batch_stats_file, 'w') as f:
                    json.dump(batch_stats, f, indent=2)
                
                print(f"Batch {cots_per_batch} padding ratio: {padding_results['total_padding_ratio']:.4f} ({padding_results['total_padding_ratio']*100:.2f}%)")
            except Exception as e:
                print(f"Warning: Could not calculate padding ratio for batch {cots_per_batch}: {e}")
                batch_stats['padding_ratio'] = None
                batch_stats['prompt_padding_ratio'] = None
                batch_stats['cot_padding_ratio'] = None
            
            print(f"Batch {cots_per_batch} saved successfully!")
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"{'Batch Size':<12} {'Num Batches':<12} {'Total CoTs':<12} {'Total Padding':<15} {'Prompt Shape':<20} {'CoT Shape':<20}")
        print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*15} {'-'*20} {'-'*20}")
        
        for batch_key, stats in batch_results.items():
            prompt_shape = str(stats['tensor_shapes']['prompt_sequences'])
            cot_shape = str(stats['tensor_shapes']['cot_sequences_tensor'])
            padding_ratio = stats.get('padding_ratio', 'N/A')
            if padding_ratio is not None:
                padding_str = f"{padding_ratio:.4f}"
            else:
                padding_str = "N/A"
            print(f"{stats['cots_per_batch']:<12} {stats['num_batches']:<12} {stats['total_cot_sequences']:<12} {padding_str:<15} {prompt_shape:<20} {cot_shape:<20}")
        
        # Save overall statistics to main directory
        stats_file = os.path.join(output_dir, "dataset_stats.txt")
        print(f"\nSaving overall statistics to {stats_file}...")
        
        with open(stats_file, 'w') as f:
            f.write("Dataset Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Source file: {file_path}\n")
            f.write(f"Processing date: {pd.Timestamp.now()}\n")
            f.write(f"Max prompt length: {max_prompt_length}\n")
            f.write(f"Max CoT length: {max_cot_length}\n")
            f.write(f"Min qualifying CoTs: {min_qualifying_cots}\n")
            f.write(f"Batch sizes created: {cots_per_batch_values}\n\n")
            
            f.write("Filtering Summary:\n")
            f.write(f"  Original prompts: {len(prompts):,}\n")
            f.write(f"  Filtered prompts: {len(filtered_prompts):,}\n")
            f.write(f"  Excluded prompts (length): {len(excluded_prompts):,}\n")
            f.write(f"  Excluded prompts (insufficient CoTs): {len(excluded_insufficient_cots):,}\n")
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
            
            f.write("Batch Processing Results:\n")
            for batch_key, stats in batch_results.items():
                f.write(f"  {batch_key}:\n")
                f.write(f"    Number of batches: {stats['num_batches']:,}\n")
                f.write(f"    Total CoT sequences: {stats['total_cot_sequences']:,}\n")
                f.write(f"    Prompt tensor shape: {stats['tensor_shapes']['prompt_sequences']}\n")
                f.write(f"    CoT tensor shape: {stats['tensor_shapes']['cot_sequences_tensor']}\n")
                if stats.get('padding_ratio') is not None:
                    f.write(f"    Total padding ratio: {stats['padding_ratio']:.4f} ({stats['padding_ratio']*100:.2f}%)\n")
                    f.write(f"    Prompt padding ratio: {stats['prompt_padding_ratio']:.4f} ({stats['prompt_padding_ratio']*100:.2f}%)\n")
                    f.write(f"    CoT padding ratio: {stats['cot_padding_ratio']:.4f} ({stats['cot_padding_ratio']*100:.2f}%)\n")
                else:
                    f.write(f"    Padding ratio: Not calculated\n")
                f.write(f"\n")
            
            f.write("Directory Structure:\n")
            f.write(f"  {output_dir}/\n")
            f.write(f"    dataset_stats.txt: This file\n")
            f.write(f"    excluded_data.json: Details of excluded prompts and CoTs\n")
            for batch_size in cots_per_batch_values:
                f.write(f"    batch_{batch_size}/\n")
                f.write(f"      prompt_sequences.pt: Training-ready prompt sequences\n")
                f.write(f"      cot_sequences_tensor.pt: Training-ready CoT sequences\n")
                f.write(f"      prompt_mask.pt: Attention mask for prompts\n")
                f.write(f"      cot_mask.pt: Attention mask for CoT sequences\n")
                f.write(f"      batch_stats.json: Batch-specific statistics\n")
        
        # Save excluded data details
        excluded_data = {
            'filtering_parameters': {
                'max_prompt_length': max_prompt_length,
                'max_cot_length': max_cot_length,
                'min_qualifying_cots': min_qualifying_cots,
                'batch_sizes_created': cots_per_batch_values
            },
            'summary': {
                'original_prompts': len(prompts),
                'filtered_prompts': len(filtered_prompts),
                'excluded_prompts_length': len(excluded_prompts),
                'excluded_prompts_insufficient_cots': len(excluded_insufficient_cots),
                'prompts_with_excluded_cots': len(excluded_cots_info)
            },
            'batch_results': batch_results,
            'excluded_prompts': excluded_prompts,
            'excluded_insufficient_cots': excluded_insufficient_cots,
            'excluded_cots_info': excluded_cots_info
        }
        
        import json
        excluded_file = os.path.join(output_dir, "excluded_data.json")
        print(f"Saving excluded data details to {excluded_file}...")
        
        # Convert token lists to strings for JSON serialization
        json_excluded_data = {
            'filtering_parameters': excluded_data['filtering_parameters'],
            'summary': excluded_data['summary'],
            'batch_results': batch_results,
            'excluded_prompts': [
                {
                    'index': item['index'],
                    'prompt_length': item['prompt_length'],
                    'max_allowed': item['max_allowed'],
                    'prompt_tokens': [str(token) for token in item['prompt_tokens']]
                }
                for item in excluded_data['excluded_prompts']
            ],
            'excluded_insufficient_cots': [
                {
                    'index': item['index'],
                    'prompt_length': item['prompt_length'],
                    'valid_cots_count': item['valid_cots_count'],
                    'min_required': item['min_required'],
                    'prompt_tokens': [str(token) for token in item['prompt_tokens']]
                }
                for item in excluded_data['excluded_insufficient_cots']
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
        print(f"Batch sizes created: {cots_per_batch_values}")
        print(f"Files created:")
        print(f"  - dataset_stats.txt")
        print(f"  - excluded_data.json")
        for batch_size in cots_per_batch_values:
            print(f"  - batch_{batch_size}/ (complete dataset)")
        
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
    prompts, cot_sequences, _ = read_parquet_to_tensors(
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

def analyze_and_generate_datasets(file_path: str,
                                 min_qualifying_cots: int = 8,
                                 tokenizer: Optional[GPT2TokenizerFast] = None,
                                 max_prompts: Optional[int] = None,
                                 random_seed: int = 42,
                                 save_results: bool = True) -> dict:
    """
    Analyze different prompt/CoT length combinations and generate datasets.
    
    This function:
    1. Considers possible prompt lengths as multiples of 2 (64, 128, 256, etc.)
    2. Considers possible CoT lengths as 1x or 3x times the prompt length
    3. For each combination, counts qualifying prompts and CoTs
    4. Excludes prompts that don't meet the minimum qualifying CoT threshold
    5. Calculates padding statistics and generates datasets
    
    Args:
        file_path (str): Path to the parquet file
        min_qualifying_cots (int): Minimum number of qualifying CoTs required per prompt
        tokenizer (GPT2TokenizerFast, optional): Tokenizer to use. If None, will load GPT2 tokenizer
        max_prompts (int, optional): Maximum number of prompts to process. If None, processes all
        random_seed (int): Random seed for sampling if max_prompts is specified
        save_results (bool): Whether to save results and datasets to data/GSM8K directory
    
    Returns:
        dict: Analysis results with statistics for each length combination
    """
    # Use read_parquet_to_tensors to get tokenized data
    print(f"Reading and tokenizing data from: {file_path}")
    prompts, cot_sequences, eos_token_id = read_parquet_to_tensors(
        file_path, 
        tokenizer=tokenizer,
        max_prompts=max_prompts,
        random_seed=random_seed
    )
    
    print(f"Analysis complete!")
    print(f"Total prompts: {len(prompts):,}")
    print(f"Total CoT sequences: {sum(len(cots) for cots in cot_sequences):,}")
    
    # Define prompt lengths as powers of 2 (starting from 64)
    prompt_lengths = [2**i for i in range(6, 9)]  # 64, 128, 256
    
    # Calculate CoT lengths as 1x and 3x of prompt lengths
    cot_lengths_1x = [prompt_len for prompt_len in prompt_lengths]  # 1x: 64, 128, 256
    cot_lengths_3x = [prompt_len * 3 for prompt_len in prompt_lengths]  # 3x: 192, 384, 768
    
    # Create all combinations (prompt_length, cot_length)
    combinations = []
    for prompt_len in prompt_lengths:
        combinations.append((prompt_len, prompt_len))  # 1x
        combinations.append((prompt_len, prompt_len * 3))  # 3x
    
    # Store results for each combination
    results = {
        'file_path': file_path,
        'analysis_date': str(pd.Timestamp.now()),
        'min_qualifying_cots': min_qualifying_cots,
        'combinations': {},
        'summary': {
            'total_original_prompts': len(prompts),
            'total_original_cots': sum(len(cots) for cots in cot_sequences),
            'prompt_lengths_tested': prompt_lengths,
            'cot_lengths_1x': cot_lengths_1x,
            'cot_lengths_3x': cot_lengths_3x,
            'total_combinations': len(combinations)
        }
    }
    
    print(f"\n{'='*100}")
    print("PROMPT/COT LENGTH COMBINATION ANALYSIS")
    print(f"{'='*100}")
    print(f"Source file: {file_path}")
    print(f"Analysis date: {pd.Timestamp.now()}")
    print(f"Minimum qualifying CoTs per prompt: {min_qualifying_cots}")
    print(f"Prompt lengths to test: {prompt_lengths}")
    print(f"CoT lengths (1x): {cot_lengths_1x}")
    print(f"CoT lengths (3x): {cot_lengths_3x}")
    print(f"Total combinations to test: {len(combinations)}")
    
    # Analyze each combination
    for prompt_len, cot_len in combinations:
        print(f"\n{'='*60}")
        print(f"ANALYZING: Prompt Length = {prompt_len}, CoT Length = {cot_len} ({cot_len//prompt_len}x)")
        print(f"{'='*60}")
        
        # Find qualifying prompts (prompts that fit within prompt_len)
        qualifying_prompts = []
        qualifying_cot_sequences = []
        prompt_indices = []
        
        for i, (prompt, cots) in enumerate(zip(prompts, cot_sequences)):
            if len(prompt) <= prompt_len:
                # Count qualifying CoTs for this prompt
                qualifying_cots = [cot for cot in cots if len(cot) <= cot_len]
                
                # Only include if we have enough qualifying CoTs
                if len(qualifying_cots) >= min_qualifying_cots:
                    qualifying_prompts.append(prompt)
                    qualifying_cot_sequences.append(qualifying_cots)
                    prompt_indices.append(i)
        
        # Calculate statistics
        total_qualifying_prompts = len(qualifying_prompts)
        total_qualifying_cots = sum(len(cots) for cots in qualifying_cot_sequences)
        
        # Calculate padding statistics
        total_prompt_padding = 0
        total_cot_padding = 0
        total_prompt_tokens = 0
        total_cot_tokens = 0
        
        # Find actual maximum lengths of qualifying data
        max_actual_prompt_len = max(len(prompt) for prompt in qualifying_prompts) if qualifying_prompts else 0
        max_actual_cot_len = max(len(cot) for cot_group in qualifying_cot_sequences for cot in cot_group) if qualifying_cot_sequences else 0
        
        for prompt in qualifying_prompts:
            prompt_padding = max_actual_prompt_len - len(prompt)
            total_prompt_padding += prompt_padding
            total_prompt_tokens += max_actual_prompt_len
        
        for cot_group in qualifying_cot_sequences:
            for cot in cot_group:
                cot_padding = max_actual_cot_len - len(cot)
                total_cot_padding += cot_padding
                total_cot_tokens += max_actual_cot_len
        
        # Calculate percentages
        prompt_padding_percentage = (total_prompt_padding / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0
        cot_padding_percentage = (total_cot_padding / total_cot_tokens * 100) if total_cot_tokens > 0 else 0
        total_padding_percentage = ((total_prompt_padding + total_cot_padding) / (total_prompt_tokens + total_cot_tokens) * 100) if (total_prompt_tokens + total_cot_tokens) > 0 else 0
        
        # Store results
        multiplier = "1x" if cot_len == prompt_len else "3x"
        combination_key = f"prompt_{prompt_len}_cot_{cot_len}_{multiplier}"
        results['combinations'][combination_key] = {
            'prompt_length': prompt_len,
            'cot_length': cot_len,
            'cot_multiplier': multiplier,
            'qualifying_prompts': total_qualifying_prompts,
            'qualifying_cots': total_qualifying_cots,
            'avg_cots_per_prompt': total_qualifying_cots / total_qualifying_prompts if total_qualifying_prompts > 0 else 0,
            'padding_stats': {
                'prompt_padding_tokens': total_prompt_padding,
                'cot_padding_tokens': total_cot_padding,
                'total_padding_tokens': total_prompt_padding + total_cot_padding,
                'prompt_padding_percentage': prompt_padding_percentage,
                'cot_padding_percentage': cot_padding_percentage,
                'total_padding_percentage': total_padding_percentage,
                'total_tokens': total_prompt_tokens + total_cot_tokens
            },
            'prompt_indices': prompt_indices
        }
        
        # Print results
        print(f"Qualifying prompts: {total_qualifying_prompts:,}")
        print(f"Qualifying CoTs: {total_qualifying_cots:,}")
        print(f"Average CoTs per prompt: {total_qualifying_cots / total_qualifying_prompts:.2f}" if total_qualifying_prompts > 0 else "N/A")
        print(f"Prompt padding: {total_prompt_padding:,} tokens ({prompt_padding_percentage:.2f}%)")
        print(f"CoT padding: {total_cot_padding:,} tokens ({cot_padding_percentage:.2f}%)")
        print(f"Total padding: {total_prompt_padding + total_cot_padding:,} tokens ({total_padding_percentage:.2f}%)")
        
        # Generate and save dataset if requested
        if save_results and total_qualifying_prompts > 0:
            print(f"Generating dataset for this combination...")
            
            # Process into training format
            prompt_sequences, cot_sequences_tensor, prompt_mask, cot_mask = process_dataset(
                qualifying_prompts, qualifying_cot_sequences, pad_token_id=eos_token_id
            )
            
            # Create output directory
            output_dir = f"data/GSM8K/prompt_{prompt_len}_cot_{cot_len}_{multiplier}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save tensors
            torch.save(prompt_sequences, os.path.join(output_dir, "prompt_sequences.pt"))
            torch.save(cot_sequences_tensor, os.path.join(output_dir, "cot_sequences_tensor.pt"))
            torch.save(prompt_mask, os.path.join(output_dir, "prompt_mask.pt"))
            torch.save(cot_mask, os.path.join(output_dir, "cot_mask.pt"))
            
            # Save raw data for reference
            torch.save(qualifying_prompts, os.path.join(output_dir, "prompts_raw.pt"))
            torch.save(qualifying_cot_sequences, os.path.join(output_dir, "cot_sequences_raw.pt"))
            
            # Save statistics
            stats = {
                'combination': combination_key,
                'prompt_length': prompt_len,
                'cot_length': cot_len,
                'cot_multiplier': multiplier,
                'min_qualifying_cots': min_qualifying_cots,
                'qualifying_prompts': total_qualifying_prompts,
                'qualifying_cots': total_qualifying_cots,
                'avg_cots_per_prompt': total_qualifying_cots / total_qualifying_prompts if total_qualifying_prompts > 0 else 0,
                'padding_stats': results['combinations'][combination_key]['padding_stats'],
                'tensor_shapes': {
                    'prompt_sequences': prompt_sequences.shape,
                    'cot_sequences_tensor': cot_sequences_tensor.shape,
                    'prompt_mask': prompt_mask.shape,
                    'cot_mask': cot_mask.shape
                }
            }
            
            import json
            with open(os.path.join(output_dir, "dataset_stats.json"), 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Dataset saved to: {output_dir}")
    
    # Print summary table
    print(f"\n{'='*140}")
    print("SUMMARY TABLE")
    print(f"{'='*140}")
    print(f"{'Prompt':<8} {'CoT':<8} {'Multiplier':<10} {'Qualifying':<12} {'Qualifying':<12} {'Avg CoTs':<10} {'Total Padding':<15} {'Padding %':<12}")
    print(f"{'Length':<8} {'Length':<8} {'':<10} {'Prompts':<12} {'CoTs':<12} {'per Prompt':<10} {'(tokens)':<15} {'':<12}")
    print(f"{'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*15} {'-'*12}")
    
    for prompt_len, cot_len in combinations:
        multiplier = "1x" if cot_len == prompt_len else "3x"
        combination_key = f"prompt_{prompt_len}_cot_{cot_len}_{multiplier}"
        data = results['combinations'][combination_key]
        
        print(f"{prompt_len:<8} {cot_len:<8} {multiplier:<10} {data['qualifying_prompts']:<12} {data['qualifying_cots']:<12} "
              f"{data['avg_cots_per_prompt']:<10.2f} {data['padding_stats']['total_padding_tokens']:<15} "
              f"{data['padding_stats']['total_padding_percentage']:<12.2f}%")
    
    # Save overall results if requested
    if save_results:
        # Create output directory
        output_dir = "data/GSM8K"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        import json
        json_file = os.path.join(output_dir, "length_combination_analysis.json")
        print(f"\nSaving overall results to {json_file}...")
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = {
            'file_path': results['file_path'],
            'analysis_date': results['analysis_date'],
            'min_qualifying_cots': results['min_qualifying_cots'],
            'summary': results['summary'],
            'combinations': {}
        }
        
        for key, data in results['combinations'].items():
            json_results['combinations'][key] = {
                'prompt_length': int(data['prompt_length']),
                'cot_length': int(data['cot_length']),
                'cot_multiplier': data['cot_multiplier'],
                'qualifying_prompts': int(data['qualifying_prompts']),
                'qualifying_cots': int(data['qualifying_cots']),
                'avg_cots_per_prompt': float(data['avg_cots_per_prompt']),
                'padding_stats': {
                    'prompt_padding_tokens': int(data['padding_stats']['prompt_padding_tokens']),
                    'cot_padding_tokens': int(data['padding_stats']['cot_padding_tokens']),
                    'total_padding_tokens': int(data['padding_stats']['total_padding_tokens']),
                    'prompt_padding_percentage': float(data['padding_stats']['prompt_padding_percentage']),
                    'cot_padding_percentage': float(data['padding_stats']['cot_padding_percentage']),
                    'total_padding_percentage': float(data['padding_stats']['total_padding_percentage']),
                    'total_tokens': int(data['padding_stats']['total_tokens'])
                },
                'prompt_indices': data['prompt_indices']
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save formatted text report
        text_file = os.path.join(output_dir, "length_combination_report.txt")
        print(f"Saving text report to {text_file}...")
        
        with open(text_file, 'w') as f:
            f.write("PROMPT/COT LENGTH COMBINATION ANALYSIS REPORT\n")
            f.write("=" * 140 + "\n\n")
            f.write(f"Source file: {file_path}\n")
            f.write(f"Analysis date: {pd.Timestamp.now()}\n")
            f.write(f"Minimum qualifying CoTs per prompt: {min_qualifying_cots}\n")
            f.write(f"Total original prompts: {len(prompts):,}\n")
            f.write(f"Total original CoT sequences: {sum(len(cots) for cots in cot_sequences):,}\n")
            f.write(f"Total combinations tested: {len(combinations)}\n\n")
            
            f.write("SUMMARY TABLE:\n")
            f.write(f"{'Prompt':<8} {'CoT':<8} {'Multiplier':<10} {'Qualifying':<12} {'Qualifying':<12} {'Avg CoTs':<10} {'Total Padding':<15} {'Padding %':<12}\n")
            f.write(f"{'Length':<8} {'Length':<8} {'':<10} {'Prompts':<12} {'CoTs':<12} {'per Prompt':<10} {'(tokens)':<15} {'':<12}\n")
            f.write(f"{'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*15} {'-'*12}\n")
            
            for prompt_len, cot_len in combinations:
                multiplier = "1x" if cot_len == prompt_len else "3x"
                combination_key = f"prompt_{prompt_len}_cot_{cot_len}_{multiplier}"
                data = results['combinations'][combination_key]
                
                f.write(f"{prompt_len:<8} {cot_len:<8} {multiplier:<10} {data['qualifying_prompts']:<12} {data['qualifying_cots']:<12} "
                       f"{data['avg_cots_per_prompt']:<10.2f} {data['padding_stats']['total_padding_tokens']:<15} "
                       f"{data['padding_stats']['total_padding_percentage']:<12.2f}%\n")
            
            f.write(f"\nDETAILED RESULTS:\n")
            for prompt_len, cot_len in combinations:
                multiplier = "1x" if cot_len == prompt_len else "3x"
                combination_key = f"prompt_{prompt_len}_cot_{cot_len}_{multiplier}"
                data = results['combinations'][combination_key]
                
                f.write(f"\nPrompt Length: {prompt_len}, CoT Length: {cot_len} ({multiplier})\n")
                f.write(f"  Qualifying prompts: {data['qualifying_prompts']:,}\n")
                f.write(f"  Qualifying CoTs: {data['qualifying_cots']:,}\n")
                f.write(f"  Average CoTs per prompt: {data['avg_cots_per_prompt']:.2f}\n")
                f.write(f"  Prompt padding: {data['padding_stats']['prompt_padding_tokens']:,} tokens ({data['padding_stats']['prompt_padding_percentage']:.2f}%)\n")
                f.write(f"  CoT padding: {data['padding_stats']['cot_padding_tokens']:,} tokens ({data['padding_stats']['cot_padding_percentage']:.2f}%)\n")
                f.write(f"  Total padding: {data['padding_stats']['total_padding_tokens']:,} tokens ({data['padding_stats']['total_padding_percentage']:.2f}%)\n")
                f.write(f"  Dataset directory: data/GSM8K/prompt_{prompt_len}_cot_{cot_len}_{multiplier}/\n")
            
            f.write(f"\nANALYSIS NOTES:\n")
            f.write("- This analysis tests different prompt/CoT length combinations\n")
            f.write("- Prompt lengths are powers of 2 (64, 128, 256, 512, 1024, 2048)\n")
            f.write("- CoT lengths are 1x or 3x of the corresponding prompt length\n")
            f.write("- Only prompts with at least the minimum number of qualifying CoTs are included\n")
            f.write("- Padding statistics show efficiency of each combination\n")
            f.write("- Generated datasets are saved in separate directories for each combination\n")
            f.write("- 1x combinations have equal prompt and CoT lengths\n")
            f.write("- 3x combinations have CoT lengths three times the prompt length\n")
        
        print(f"Results saved successfully!")
        print(f"Files created:")
        print(f"  - {json_file}")
        print(f"  - {text_file}")
        print(f"  - Individual dataset directories for each combination")
    
    return results

def calculate_padding_ratio(dataset_dir: str) -> dict:
    """
    Calculate the ratio of padding to full sequence tensors for an existing dataset.
    
    This function reads the saved tensors and uses the masks to efficiently calculate
    padding ratios without needing to examine individual token values.
    
    Args:
        dataset_dir (str): Directory containing the dataset files:
                          - prompt_sequences.pt
                          - cot_sequences_tensor.pt
                          - prompt_mask.pt
                          - cot_mask.pt
    
    Returns:
        dict: Dictionary containing padding statistics:
              - prompt_padding_ratio: Ratio of padding tokens in prompts
              - cot_padding_ratio: Ratio of padding tokens in CoTs
              - total_padding_ratio: Overall padding ratio
              - prompt_stats: Detailed prompt statistics
              - cot_stats: Detailed CoT statistics
              - tensor_shapes: Shapes of all tensors
    """
    import os
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Define expected file paths
    prompt_sequences_path = os.path.join(dataset_dir, "prompt_sequences.pt")
    cot_sequences_path = os.path.join(dataset_dir, "cot_sequences_tensor.pt")
    prompt_mask_path = os.path.join(dataset_dir, "prompt_mask.pt")
    cot_mask_path = os.path.join(dataset_dir, "cot_mask.pt")
    
    # Check if all required files exist
    required_files = [prompt_sequences_path, cot_sequences_path, prompt_mask_path, cot_mask_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")
    
    print(f"Loading dataset from: {dataset_dir}")
    
    # Load tensors
    prompt_sequences = torch.load(prompt_sequences_path)
    cot_sequences_tensor = torch.load(cot_sequences_path)
    prompt_mask = torch.load(prompt_mask_path)
    cot_mask = torch.load(cot_mask_path)
    
    print(f"Loaded tensors:")
    print(f"  prompt_sequences: {prompt_sequences.shape}")
    print(f"  cot_sequences_tensor: {cot_sequences_tensor.shape}")
    print(f"  prompt_mask: {prompt_mask.shape}")
    print(f"  cot_mask: {cot_mask.shape}")
    
    # Calculate padding statistics using masks
    # Mask values: 1.0 = actual token, 0.0 = padding token
    
    # Prompt padding calculation
    total_prompt_tokens = prompt_mask.numel()
    actual_prompt_tokens = prompt_mask.sum().item()
    prompt_padding_tokens = total_prompt_tokens - actual_prompt_tokens
    prompt_padding_ratio = prompt_padding_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0
    
    # CoT padding calculation
    total_cot_tokens = cot_mask.numel()
    actual_cot_tokens = cot_mask.sum().item()
    cot_padding_tokens = total_cot_tokens - actual_cot_tokens
    cot_padding_ratio = cot_padding_tokens / total_cot_tokens if total_cot_tokens > 0 else 0
    
    # Overall padding calculation
    total_tokens = total_prompt_tokens + total_cot_tokens
    total_actual_tokens = actual_prompt_tokens + actual_cot_tokens
    total_padding_tokens = total_padding_tokens = total_tokens - total_actual_tokens
    total_padding_ratio = total_padding_tokens / total_tokens if total_tokens > 0 else 0
    
    # Calculate per-prompt statistics
    num_prompts = prompt_mask.shape[0]
    prompt_length = prompt_mask.shape[1]
    num_cots_per_prompt = cot_mask.shape[1] if len(cot_mask.shape) > 2 else 1
    cot_length = cot_mask.shape[-1]
    
    # Average tokens per prompt (actual, not padded)
    avg_actual_prompt_tokens = actual_prompt_tokens / num_prompts if num_prompts > 0 else 0
    
    # Average tokens per CoT (actual, not padded)
    total_cot_sequences = num_prompts * num_cots_per_prompt
    avg_actual_cot_tokens = actual_cot_tokens / total_cot_sequences if total_cot_sequences > 0 else 0
    
    # Compile results
    results = {
        'dataset_dir': dataset_dir,
        'prompt_padding_ratio': prompt_padding_ratio,
        'cot_padding_ratio': cot_padding_ratio,
        'total_padding_ratio': total_padding_ratio,
        'prompt_stats': {
            'total_tokens': total_prompt_tokens,
            'actual_tokens': actual_prompt_tokens,
            'padding_tokens': prompt_padding_tokens,
            'padding_ratio': prompt_padding_ratio,
            'avg_actual_tokens_per_prompt': avg_actual_prompt_tokens,
            'max_prompt_length': prompt_length
        },
        'cot_stats': {
            'total_tokens': total_cot_tokens,
            'actual_tokens': actual_cot_tokens,
            'padding_tokens': cot_padding_tokens,
            'padding_ratio': cot_padding_ratio,
            'avg_actual_tokens_per_cot': avg_actual_cot_tokens,
            'max_cot_length': cot_length,
            'num_cots_per_prompt': num_cots_per_prompt
        },
        'overall_stats': {
            'total_tokens': total_tokens,
            'actual_tokens': total_actual_tokens,
            'padding_tokens': total_padding_tokens,
            'padding_ratio': total_padding_ratio,
            'num_prompts': num_prompts,
            'total_cot_sequences': total_cot_sequences
        },
        'tensor_shapes': {
            'prompt_sequences': prompt_sequences.shape,
            'cot_sequences_tensor': cot_sequences_tensor.shape,
            'prompt_mask': prompt_mask.shape,
            'cot_mask': cot_mask.shape
        }
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("PADDING RATIO ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_dir}")
    print(f"Number of prompts: {num_prompts:,}")
    print(f"Total CoT sequences: {total_cot_sequences:,}")
    print(f"Max prompt length: {prompt_length}")
    print(f"Max CoT length: {cot_length}")
    print(f"CoTs per prompt: {num_cots_per_prompt}")
    
    print(f"\nPadding Ratios:")
    print(f"  Prompts: {prompt_padding_ratio:.4f} ({prompt_padding_ratio*100:.2f}%)")
    print(f"  CoTs: {cot_padding_ratio:.4f} ({cot_padding_ratio*100:.2f}%)")
    print(f"  Overall: {total_padding_ratio:.4f} ({total_padding_ratio*100:.2f}%)")
    
    print(f"\nToken Counts:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Actual tokens: {total_actual_tokens:,}")
    print(f"  Padding tokens: {total_padding_tokens:,}")
    
    print(f"\nAverage Tokens per Sequence:")
    print(f"  Prompts: {avg_actual_prompt_tokens:.1f} / {prompt_length}")
    print(f"  CoTs: {avg_actual_cot_tokens:.1f} / {cot_length}")
    
    return results

if __name__ == "__main__":
    # GSM8K dataset
    FILE_PATH = "dart-math-uniform/data/train-00001-of-00002.parquet"
    OUTPUT_DIR = "data/GSM8K/128_128"

    save_tensors_in_usable_form(FILE_PATH, output_dir=OUTPUT_DIR, 
                                max_prompt_length=128, max_cot_length=128, 
                                min_qualifying_cots=1)

    # analyze_length_distribution(file_path=FILE_PATH, 
    #                             save_results=True)
    
    # Example usage of the new function
    # analyze_and_generate_datasets(file_path=FILE_PATH,
    #                              min_qualifying_cots=8,
    #                              save_results=True)

    # calculate_padding_ratio(dataset_dir=r"data/GSM8K/256_784")
    # calculate_padding_ratio(dataset_dir=r"data/GSM8K/128_128")