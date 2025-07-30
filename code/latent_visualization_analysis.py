#!/usr/bin/env python3
"""
Latent Visualization Analysis for Enhanced VQVAE

This module provides comprehensive visualization tools for analyzing the latent representations
learned by the Enhanced VQVAE system, focusing on:

1. Code-to-Content Mapping: Understanding what semantic content each code represents
2. Sequence-Level Analysis: Understanding how codes are used in sequence generation

Author: Akira Kudo
Created: 2025/07/07
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
import json
import yaml
from pathlib import Path

# Add the code directory to the path to import the VQVAE modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE
from transformers import GPT2Tokenizer

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LatentVisualizationAnalyzer:
    """
    Comprehensive analyzer for latent representations in Enhanced VQVAE.
    
    This class provides tools for:
    1. Code-to-Content Mapping: Understanding semantic meaning of codes
    2. Sequence-Level Analysis: Analyzing code usage patterns in sequences
    3. Statistical Analysis: Computing various metrics and statistics
    """
    
    def __init__(self, model: EnhancedGPT2VQVAE, tokenizer: GPT2Tokenizer, device: str = "cuda"):
        """
        Initialize the analyzer.
        
        Args:
            model: Loaded Enhanced VQVAE model
            tokenizer: GPT2 tokenizer for text decoding
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Extract model parameters
        self.num_embeddings = model.vector_quantizer.num_embeddings
        self.embedding_dim = model.vector_quantizer.embedding_dim
        self.num_thoughts = model.num_thoughts
        
        print(f"Initialized analyzer for model with {self.num_embeddings} embeddings")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Number of thoughts: {self.num_thoughts}")
    
    def load_dataset(self, data_dir: str, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load dataset from the specified directory.
        
        Args:
            data_dir: Directory containing the dataset files
            max_samples: Maximum number of samples to load (for memory constraints)
            
        Returns:
            Tuple of (prompt_sequences, cot_sequences, prompt_mask, cot_mask)
        """
        required_files = [
            "prompt_sequences.pt",
            "cot_sequences_tensor.pt", 
            "prompt_mask.pt",
            "cot_mask.pt"
        ]
        
        # Check if all required files exist
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(f"Missing data files in {data_dir}: {missing_files}")
        
        # Load tensors
        print(f"Loading dataset from: {data_dir}")
        prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"), map_location='cpu')
        cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"), map_location='cpu')
        prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"), map_location='cpu')
        cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"), map_location='cpu')
        
        print(f"Dataset shapes:")
        print(f"  prompt_sequences: {prompt_sequences.shape}")
        print(f"  cot_sequences: {cot_sequences.shape}")
        print(f"  prompt_mask: {prompt_mask.shape}")
        print(f"  cot_mask: {cot_mask.shape}")
        
        # Limit samples if specified
        if max_samples is not None:
            prompt_sequences = prompt_sequences[:max_samples]
            cot_sequences = cot_sequences[:max_samples]
            prompt_mask = prompt_mask[:max_samples]
            cot_mask = cot_mask[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        return prompt_sequences, cot_sequences, prompt_mask, cot_mask
    
    def decode_tokens(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
        """
        Decode tokens to text using the tokenizer.
        
        Args:
            tokens: Token tensor
            mask: Optional mask to apply before decoding
            
        Returns:
            Decoded text string
        """
        if mask is not None:
            # Apply mask to remove padding
            tokens = tokens[mask.bool()]
        
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            return f"[Decode error: {e}, tokens: {tokens.tolist()}]"
    
    def extract_code_usage_data(self, prompt_sequences: torch.Tensor, cot_sequences: torch.Tensor, 
                               prompt_mask: torch.Tensor, cot_mask: torch.Tensor, 
                               sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract code usage data from the dataset.
        
        Args:
            prompt_sequences: Prompt sequences tensor
            cot_sequences: CoT sequences tensor
            prompt_mask: Prompt mask tensor
            cot_mask: CoT mask tensor
            sample_size: Number of samples to process (None for all)
            
        Returns:
            Dictionary containing code usage data
        """
        print("Extracting code usage data...")
        
        # Determine sample size
        total_samples = len(prompt_sequences)
        if sample_size is None:
            sample_size = total_samples
        else:
            sample_size = min(sample_size, total_samples)
        
        # Initialize data structures
        code_contexts = defaultdict(list)  # code -> list of contexts
        code_positions = defaultdict(list)  # code -> list of positions
        all_indices = []
        all_texts = []
        
        # Process samples
        with torch.no_grad():
            for i in range(sample_size):
                if i % 100 == 0:
                    print(f"Processing sample {i}/{sample_size}")
                
                # Prepare single example
                prompt = prompt_sequences[i:i+1].to(self.device)
                cot_gt = cot_sequences[i:i+1].to(self.device)
                prompt_mask_ex = prompt_mask[i:i+1].to(self.device) if prompt_mask is not None else None
                cot_mask_ex = cot_mask[i:i+1].to(self.device) if cot_mask is not None else None
                
                # Get encoding indices
                try:
                    _, _, _, indices, _ = self.model.encode(
                        prompt, cot_gt, prompt_mask_ex, cot_mask_ex, 
                        quantize_cot_only=True
                    )
                    
                    # Decode text for context
                    prompt_text = self.decode_tokens(prompt[0], prompt_mask_ex[0] if prompt_mask_ex is not None else None)
                    cot_text = self.decode_tokens(cot_gt[0, 0], cot_mask_ex[0, 0] if cot_mask_ex is not None else None)
                    full_text = f"Prompt: {prompt_text}\nCoT: {cot_text}"
                    
                    # Store data
                    all_indices.append(indices.cpu())
                    all_texts.append(full_text)
                    
                    # Map codes to their context
                    for pos, code in enumerate(indices[0]):
                        code_item = code.item()
                        code_contexts[code_item].append(full_text)
                        code_positions[code_item].append(pos)
                
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
        
        # Compile statistics
        code_usage_counts = Counter()
        for indices_batch in all_indices:
            code_usage_counts.update(indices_batch.flatten().tolist())
        
        return {
            'code_contexts': dict(code_contexts),
            'code_positions': dict(code_positions),
            'code_usage_counts': code_usage_counts,
            'all_indices': all_indices,
            'all_texts': all_texts,
            'total_samples': sample_size
        }
    
    def analyze_code_semantics(self, code_usage_data: Dict[str, Any], 
                              output_dir: str, top_k_codes: int = 20) -> None:
        """
        Analyze semantic meaning of codes by examining their usage patterns.
        
        Args:
            code_usage_data: Data from extract_code_usage_data
            output_dir: Directory to save visualizations
            top_k_codes: Number of top codes to analyze in detail
        """
        print("Analyzing code semantics...")
        
        code_contexts = code_usage_data['code_contexts']
        code_usage_counts = code_usage_data['code_usage_counts']
        
        # Get top used codes
        top_codes = sorted(code_usage_counts.items(), key=lambda x: x[1], reverse=True)[:top_k_codes]
        
        # Create semantic analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Usage distribution
        codes, counts = zip(*top_codes)
        axes[0, 0].bar(range(len(codes)), counts)
        axes[0, 0].set_title('Top Code Usage Distribution')
        axes[0, 0].set_xlabel('Code Index')
        axes[0, 0].set_ylabel('Usage Count')
        axes[0, 0].set_xticks(range(len(codes)))
        axes[0, 0].set_xticklabels(codes, rotation=45)
        
        # 2. Positional analysis
        code_positions = code_usage_data['code_positions']
        position_data = []
        for code in codes:
            if code in code_positions:
                position_data.append(code_positions[code])
        
        if position_data:
            axes[0, 1].boxplot(position_data, labels=codes)
            axes[0, 1].set_title('Code Usage by Position')
            axes[0, 1].set_xlabel('Code Index')
            axes[0, 1].set_ylabel('Position in Sequence')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # TODO CONSIDER: UNSURE IF USEFUL
        # 3. Context similarity analysis
        context_lengths = [len(code_contexts.get(code, [])) for code in codes]
        axes[1, 0].bar(range(len(codes)), context_lengths)
        axes[1, 0].set_title('Number of Contexts per Code')
        axes[1, 0].set_xlabel('Code Index')
        axes[1, 0].set_ylabel('Number of Contexts')
        axes[1, 0].set_xticks(range(len(codes)))
        axes[1, 0].set_xticklabels(codes, rotation=45)
        
        # 4. Usage vs. context diversity
        axes[1, 1].scatter(counts, context_lengths)
        axes[1, 1].set_title('Usage Count vs. Context Diversity')
        axes[1, 1].set_xlabel('Usage Count')
        axes[1, 1].set_ylabel('Number of Contexts')
        
        # Add code labels to scatter plot
        for i, code in enumerate(codes):
            axes[1, 1].annotate(str(code), (counts[i], context_lengths[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'code_semantics_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed code analysis
        self._save_detailed_code_analysis(code_usage_data, output_dir, top_codes)
    
    def _save_detailed_code_analysis(self, code_usage_data: Dict[str, Any], 
                                   output_dir: str, top_codes: List[Tuple[int, int]]) -> None:
        """
        Save detailed analysis of individual codes.
        
        Args:
            code_usage_data: Data from extract_code_usage_data
            output_dir: Directory to save analysis
            top_codes: List of (code, count) tuples
        """
        code_contexts = code_usage_data['code_contexts']
        code_positions = code_usage_data['code_positions']
        
        # Create detailed analysis file
        analysis_file = os.path.join(output_dir, 'detailed_code_analysis.txt')
        
        with open(analysis_file, 'w') as f:
            f.write("Detailed Code Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for code, count in top_codes:
                f.write(f"Code {code} (used {count} times)\n")
                f.write("-" * 30 + "\n")
                
                # Position analysis
                if code in code_positions:
                    positions = code_positions[code]
                    f.write(f"Position statistics:\n")
                    f.write(f"  Mean position: {np.mean(positions):.2f}\n")
                    f.write(f"  Std position: {np.std(positions):.2f}\n")
                    f.write(f"  Min position: {min(positions)}\n")
                    f.write(f"  Max position: {max(positions)}\n")
                
                # Start/End frequency analysis (from transition matrix)
                # This will be computed separately if needed
                f.write(f"Start/End analysis: Available in transition matrix\n")
                
                # Sample contexts
                if code in code_contexts:
                    contexts = code_contexts[code]
                    f.write(f"Sample contexts (showing first 3):\n")
                    for i, context in enumerate(contexts[:3]):
                        f.write(f"  {i+1}. {context[:200]}...\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"Detailed analysis saved to: {analysis_file}")
    
    def analyze_sequence_patterns(self, code_usage_data: Dict[str, Any], 
                                output_dir: str, max_sequence_length: int = 50) -> None:
        """
        Analyze sequence-level patterns in code usage.
        
        Args:
            code_usage_data: Data from extract_code_usage_data
            output_dir: Directory to save visualizations
            max_sequence_length: Maximum sequence length to analyze
        """
        print("Analyzing sequence patterns...")
        
        all_indices = code_usage_data['all_indices']
        
        # Create sequence analysis visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Code transition matrix
        transition_matrix = self._compute_transition_matrix(all_indices)
        
        # Plot transition matrix (top codes only) - exclude start/end row/column
        top_codes = sorted(code_usage_data['code_usage_counts'].items(), 
                          key=lambda x: x[1], reverse=True)[:20]
        top_code_indices = [code for code, _ in top_codes]
        
        # Extract submatrix excluding the start/end row and column
        transition_submatrix = transition_matrix[np.ix_(top_code_indices, top_code_indices)]
        
        im = axes[0, 0].imshow(transition_submatrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Code Transition Matrix (Top 20 Codes)')
        axes[0, 0].set_xlabel('Next Code')
        axes[0, 0].set_ylabel('Current Code')
        axes[0, 0].set_xticks(range(len(top_code_indices)))
        axes[0, 0].set_xticklabels(top_code_indices, rotation=45)
        axes[0, 0].set_yticks(range(len(top_code_indices)))
        axes[0, 0].set_yticklabels(top_code_indices)
        plt.colorbar(im, ax=axes[0, 0], label='Transition Count')
        
        # 2. Position-dependent code usage
        position_usage = self._compute_position_usage(all_indices, max_sequence_length)
        
        # Plot heatmap of position vs code usage
        top_codes_for_pos = sorted(code_usage_data['code_usage_counts'].items(), 
                                  key=lambda x: x[1], reverse=True)[:15]
        top_code_indices_pos = [code for code, _ in top_codes_for_pos]
        
        position_submatrix = position_usage[top_code_indices_pos, :max_sequence_length]
        
        im = axes[0, 1].imshow(position_submatrix, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Code Usage by Position (Top 15 Codes)')
        axes[0, 1].set_xlabel('Position in Sequence')
        axes[0, 1].set_ylabel('Code Index')
        axes[0, 1].set_yticks(range(len(top_code_indices_pos)))
        axes[0, 1].set_yticklabels(top_code_indices_pos)
        plt.colorbar(im, ax=axes[0, 1], label='Usage Count')
        
        # 3. Common code sequences
        common_sequences = self._find_common_sequences(all_indices, min_length=2, max_length=4)
        
        if common_sequences:
            sequences, counts = zip(*common_sequences[:10])
            sequence_labels = ['→'.join(map(str, seq)) for seq in sequences]
            
            axes[1, 0].barh(range(len(sequences)), counts)
            axes[1, 0].set_title('Most Common Code Sequences')
            axes[1, 0].set_xlabel('Count')
            axes[1, 0].set_ylabel('Sequence')
            axes[1, 0].set_yticks(range(len(sequences)))
            axes[1, 0].set_yticklabels(sequence_labels)
        
        # 4. Code diversity over sequence length
        diversity_by_position = self._compute_diversity_by_position(all_indices, max_sequence_length)
        
        axes[1, 1].plot(range(len(diversity_by_position)), diversity_by_position, marker='o')
        axes[1, 1].set_title('Code Diversity by Position')
        axes[1, 1].set_xlabel('Position in Sequence')
        axes[1, 1].set_ylabel('Number of Unique Codes')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sequence_patterns_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save sequence statistics
        self._save_sequence_statistics(code_usage_data, output_dir, transition_matrix, common_sequences)
        
        # Create start/end frequency visualization
        self._visualize_start_end_frequencies(transition_matrix, output_dir)
    
    def _compute_transition_matrix(self, all_indices: List[torch.Tensor]) -> np.ndarray:
        """
        Compute transition matrix between codes with start/end frequencies.
        
        Returns:
            Extended transition matrix with shape (num_embeddings + 1, num_embeddings + 1)
            where:
            - [i, j] for i,j < num_embeddings: frequency of code i followed by code j
            - [i, num_embeddings] for i < num_embeddings: frequency of code i at end of CoT
            - [num_embeddings, j] for j < num_embeddings: frequency of code j at start of CoT
            - [num_embeddings, num_embeddings]: total number of CoT sequences
        """
        # Extended matrix: +1 for start/end row/column
        extended_size = self.num_embeddings + 1
        transition_counts = np.zeros((extended_size, extended_size))
        
        for indices in all_indices:
            indices_flat = indices.flatten().numpy()
            seq_length = len(indices_flat)
            
            if seq_length == 0:
                continue
                
            # Count start token (first token in sequence)
            start_code = indices_flat[0]
            transition_counts[start_code, self.num_embeddings] += 1  # Start frequency
            
            # Count end token (last token in sequence)
            end_code = indices_flat[-1]
            transition_counts[self.num_embeddings, end_code] += 1  # End frequency
            
            # Count transitions between consecutive tokens
            for i in range(seq_length - 1):
                current_code = indices_flat[i]
                next_code = indices_flat[i + 1]
                transition_counts[current_code, next_code] += 1
            
            # Increment total sequence count
            transition_counts[self.num_embeddings, self.num_embeddings] += 1
        
        return transition_counts
    
    def _compute_position_usage(self, all_indices: List[torch.Tensor], 
                              max_length: int) -> np.ndarray:
        """Compute code usage by position."""
        position_usage = np.zeros((self.num_embeddings, max_length))
        
        for indices in all_indices:
            indices_flat = indices.flatten().numpy()
            for pos, code in enumerate(indices_flat):
                if pos < max_length:
                    position_usage[code, pos] += 1
        
        return position_usage
    
    def _find_common_sequences(self, all_indices: List[torch.Tensor], 
                             min_length: int = 2, max_length: int = 4) -> List[Tuple[Tuple, int]]:
        """Find most common code sequences."""
        sequence_counts = Counter()
        
        for indices in all_indices:
            indices_flat = indices.flatten().numpy()
            for length in range(min_length, max_length + 1):
                for i in range(len(indices_flat) - length + 1):
                    sequence = tuple(indices_flat[i:i+length])
                    sequence_counts[sequence] += 1
        
        return sequence_counts.most_common(20)
    
    def _compute_diversity_by_position(self, all_indices: List[torch.Tensor], 
                                     max_length: int) -> List[int]:
        """Compute number of unique codes used at each position."""
        diversity = []
        
        for pos in range(max_length):
            codes_at_position = set()
            for indices in all_indices:
                indices_flat = indices.flatten().numpy()
                if pos < len(indices_flat):
                    codes_at_position.add(indices_flat[pos])
            diversity.append(len(codes_at_position))
        
        return diversity
    
    def _save_sequence_statistics(self, code_usage_data: Dict[str, Any], 
                                output_dir: str, transition_matrix: np.ndarray, 
                                common_sequences: List[Tuple[Tuple, int]]) -> None:
        """Save sequence statistics to file."""
        stats_file = os.path.join(output_dir, 'sequence_statistics.txt')
        
        with open(stats_file, 'w') as f:
            f.write("Sequence Analysis Statistics\n")
            f.write("=" * 40 + "\n\n")
            
            # Transition statistics
            f.write("Transition Matrix Statistics:\n")
            f.write(f"  Matrix shape: {transition_matrix.shape}\n")
            f.write(f"  Non-zero transitions: {np.count_nonzero(transition_matrix)}\n")
            f.write(f"  Total sequences: {transition_matrix[-1, -1]:.0f}\n")
            f.write(f"  Mean transition count: {np.mean(transition_matrix[:-1, :-1][transition_matrix[:-1, :-1] > 0]):.2f}\n")
            f.write(f"  Max transition count: {np.max(transition_matrix[:-1, :-1]):.0f}\n\n")
            
            # Start/End statistics
            start_frequencies = transition_matrix[-1, :-1]  # Last row, excluding last column
            end_frequencies = transition_matrix[:-1, -1]    # Last column, excluding last row
            
            f.write("Start/End Token Statistics:\n")
            f.write(f"  Most common start tokens:\n")
            start_indices = np.argsort(start_frequencies)[::-1][:10]
            for i, idx in enumerate(start_indices):
                if start_frequencies[idx] > 0:
                    f.write(f"    {i+1}. Code {idx}: {start_frequencies[idx]:.0f} times\n")
            
            f.write(f"  Most common end tokens:\n")
            end_indices = np.argsort(end_frequencies)[::-1][:10]
            for i, idx in enumerate(end_indices):
                if end_frequencies[idx] > 0:
                    f.write(f"    {i+1}. Code {idx}: {end_frequencies[idx]:.0f} times\n")
            
            # Common sequences
            f.write("\nMost Common Sequences:\n")
            for i, (sequence, count) in enumerate(common_sequences[:10]):
                f.write(f"  {i+1}. {sequence} (count: {count})\n")
            
            f.write("\n" + "=" * 40 + "\n")
        
        print(f"Sequence statistics saved to: {stats_file}")
    
    def _visualize_start_end_frequencies(self, transition_matrix: np.ndarray, output_dir: str) -> None:
        """
        Create visualization for start and end token frequencies.
        
        Args:
            transition_matrix: Extended transition matrix with start/end frequencies
            output_dir: Directory to save visualization
        """
        print("Creating start/end frequency visualization...")
        
        # Extract start and end frequencies
        start_frequencies = transition_matrix[-1, :-1]  # Last row, excluding last column
        end_frequencies = transition_matrix[:-1, -1]    # Last column, excluding last row
        
        # Get top codes for visualization
        top_start_indices = np.argsort(start_frequencies)[::-1][:15]
        top_end_indices = np.argsort(end_frequencies)[::-1][:15]
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Start frequencies
        start_codes = [f"Code {idx}" for idx in top_start_indices]
        start_counts = [start_frequencies[idx] for idx in top_start_indices]
        
        axes[0].barh(range(len(start_codes)), start_counts, color='lightblue')
        axes[0].set_title('Most Common Start Tokens', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Frequency')
        axes[0].set_ylabel('Code Index')
        axes[0].set_yticks(range(len(start_codes)))
        axes[0].set_yticklabels(start_codes)
        axes[0].grid(True, alpha=0.3)
        
        # End frequencies
        end_codes = [f"Code {idx}" for idx in top_end_indices]
        end_counts = [end_frequencies[idx] for idx in top_end_indices]
        
        axes[1].barh(range(len(end_codes)), end_counts, color='lightcoral')
        axes[1].set_title('Most Common End Tokens', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Code Index')
        axes[1].set_yticks(range(len(end_codes)))
        axes[1].set_yticklabels(end_codes)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'start_end_frequencies.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Start/end frequency visualization saved to: {output_dir}/start_end_frequencies.png")
    
    def create_comprehensive_visualization(self, code_usage_data: Dict[str, Any], 
                                         output_dir: str) -> None:
        """
        Create a comprehensive visualization dashboard.
        
        Args:
            code_usage_data: Data from extract_code_usage_data
            output_dir: Directory to save visualizations
        """
        print("Creating comprehensive visualization dashboard...")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 20))
        
        # Grid layout: 4 rows, 3 columns
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Code usage distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        code_usage_counts = code_usage_data['code_usage_counts']
        top_codes = sorted(code_usage_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        codes, counts = zip(*top_codes)
        ax1.bar(range(len(codes)), counts, color='skyblue')
        ax1.set_title('Top 15 Code Usage', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Code Index')
        ax1.set_ylabel('Usage Count')
        ax1.set_xticks(range(len(codes)))
        ax1.set_xticklabels(codes, rotation=45)
        
        # 2. Usage vs. context diversity scatter (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        code_contexts = code_usage_data['code_contexts']
        usage_vs_context = [(count, len(code_contexts.get(code, []))) 
                           for code, count in top_codes]
        usage_counts, context_counts = zip(*usage_vs_context)
        ax2.scatter(usage_counts, context_counts, alpha=0.7, s=100)
        ax2.set_title('Usage vs. Context Diversity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Usage Count')
        ax2.set_ylabel('Number of Contexts')
        
        # 3. Position analysis (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        code_positions = code_usage_data['code_positions']
        position_data = [code_positions.get(code, []) for code in codes[:10]]
        if any(position_data):
            ax3.boxplot(position_data, labels=codes[:10])
            ax3.set_title('Code Usage by Position', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Code Index')
            ax3.set_ylabel('Position in Sequence')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Transition matrix heatmap (middle left, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        all_indices = code_usage_data['all_indices']
        transition_matrix = self._compute_transition_matrix(all_indices)
        
        # Use top 10 codes for visualization - exclude start/end row/column
        top_10_codes = [code for code, _ in top_codes[:10]]
        transition_submatrix = transition_matrix[np.ix_(top_10_codes, top_10_codes)]
        
        im = ax4.imshow(transition_submatrix, cmap='viridis', aspect='auto')
        ax4.set_title('Code Transition Matrix (Top 10 Codes)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Next Code')
        ax4.set_ylabel('Current Code')
        ax4.set_xticks(range(len(top_10_codes)))
        ax4.set_xticklabels(top_10_codes)
        ax4.set_yticks(range(len(top_10_codes)))
        ax4.set_yticklabels(top_10_codes)
        plt.colorbar(im, ax=ax4, label='Transition Count')
        
        # 5. Position usage heatmap (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        position_usage = self._compute_position_usage(all_indices, 30)
        position_submatrix = position_usage[top_10_codes, :30]
        
        im = ax5.imshow(position_submatrix, cmap='viridis', aspect='auto')
        ax5.set_title('Code Usage by Position', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Position')
        ax5.set_ylabel('Code Index')
        ax5.set_yticks(range(len(top_10_codes)))
        ax5.set_yticklabels(top_10_codes)
        plt.colorbar(im, ax=ax5, label='Usage Count')
        
        # 6. Common sequences (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        common_sequences = self._find_common_sequences(all_indices, min_length=2, max_length=3)
        if common_sequences:
            sequences, counts = zip(*common_sequences[:8])
            sequence_labels = ['→'.join(map(str, seq)) for seq in sequences]
            
            ax6.barh(range(len(sequences)), counts, color='lightcoral')
            ax6.set_title('Most Common Sequences', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Count')
            ax6.set_ylabel('Sequence')
            ax6.set_yticks(range(len(sequences)))
            ax6.set_yticklabels(sequence_labels)
        
        # 7. Diversity over position (bottom middle)
        ax7 = fig.add_subplot(gs[2, 1])
        diversity_by_position = self._compute_diversity_by_position(all_indices, 30)
        ax7.plot(range(len(diversity_by_position)), diversity_by_position, 
                marker='o', linewidth=2, markersize=6, color='green')
        ax7.set_title('Code Diversity by Position', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Position in Sequence')
        ax7.set_ylabel('Number of Unique Codes')
        ax7.grid(True, alpha=0.3)
        
        # 8. Code embedding similarity (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        embedding_weight = self.model.vector_quantizer.embedding.weight
        normalized_embeddings = F.normalize(embedding_weight, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Use top 10 codes for similarity visualization
        similarity_submatrix = similarity_matrix[top_10_codes][:, top_10_codes].cpu().numpy()
        
        im = ax8.imshow(similarity_submatrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax8.set_title('Code Embedding Similarity', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Code Index')
        ax8.set_ylabel('Code Index')
        ax8.set_xticks(range(len(top_10_codes)))
        ax8.set_xticklabels(top_10_codes)
        ax8.set_yticks(range(len(top_10_codes)))
        ax8.set_yticklabels(top_10_codes)
        plt.colorbar(im, ax=ax8, label='Cosine Similarity')
        
        # 9. Summary statistics (bottom row, spanning all columns)
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')
        
        # Calculate summary statistics
        total_codes_used = len(code_usage_counts)
        total_usage = sum(code_usage_counts.values())
        unused_codes = self.num_embeddings - total_codes_used
        usage_ratio = total_codes_used / self.num_embeddings
        
        # Get model statistics
        model_stats = self.model.get_vector_quantizer_stats()
        diversity_metrics = self.model.get_embedding_diversity()
        
        summary_text = f"""
        SUMMARY STATISTICS
        ==================
        
        Codebook Usage:
        • Total codes used: {total_codes_used}/{self.num_embeddings} ({usage_ratio:.1%})
        • Unused codes: {unused_codes}
        • Total usage count: {total_usage:,}
        • Most used code: {max(code_usage_counts.items(), key=lambda x: x[1])[0]} ({max(code_usage_counts.values()):,} times)
        
        Model Statistics:
        • Unused ratio: {model_stats.get('unused_ratio', 0):.3f}
        • Reset counter: {model_stats.get('reset_counter', 0)}
        • Mean embedding similarity: {diversity_metrics.get('mean_similarity', 0):.4f}
        • Embedding norm mean: {diversity_metrics.get('embedding_norm_mean', 0):.4f}
        
        Dataset Statistics:
        • Total samples analyzed: {code_usage_data['total_samples']}
        • Average sequence length: {np.mean([len(indices.flatten()) for indices in code_usage_data['all_indices']]):.1f}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.savefig(os.path.join(output_dir, 'comprehensive_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive dashboard saved to: {output_dir}/comprehensive_dashboard.png")
    
    def visualize_latent_sequence(self, latents: torch.Tensor, 
                                 title: str = "Latent Sequence Visualization",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 cmap: str = 'viridis',
                                 show_norms: bool = True,
                                 show_similarity: bool = True) -> None:
        """
        Visualize a sequence of encoded latents in a compact figure.
        
        Args:
            latents: Tensor of shape (L, embedding_dim) where L is sequence length
            title: Title for the visualization
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
            cmap: Colormap for the heatmap
            show_norms: Whether to show embedding norms over sequence
            show_similarity: Whether to show similarity between consecutive latents
        """
        if latents.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got shape {latents.shape}")
        
        L, embedding_dim = latents.shape
        print(f"Visualizing latent sequence of length {L} with embedding dimension {embedding_dim}")
        
        # Convert to numpy for visualization
        latents_np = latents.detach().cpu().numpy()
        
        # Create figure with subplots
        if show_norms and show_similarity:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Main heatmap (top, full width)
            ax2 = fig.add_subplot(gs[1, 0])  # Norms (bottom left)
            ax3 = fig.add_subplot(gs[1, 1])  # Similarity (bottom right)
        elif show_norms or show_similarity:
            fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 0.7))
            gs = fig.add_gridspec(2, 1, hspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Main heatmap (top)
            if show_norms:
                ax2 = fig.add_subplot(gs[1, :])  # Norms (bottom)
                ax3 = None
            else:
                ax2 = None
                ax3 = fig.add_subplot(gs[1, :])  # Similarity (bottom)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.5))
            ax2 = None
            ax3 = None
        
        # 1. Main heatmap: latent values over sequence
        im1 = ax1.imshow(latents_np.T, cmap=cmap, aspect='auto', interpolation='nearest')
        ax1.set_title(f'{title}\nLatent Values Over Sequence', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sequence Position')
        ax1.set_ylabel('Embedding Dimension')
        ax1.set_xticks(range(0, L, max(1, L // 10)))  # Show ~10 tick marks
        ax1.set_xticklabels(range(0, L, max(1, L // 10)))
        plt.colorbar(im1, ax=ax1, label='Latent Value')
        
        # 2. Embedding norms over sequence
        if show_norms and ax2 is not None:
            norms = np.linalg.norm(latents_np, axis=1)
            ax2.plot(range(L), norms, 'b-o', linewidth=2, markersize=4, alpha=0.8)
            ax2.set_title('Embedding Norms Over Sequence', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Sequence Position')
            ax2.set_ylabel('L2 Norm')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(0, L, max(1, L // 10)))
            ax2.set_xticklabels(range(0, L, max(1, L // 10)))
            
            # Add statistics
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            ax2.axhline(mean_norm, color='r', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_norm:.3f}')
            ax2.fill_between(range(L), mean_norm - std_norm, mean_norm + std_norm, 
                           alpha=0.2, color='r', label=f'±1σ: {std_norm:.3f}')
            ax2.legend()
        
        # 3. Similarity between consecutive latents
        if show_similarity and ax3 is not None:
            # Normalize latents for cosine similarity
            normalized_latents = latents_np / np.linalg.norm(latents_np, axis=1, keepdims=True)
            
            # Compute cosine similarity between consecutive positions
            similarities = []
            for i in range(L - 1):
                sim = np.dot(normalized_latents[i], normalized_latents[i + 1])
                similarities.append(sim)
            
            ax3.plot(range(1, L), similarities, 'g-o', linewidth=2, markersize=4, alpha=0.8)
            ax3.set_title('Cosine Similarity Between Consecutive Latents', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Sequence Position')
            ax3.set_ylabel('Cosine Similarity')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(0, L, max(1, L // 10)))
            ax3.set_xticklabels(range(0, L, max(1, L // 10)))
            
            # Add statistics
            mean_sim = np.mean(similarities)
            ax3.axhline(mean_sim, color='r', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_sim:.3f}')
            ax3.axhline(1.0, color='k', linestyle=':', alpha=0.5, label='Perfect similarity')
            ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latent sequence visualization saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\nLatent Sequence Statistics:")
        print(f"  Sequence length: {L}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Mean latent value: {np.mean(latents_np):.4f}")
        print(f"  Std latent value: {np.std(latents_np):.4f}")
        print(f"  Min latent value: {np.min(latents_np):.4f}")
        print(f"  Max latent value: {np.max(latents_np):.4f}")
        
        if show_norms:
            norms = np.linalg.norm(latents_np, axis=1)
            print(f"  Mean embedding norm: {np.mean(norms):.4f}")
            print(f"  Std embedding norm: {np.std(norms):.4f}")
        
        if show_similarity:
            normalized_latents = latents_np / np.linalg.norm(latents_np, axis=1, keepdims=True)
            similarities = [np.dot(normalized_latents[i], normalized_latents[i + 1]) 
                           for i in range(L - 1)]
            print(f"  Mean consecutive similarity: {np.mean(similarities):.4f}")
            print(f"  Min consecutive similarity: {np.min(similarities):.4f}")
            print(f"  Max consecutive similarity: {np.max(similarities):.4f}")
    
    def visualize_cot_latent_alignment(self, original_tokens: List[str], encoded_latents: torch.Tensor, 
                                     reconstructed_tokens: List[str], 
                                     latent_indices: Optional[torch.Tensor] = None,
                                     title: str = "CoT-Latent-Reconstruction Alignment",
                                     save_path: Optional[str] = None,
                                     figsize: Tuple[int, int] = (16, 12),
                                     tokens_per_chunk: int = 10,
                                     max_chunks: Optional[int] = None,
                                     cmap: str = 'viridis',
                                     show_token_boundaries: bool = True,
                                     show_latent_stats: bool = True,
                                     create_folder: bool = True) -> None:
        """
        Visualize the alignment between original CoT tokens, encoded latents, and reconstructed CoT tokens.
        
        Args:
            original_tokens: List of original CoT tokens (length L)
            encoded_latents: Tensor of shape (L, embedding_dim) where L is sequence length
            reconstructed_tokens: List of reconstructed CoT tokens (length L)
            latent_indices: Optional tensor of latent indices
            title: Title for the visualization
            save_path: Optional path to save the figure(s)
            figsize: Figure size (width, height)
            tokens_per_chunk: Number of tokens per chunk
            max_chunks: Maximum number of chunks to display (None for all chunks)
            cmap: Colormap for the latent heatmap
            show_token_boundaries: Whether to show token boundaries in text
            show_latent_stats: Whether to show latent statistics for each chunk
            create_folder: Whether to create a folder and save multiple images for all chunks
        """
        if encoded_latents.dim() != 2:
            raise ValueError(f"Expected 2D tensor for latents, got shape {encoded_latents.shape}")
        
        L, embedding_dim = encoded_latents.shape
        
        # Handle token length mismatches by padding
        def pad_tokens_to_length(tokens: List[str], target_length: int, pad_token: str = "[PAD]") -> List[str]:
            """Pad or truncate tokens to match target length."""
            if len(tokens) < target_length:
                # Pad with pad_token
                return tokens + [pad_token] * (target_length - len(tokens))
            elif len(tokens) > target_length:
                # Truncate
                return tokens[:target_length]
            else:
                return tokens
        
        # Pad/truncate tokens to match latent sequence length
        original_tokens = pad_tokens_to_length(original_tokens, L)
        reconstructed_tokens = pad_tokens_to_length(reconstructed_tokens, L)
        
        print(f"Token length adjustment:")
        print(f"  Original tokens adjusted to: {len(original_tokens)}")
        print(f"  Reconstructed tokens adjusted to: {len(reconstructed_tokens)}")
        print(f"  Latent sequence length: {L}")
        
        print(f"Visualizing CoT-latent alignment:")
        print(f"  Original tokens: {len(original_tokens)}")
        print(f"  Latent sequence length: {L}")
        print(f"  Reconstructed tokens: {len(reconstructed_tokens)}")
        if latent_indices is not None:
            print(f"  Latent indices provided: {latent_indices.shape}")
        else:
            print(f"  No latent indices provided")
        
        # Convert latents to numpy
        latents_np = encoded_latents.detach().cpu().numpy()
        
        # Split tokens into chunks
        def split_tokens_into_chunks(tokens: List[str], chunk_size: int) -> List[List[str]]:
            """Split tokens into chunks of specified size."""
            return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        
        # Split both token lists into chunks
        original_chunks = split_tokens_into_chunks(original_tokens, tokens_per_chunk)
        reconstructed_chunks = split_tokens_into_chunks(reconstructed_tokens, tokens_per_chunk)
        
        # Determine number of chunks to process
        if max_chunks is None:
            # Process all chunks
            num_chunks = min(len(original_chunks), len(reconstructed_chunks))
        else:
            # Limit to specified number of chunks
            num_chunks = min(len(original_chunks), len(reconstructed_chunks), max_chunks)
        
        original_chunks = original_chunks[:num_chunks]
        reconstructed_chunks = reconstructed_chunks[:num_chunks]
        
        # Calculate latent chunks and indices (exact alignment with tokens)
        latent_chunks = []
        latent_indices_chunks = []
        latent_stats = []
        
        for i in range(num_chunks):
            # Calculate exact start and end positions for this chunk
            start_pos = i * tokens_per_chunk
            end_pos = min(start_pos + tokens_per_chunk, L)
            
            chunk_latents = latents_np[start_pos:end_pos]
            latent_chunks.append(chunk_latents)
            
            # Get corresponding indices if provided
            if latent_indices is not None:
                chunk_indices = latent_indices[start_pos:end_pos].cpu().numpy()
                latent_indices_chunks.append(chunk_indices)
            else:
                latent_indices_chunks.append(None)
            
            # Calculate statistics for this chunk
            if show_latent_stats and len(chunk_latents) > 0:
                chunk_stats = {
                    'mean_norm': np.mean(np.linalg.norm(chunk_latents, axis=1)),
                    'std_norm': np.std(np.linalg.norm(chunk_latents, axis=1)),
                    'mean_value': np.mean(chunk_latents),
                    'std_value': np.std(chunk_latents),
                    'min_value': np.min(chunk_latents),
                    'max_value': np.max(chunk_latents)
                }
                latent_stats.append(chunk_stats)
        
        # Determine save strategy
        if create_folder and save_path is not None:
            # Create folder for multiple images
            base_path = save_path.rsplit('.', 1)[0]  # Remove extension
            folder_path = f"{base_path}_chunks"
            os.makedirs(folder_path, exist_ok=True)
            print(f"Creating chunk visualizations in folder: {folder_path}")
            
            # Save individual chunk images
            for chunk_idx in range(num_chunks):
                chunk_save_path = os.path.join(folder_path, f"chunk_{chunk_idx + 1:03d}.png")
                self._create_single_chunk_visualization(
                    chunk_idx=chunk_idx,
                    latent_chunks=latent_chunks,
                    latent_indices_chunks=latent_indices_chunks,
                    latent_stats=latent_stats,
                    original_tokens=original_tokens,
                    reconstructed_tokens=reconstructed_tokens,
                    tokens_per_chunk=tokens_per_chunk,
                    title=f"{title} - Chunk {chunk_idx + 1}/{num_chunks}",
                    save_path=chunk_save_path,
                    figsize=figsize,
                    cmap=cmap,
                    show_latent_stats=show_latent_stats
                )
            
            # Also create a combined visualization if there are multiple chunks
            if num_chunks > 1:
                combined_save_path = os.path.join(folder_path, "combined_all_chunks.png")
                self._create_combined_visualization(
                    num_chunks=num_chunks,
                    latent_chunks=latent_chunks,
                    latent_indices_chunks=latent_indices_chunks,
                    latent_stats=latent_stats,
                    original_tokens=original_tokens,
                    reconstructed_tokens=reconstructed_tokens,
                    tokens_per_chunk=tokens_per_chunk,
                    title=title,
                    save_path=combined_save_path,
                    figsize=figsize,
                    cmap=cmap,
                    show_latent_stats=show_latent_stats
                )
            
            print(f"Created {num_chunks} individual chunk visualizations + 1 combined visualization")
            
        else:
            # Create single combined visualization (original behavior)
            self._create_combined_visualization(
                num_chunks=num_chunks,
                latent_chunks=latent_chunks,
                latent_indices_chunks=latent_indices_chunks,
                latent_stats=latent_stats,
                original_tokens=original_tokens,
                reconstructed_tokens=reconstructed_tokens,
                tokens_per_chunk=tokens_per_chunk,
                title=title,
                save_path=save_path,
                figsize=figsize,
                cmap=cmap,
                show_latent_stats=show_latent_stats
            )
        
        # Print summary statistics
        print(f"\nCoT-Latent Alignment Summary:")
        print(f"  Number of chunks displayed: {num_chunks}")
        print(f"  Original token chunks: {len(original_chunks)}")
        print(f"  Reconstructed token chunks: {len(reconstructed_chunks)}")
        print(f"  Total latent sequence length: {L}")
        print(f"  Tokens per chunk: {tokens_per_chunk}")
        
        # Calculate reconstruction quality metrics
        if len(original_tokens) > 0 and len(reconstructed_tokens) > 0:
            # Token-level accuracy
            correct_tokens = sum(1 for orig, recon in zip(original_tokens, reconstructed_tokens) if orig == recon)
            token_accuracy = correct_tokens / len(original_tokens)
            print(f"  Token-level accuracy: {token_accuracy:.3f} ({correct_tokens}/{len(original_tokens)})")
        
        # Overall latent statistics
        overall_mean_norm = np.mean(np.linalg.norm(latents_np, axis=1))
        overall_std_norm = np.std(np.linalg.norm(latents_np, axis=1))
        print(f"  Overall mean latent norm: {overall_mean_norm:.3f} ± {overall_std_norm:.3f}")
    
    def _create_single_chunk_visualization(self, chunk_idx: int, latent_chunks: List[np.ndarray],
                                         latent_indices_chunks: List[Optional[np.ndarray]],
                                         latent_stats: List[Dict[str, float]],
                                         original_tokens: List[str], reconstructed_tokens: List[str],
                                         tokens_per_chunk: int, title: str, save_path: str,
                                         figsize: Tuple[int, int], cmap: str, show_latent_stats: bool) -> None:
        """
        Create visualization for a single chunk.
        """
        chunk_latents = latent_chunks[chunk_idx]
        
        # Calculate subplot layout
        rows = 1
        if show_latent_stats and chunk_idx < len(latent_stats):
            rows += 1
        
        _, axes = plt.subplots(rows, 1, figsize=(figsize[0], figsize[1] * 0.4 * rows))
        if rows == 1:
            axes = [axes]
        
        if len(chunk_latents) > 0:
            # Get corresponding tokens for this chunk
            start_token_idx = chunk_idx * tokens_per_chunk
            end_token_idx = min(start_token_idx + tokens_per_chunk, len(original_tokens))
            chunk_original_tokens = original_tokens[start_token_idx:end_token_idx]
            chunk_reconstructed_tokens = reconstructed_tokens[start_token_idx:end_token_idx]
            chunk_indices = latent_indices_chunks[chunk_idx]
            
            # Create the heatmap
            im = axes[0].imshow(chunk_latents.T, cmap=cmap, aspect='auto', interpolation='nearest')
            axes[0].set_title(f'{title})', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Token Position')
            axes[0].set_ylabel('Embedding Dimension')
            
            # Add colorbar for this chunk
            cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
            cbar.set_label('Latent Value')
            
            # Add original token labels at the top
            for i, token in enumerate(chunk_original_tokens):
                axes[0].text(i, chunk_latents.shape[1] - 0.5, token, ha='center', va='bottom', 
                          fontsize=8, fontweight='bold', color='blue',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
            
            # Add latent indices in the middle (if provided)
            if chunk_indices is not None:
                for i, idx in enumerate(chunk_indices):
                    mid_y = chunk_latents.shape[1] / 2
                    axes[0].text(i, mid_y, f'[{idx}]', ha='center', va='center', 
                              fontsize=7, fontweight='bold', color='white',
                              bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7))
            
            # Add reconstructed token labels at the bottom
            for i, token in enumerate(chunk_reconstructed_tokens):
                axes[0].text(i, 0.5, token, ha='center', va='top', 
                          fontsize=8, fontweight='bold', color='black',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.8))
            
            # Adjust y-axis limits to accommodate labels
            axes[0].set_ylim(-1, chunk_latents.shape[1] + 1)
            
            # Show position ticks
            if len(chunk_latents) <= 15:
                axes[0].set_xticks(range(len(chunk_latents)))
                axes[0].set_xticklabels(range(start_token_idx, end_token_idx))
            else:
                tick_positions = range(start_token_idx, end_token_idx, max(1, len(chunk_latents) // 5))
                axes[0].set_xticks(tick_positions)
                axes[0].set_xticklabels(tick_positions)
            
            # Add legend for token types
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', alpha=0.8, label='Original Tokens'),
                Patch(facecolor='black', alpha=0.7, label='Latent Indices'),
                Patch(facecolor='lightcoral', alpha=0.8, label='Reconstructed Tokens')
            ]
            axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)
            
        else:
            axes[0].text(0.5, 0.5, 'No latent data for this chunk', 
                      transform=axes[0].transAxes, ha='center', va='center',
                      fontsize=12, style='italic')
            axes[0].axis('off')
        
        # Add latent statistics (optional)
        if show_latent_stats and chunk_idx < len(latent_stats) and rows > 1:
            stats = latent_stats[chunk_idx]
            
            stats_text = f"""Latent Statistics:
• Mean norm: {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}
• Mean value: {stats['mean_value']:.3f} ± {stats['std_value']:.3f}
• Value range: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]
• Sequence length: {len(chunk_latents)}"""
            
            axes[1].text(0.05, 0.5, stats_text, transform=axes[1].transAxes, 
                        fontsize=9, verticalalignment='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(0, 1)
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_combined_visualization(self, num_chunks: int, latent_chunks: List[np.ndarray],
                                     latent_indices_chunks: List[Optional[np.ndarray]],
                                     latent_stats: List[Dict[str, float]],
                                     original_tokens: List[str], reconstructed_tokens: List[str],
                                     tokens_per_chunk: int, title: str, save_path: Optional[str],
                                     figsize: Tuple[int, int], cmap: str, show_latent_stats: bool) -> None:
        """
        Create combined visualization with all chunks in one figure.
        """
        fig = plt.figure(figsize=figsize)
        
        # Calculate subplot layout
        rows_per_chunk = 1  # just the latent heatmap
        if show_latent_stats:
            rows_per_chunk += 1  # add stats row
        
        total_rows = num_chunks * rows_per_chunk
        gs = fig.add_gridspec(total_rows, 1, hspace=0.4, wspace=0.1)
        
        for chunk_idx in range(num_chunks):
            row_start = chunk_idx * rows_per_chunk
            
            # 1. Latent representation with token labels
            ax_latents = fig.add_subplot(gs[row_start, 0])
            chunk_latents = latent_chunks[chunk_idx]
            
            if len(chunk_latents) > 0:
                # Get corresponding tokens for this chunk
                start_token_idx = chunk_idx * tokens_per_chunk
                end_token_idx = min(start_token_idx + tokens_per_chunk, len(original_tokens))
                chunk_original_tokens = original_tokens[start_token_idx:end_token_idx]
                chunk_reconstructed_tokens = reconstructed_tokens[start_token_idx:end_token_idx]
                chunk_indices = latent_indices_chunks[chunk_idx]
                
                # Create the heatmap
                im = ax_latents.imshow(chunk_latents.T, cmap=cmap, aspect='auto', interpolation='nearest')
                ax_latents.set_title(f'CoT Chunk {chunk_idx + 1} - Latent Alignment (Shape: {chunk_latents.shape})', 
                                   fontsize=12, fontweight='bold')
                ax_latents.set_xlabel('Token Position')
                ax_latents.set_ylabel('Embedding Dimension')
                
                # Add colorbar for this chunk
                cbar = plt.colorbar(im, ax=ax_latents, shrink=0.8)
                cbar.set_label('Latent Value')
                
                # Add original token labels at the top
                for i, token in enumerate(chunk_original_tokens):
                    ax_latents.text(i, -0.5, token, ha='center', va='bottom', 
                                  fontsize=8, fontweight='bold', color='blue',
                                  bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
                
                # Add latent indices in the middle (if provided)
                if chunk_indices is not None:
                    for i, idx in enumerate(chunk_indices):
                        mid_y = chunk_latents.shape[1] / 2
                        ax_latents.text(i, mid_y, f'[{idx}]', ha='center', va='center', 
                                      fontsize=7, fontweight='bold', color='white',
                                      bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7))
                
                # Add reconstructed token labels at the bottom
                for i, token in enumerate(chunk_reconstructed_tokens):
                    ax_latents.text(i, chunk_latents.shape[1] + 0.5, token, ha='center', va='top', 
                                  fontsize=8, fontweight='bold', color='black',
                                  bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.8))
                
                # Adjust y-axis limits to accommodate labels
                ax_latents.set_ylim(-1, chunk_latents.shape[1] + 1)
                
                # Show position ticks
                if len(chunk_latents) <= 15:
                    ax_latents.set_xticks(range(len(chunk_latents)))
                    ax_latents.set_xticklabels(range(len(chunk_latents)))
                else:
                    tick_positions = range(0, len(chunk_latents), max(1, len(chunk_latents) // 5))
                    ax_latents.set_xticks(tick_positions)
                    ax_latents.set_xticklabels(tick_positions)
                
                # Add legend for token types
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='lightblue', alpha=0.8, label='Original Tokens'),
                    Patch(facecolor='black', alpha=0.7, label='Latent Indices'),
                    Patch(facecolor='lightcoral', alpha=0.8, label='Reconstructed Tokens')
                ]
                ax_latents.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
            else:
                ax_latents.text(0.5, 0.5, 'No latent data for this chunk', 
                              transform=ax_latents.transAxes, ha='center', va='center',
                              fontsize=12, style='italic')
                ax_latents.axis('off')
            
            # 2. Latent statistics (optional)
            if show_latent_stats and chunk_idx < len(latent_stats):
                ax_stats = fig.add_subplot(gs[row_start + 1, 0])
                stats = latent_stats[chunk_idx]
                
                stats_text = f"""Latent Statistics (Chunk {chunk_idx + 1}):
• Mean norm: {stats['mean_norm']:.3f} ± {stats['std_norm']:.3f}
• Mean value: {stats['mean_value']:.3f} ± {stats['std_value']:.3f}
• Value range: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]
• Sequence length: {len(chunk_latents)}"""
                
                ax_stats.text(0.05, 0.5, stats_text, transform=ax_stats.transAxes, 
                            fontsize=9, verticalalignment='center', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
                ax_stats.set_xlim(0, 1)
                ax_stats.set_ylim(0, 1)
                ax_stats.axis('off')
        
        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CoT-latent alignment visualization saved to: {save_path}")
        
        plt.show()

    def visualize_chain_embeddings(self, title: str = "Chain Embeddings Visualization",
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 8),
                                  cmap: str = 'viridis',
                                  show_stats: bool = True) -> None:
        """
        Visualize the chain embeddings learned by the model in a heatmap style.
        
        Args:
            title: Title for the visualization
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
            cmap: Colormap for the heatmap
            show_stats: Whether to show embedding statistics
        """
        if not hasattr(self.model, 'chain_embeddings'):
            print("Warning: Model does not have chain_embeddings attribute")
            return
        
        # Get chain embeddings
        chain_embeddings = self.model.chain_embeddings.weight.detach().cpu().numpy()
        num_chains, embedding_dim = chain_embeddings.shape
        
        print(f"Visualizing chain embeddings:")
        print(f"  Number of chains: {num_chains}")
        print(f"  Embedding dimension: {embedding_dim}")
        
        # Create figure with subplots
        if show_stats:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Main heatmap (top, full width)
            ax2 = fig.add_subplot(gs[1, 0])  # Norms (bottom left)
            ax3 = fig.add_subplot(gs[1, 1])  # Similarity (bottom right)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.6))
            ax2 = None
            ax3 = None
        
        # 1. Main heatmap: chain embeddings
        im1 = ax1.imshow(chain_embeddings.T, cmap=cmap, aspect='auto', interpolation='nearest')
        ax1.set_title(f'{title}\nChain Embeddings Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Chain Index')
        ax1.set_ylabel('Embedding Dimension')
        ax1.set_xticks(range(num_chains))
        ax1.set_xticklabels([f'Chain {i}' for i in range(num_chains)])
        ax1.set_yticks(range(0, embedding_dim, max(1, embedding_dim // 10)))
        ax1.set_yticklabels(range(0, embedding_dim, max(1, embedding_dim // 10)))
        plt.colorbar(im1, ax=ax1, label='Embedding Value')
        
        # 2. Chain embedding norms
        if show_stats and ax2 is not None:
            norms = np.linalg.norm(chain_embeddings, axis=1)
            ax2.bar(range(num_chains), norms, color='skyblue', alpha=0.8)
            ax2.set_title('Chain Embedding Norms', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Chain Index')
            ax2.set_ylabel('L2 Norm')
            ax2.set_xticks(range(num_chains))
            ax2.set_xticklabels([f'Chain {i}' for i in range(num_chains)])
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            ax2.axhline(mean_norm, color='r', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_norm:.3f}')
            ax2.fill_between(range(num_chains), mean_norm - std_norm, mean_norm + std_norm, 
                           alpha=0.2, color='r', label=f'±1σ: {std_norm:.3f}')
            ax2.legend()
        
        # 3. Similarity between chain embeddings
        if show_stats and ax3 is not None and num_chains > 1:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = chain_embeddings / np.linalg.norm(chain_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity matrix
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            im3 = ax3.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax3.set_title('Chain Embedding Similarity', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Chain Index')
            ax3.set_ylabel('Chain Index')
            ax3.set_xticks(range(num_chains))
            ax3.set_xticklabels([f'Chain {i}' for i in range(num_chains)])
            ax3.set_yticks(range(num_chains))
            ax3.set_yticklabels([f'Chain {i}' for i in range(num_chains)])
            plt.colorbar(im3, ax=ax3, label='Cosine Similarity')
            
            # Add similarity values on the heatmap
            for i in range(num_chains):
                for j in range(num_chains):
                    text = ax3.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="white", fontsize=8,
                                  fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chain embeddings visualization saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\nChain Embeddings Statistics:")
        print(f"  Number of chains: {num_chains}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Mean embedding value: {np.mean(chain_embeddings):.4f}")
        print(f"  Std embedding value: {np.std(chain_embeddings):.4f}")
        print(f"  Min embedding value: {np.min(chain_embeddings):.4f}")
        print(f"  Max embedding value: {np.max(chain_embeddings):.4f}")
        
        if show_stats:
            norms = np.linalg.norm(chain_embeddings, axis=1)
            print(f"  Mean chain norm: {np.mean(norms):.4f}")
            print(f"  Std chain norm: {np.std(norms):.4f}")
            print(f"  Min chain norm: {np.min(norms):.4f}")
            print(f"  Max chain norm: {np.max(norms):.4f}")
            
            if num_chains > 1:
                # Normalize embeddings for cosine similarity
                normalized_embeddings = chain_embeddings / np.linalg.norm(chain_embeddings, axis=1, keepdims=True)
                similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
                
                # Get off-diagonal similarities (excluding self-similarity)
                off_diagonal_similarities = []
                for i in range(num_chains):
                    for j in range(num_chains):
                        if i != j:
                            off_diagonal_similarities.append(similarity_matrix[i, j])
                
                if off_diagonal_similarities:
                    print(f"  Mean inter-chain similarity: {np.mean(off_diagonal_similarities):.4f}")
                    print(f"  Min inter-chain similarity: {np.min(off_diagonal_similarities):.4f}")
                    print(f"  Max inter-chain similarity: {np.max(off_diagonal_similarities):.4f}")
                    
                    # Find most similar and most different chain pairs
                    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix + np.eye(num_chains) * -2), similarity_matrix.shape)
                    min_sim_idx = np.unravel_index(np.argmin(similarity_matrix + np.eye(num_chains) * 2), similarity_matrix.shape)
                    
                    print(f"  Most similar chain pair: Chain {max_sim_idx[0]} ↔ Chain {max_sim_idx[1]} (sim: {similarity_matrix[max_sim_idx]:.4f})")
                    print(f"  Most different chain pair: Chain {min_sim_idx[0]} ↔ Chain {min_sim_idx[1]} (sim: {similarity_matrix[min_sim_idx]:.4f})")

    def run_full_analysis(self, data_dir: str, output_dir: str, 
                         max_samples: Optional[int] = None, 
                         sample_size: Optional[int] = None) -> None:
        """
        Run the complete analysis pipeline.
        
        Args:
            data_dir: Directory containing the dataset
            output_dir: Directory to save all visualizations and analysis
            max_samples: Maximum samples to load from dataset
            sample_size: Maximum samples to analyze (for memory constraints)
        """
        print("Starting full latent visualization analysis...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        prompt_sequences, cot_sequences, prompt_mask, cot_mask = self.load_dataset(data_dir, max_samples)
        
        # Extract code usage data
        code_usage_data = self.extract_code_usage_data(
            prompt_sequences, cot_sequences, prompt_mask, cot_mask, sample_size
        )
        
        # Run analyses
        print("\n1. Analyzing code semantics...")
        self.analyze_code_semantics(code_usage_data, output_dir)
        
        print("\n2. Analyzing sequence patterns...")
        self.analyze_sequence_patterns(code_usage_data, output_dir)
        
        print("\n3. Creating comprehensive dashboard...")
        self.create_comprehensive_visualization(code_usage_data, output_dir)
        
        print("\n4. Visualizing chain embeddings...")
        chain_embeddings_path = os.path.join(output_dir, 'chain_embeddings_visualization.png')
        self.visualize_chain_embeddings(
            title="Trained Model Chain Embeddings",
            save_path=chain_embeddings_path,
            show_stats=True
        )
        
        print("\n5. Analyzing word-to-latent mapping...")
        self.analyze_word_to_latent_mapping(
            prompt_sequences=prompt_sequences,
            cot_sequences=cot_sequences,
            prompt_mask=prompt_mask,
            cot_mask=cot_mask,
            output_dir=output_dir,
            sample_size=sample_size,
            top_k_words=20,
            top_k_codes=30
        )
        
        # Save raw data for further analysis
        print("\n6. Saving analysis data...")
        analysis_data_file = os.path.join(output_dir, 'analysis_data.json')
        
        # Convert data to JSON-serializable format
        serializable_data = {
            'code_usage_counts': dict(code_usage_data['code_usage_counts']),
            'total_samples': code_usage_data['total_samples'],
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'num_thoughts': self.num_thoughts
        }
        
        with open(analysis_data_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Analysis complete! Results saved to: {output_dir}")
        print(f"Generated files:")
        print(f"  - comprehensive_dashboard.png")
        print(f"  - code_semantics_analysis.png")
        print(f"  - sequence_patterns_analysis.png")
        print(f"  - start_end_frequencies.png")
        print(f"  - chain_embeddings_visualization.png")
        print(f"  - word_to_latent_mapping.png")
        print(f"  - detailed_code_analysis.txt")
        print(f"  - sequence_statistics.txt")
        print(f"  - word_mapping_analysis.txt")
        print(f"  - analysis_data.json")

    def analyze_word_to_latent_mapping(self, prompt_sequences: torch.Tensor, cot_sequences: torch.Tensor,
                                      prompt_mask: torch.Tensor, cot_mask: torch.Tensor,
                                      output_dir: str, sample_size: Optional[int] = None,
                                      top_k_words: int = 20, top_k_codes: int = 30) -> None:
        """
        Analyze what words/tokens are mapped to each latent embedding and show their distribution.
        
        Args:
            prompt_sequences: Prompt sequences tensor
            cot_sequences: CoT sequences tensor
            prompt_mask: Prompt mask tensor
            cot_mask: CoT mask tensor
            output_dir: Directory to save visualizations
            sample_size: Number of samples to process (None for all)
            top_k_words: Number of top words to show per code
            top_k_codes: Number of top codes to analyze in detail
        """
        print("Analyzing word-to-latent mapping...")
        
        # Determine sample size
        total_samples = len(prompt_sequences)
        if sample_size is None:
            sample_size = total_samples
        else:
            sample_size = min(sample_size, total_samples)
        
        # Initialize data structures
        code_to_words = defaultdict(lambda: defaultdict(int))  # code -> {word -> count}
        
        # Process samples
        with torch.no_grad():
            for i in range(sample_size):
                if i % 100 == 0:
                    print(f"Processing sample {i}/{sample_size}")
                
                # Prepare single example
                prompt = prompt_sequences[i:i+1].to(self.device)
                cot_gt = cot_sequences[i:i+1].to(self.device)
                prompt_mask_ex = prompt_mask[i:i+1].to(self.device) if prompt_mask is not None else None
                cot_mask_ex = cot_mask[i:i+1].to(self.device) if cot_mask is not None else None
                
                # Get encoding indices
                try:
                    _, _, _, indices, _ = self.model.encode(
                        prompt, cot_gt, prompt_mask_ex, cot_mask_ex, 
                        quantize_cot_only=True
                    )
                    
                    # Get CoT tokens for this sample
                    cot_tokens = cot_gt[0, 0]  # [seq_len]
                    cot_mask_sample = cot_mask_ex[0, 0] if cot_mask_ex is not None else None
                    
                    # Apply mask if available
                    if cot_mask_sample is not None:
                        valid_positions = cot_mask_sample.bool()
                        cot_tokens = cot_tokens[valid_positions]
                        indices = indices[0][valid_positions]
                    else:
                        indices = indices[0]
                    
                    # Convert tokens to words
                    words = self.tokenizer.convert_ids_to_tokens(cot_tokens.tolist())
                    
                    # Map codes to words and token IDs
                    for pos, (code, word, token_id) in enumerate(zip(indices, words, cot_tokens)):
                        code_item = code.item()
                        code_to_words[code_item][word] += 1
                
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
        
        # Compile statistics
        code_usage_counts = Counter()
        for code in code_to_words.keys():
            code_usage_counts[code] = sum(code_to_words[code].values())
        
        # Get top used codes
        top_codes = sorted(code_usage_counts.items(), key=lambda x: x[1], reverse=True)[:top_k_codes]
        
        # Create visualizations
        self._create_word_mapping_visualizations(
            code_to_words, top_codes, output_dir, top_k_words
        )
        
        # Save detailed word mapping analysis
        self._save_word_mapping_analysis(
            code_to_words, top_codes, output_dir, top_k_words
        )
        
        print(f"Word-to-latent mapping analysis completed. Results saved to: {output_dir}")
    
    def _create_word_mapping_visualizations(self, code_to_words: Dict[int, Dict[str, int]],
                                          top_codes: List[Tuple[int, int]], output_dir: str,
                                          top_k_words: int) -> None:
        """
        Create visualizations for word-to-latent mapping analysis.
        """
        print("Creating word mapping visualizations...")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 20))
        
        # Grid layout: 3 rows, 3 columns
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. Top word distribution across codes (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        all_words = set()
        for code in code_to_words:
            all_words.update(code_to_words[code].keys())
        
        # Get top words overall
        word_total_counts = Counter()
        for code in code_to_words:
            word_total_counts.update(code_to_words[code])
        
        top_words_overall = word_total_counts.most_common(top_k_words)
        words, counts = zip(*top_words_overall)
        
        ax1.barh(range(len(words)), counts, color='skyblue', alpha=0.8)
        ax1.set_title('Most Common Words Across All Codes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Total Count')
        ax1.set_ylabel('Word')
        ax1.set_yticks(range(len(words)))
        ax1.set_yticklabels(words)
        ax1.grid(True, alpha=0.3)
        
        # 2. Code diversity (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        code_diversity = [len(code_to_words[code]) for code, _ in top_codes[:20]]
        codes = [code for code, _ in top_codes[:20]]
        
        ax2.bar(range(len(codes)), code_diversity, color='lightcoral', alpha=0.8)
        ax2.set_title('Number of Unique Words per Code', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Code Index')
        ax2.set_ylabel('Number of Unique Words')
        ax2.set_xticks(range(len(codes)))
        ax2.set_xticklabels(codes, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Usage vs diversity scatter (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        usage_counts = [count for _, count in top_codes[:20]]
        
        ax3.scatter(usage_counts, code_diversity, alpha=0.7, s=100, color='green')
        ax3.set_title('Code Usage vs Word Diversity', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Total Usage Count')
        ax3.set_ylabel('Number of Unique Words')
        
        # Add code labels
        for i, code in enumerate(codes):
            ax3.annotate(str(code), (usage_counts[i], code_diversity[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Word distribution for top codes (middle row, spanning all columns)
        ax4 = fig.add_subplot(gs[1, :])
        
        # Create heatmap of word distribution for top codes
        top_codes_for_heatmap = top_codes[:15]  # Limit to 15 codes for readability
        top_words_for_heatmap = [word for word, _ in word_total_counts.most_common(20)]
        
        heatmap_data = np.zeros((len(top_words_for_heatmap), len(top_codes_for_heatmap)))
        
        for i, word in enumerate(top_words_for_heatmap):
            for j, (code, _) in enumerate(top_codes_for_heatmap):
                heatmap_data[i, j] = code_to_words[code].get(word, 0)
        
        im = ax4.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
        ax4.set_title('Word Distribution Across Top Codes', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Code Index')
        ax4.set_ylabel('Word')
        ax4.set_xticks(range(len(top_codes_for_heatmap)))
        ax4.set_xticklabels([code for code, _ in top_codes_for_heatmap])
        ax4.set_yticks(range(len(top_words_for_heatmap)))
        ax4.set_yticklabels(top_words_for_heatmap)
        plt.colorbar(im, ax=ax4, label='Word Count')
        
        # 5. Detailed word breakdown for top 3 codes (bottom row)
        for i, (code, count) in enumerate(top_codes[:3]):
            ax = fig.add_subplot(gs[2, i])
            
            # Get top words for this code
            code_words = code_to_words[code]
            top_words = sorted(code_words.items(), key=lambda x: x[1], reverse=True)[:top_k_words]
            
            if top_words:
                words, counts = zip(*top_words)
                ax.barh(range(len(words)), counts, color=f'C{i}', alpha=0.8)
                ax.set_title(f'Code {code} (used {count} times)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Word Count')
                ax.set_ylabel('Word')
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No words found for Code {code}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, style='italic')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_to_latent_mapping.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Word mapping visualizations saved to: {output_dir}/word_to_latent_mapping.png")
    
    def _save_word_mapping_analysis(self, code_to_words: Dict[int, Dict[str, int]],
                                  top_codes: List[Tuple[int, int]], output_dir: str,
                                  top_k_words: int) -> None:
        """
        Save detailed word mapping analysis to file.
        """
        analysis_file = os.path.join(output_dir, 'word_mapping_analysis.txt')
        
        with open(analysis_file, 'w') as f:
            f.write("Word-to-Latent Mapping Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            total_codes = len(code_to_words)
            total_words = len(set().union(*[set(words.keys()) for words in code_to_words.values()]))
            total_mappings = sum(sum(words.values()) for words in code_to_words.values())
            
            f.write(f"Overall Statistics:\n")
            f.write(f"  Total codes used: {total_codes}\n")
            f.write(f"  Total unique words: {total_words}\n")
            f.write(f"  Total word-code mappings: {total_mappings}\n")
            f.write(f"  Average words per code: {total_mappings / total_codes:.2f}\n\n")
            
            # Detailed analysis for top codes
            for code, count in top_codes:
                f.write(f"Code {code} (used {count} times)\n")
                f.write("-" * 40 + "\n")
                
                # Word statistics
                words = code_to_words[code]
                unique_words = len(words)
                f.write(f"  Unique words: {unique_words}\n")
                f.write(f"  Average word frequency: {count / unique_words:.2f}\n\n")
                
                # Top words
                top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:top_k_words]
                f.write(f"  Top {len(top_words)} words:\n")
                for i, (word, word_count) in enumerate(top_words):
                    percentage = (word_count / count) * 100
                    f.write(f"    {i+1:2d}. '{word}': {word_count} times ({percentage:.1f}%)\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"Word mapping analysis saved to: {analysis_file}")


def load_model_from_checkpoint(checkpoint_path: str, model_config: Dict[str, Any], 
                             device: str = "cuda") -> EnhancedGPT2VQVAE:
    """
    Load Enhanced VQVAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration dictionary
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = EnhancedGPT2VQVAE(**model_config).to(device)
    
    # Load checkpoint
    model.load_checkpoint(checkpoint_path, device=device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def load_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (model_config, training_config)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")
    
    model_config = config.get('model_config', {})
    training_config = config.get('training_config', {})
    
    return model_config, training_config


def main():
    """Main function to run the latent visualization analysis."""
    parser = argparse.ArgumentParser(description='Latent Visualization Analysis for Enhanced VQVAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the model configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing the dataset files')
    parser.add_argument('--output_dir', type=str, default='latent_analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run analysis on (cuda/cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to load from dataset')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Maximum number of samples to analyze (for memory constraints)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        model_config, training_config = load_config(args.config)
        
        # Load model
        model = load_model_from_checkpoint(args.checkpoint, model_config, device)
        
        # Load tokenizer
        print("Loading GPT2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize analyzer
        analyzer = LatentVisualizationAnalyzer(model, tokenizer, device)
        
        # Run analysis
        analyzer.run_full_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            sample_size=args.sample_size
        )
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 