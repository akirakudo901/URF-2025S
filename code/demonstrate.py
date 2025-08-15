# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import gc
import heapq
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latent_visualization_analysis import LatentVisualizationAnalyzer
from trainer.train_utils import compute_reconstruction_loss, create_codebook_usage_heatmap, load_training_data
from trainer.gpt2_vqvae_trainer import GPUMemoryMonitor
from vqvae_gpt2 import GPT2VQVAE
from vqvae_gpt2_simple import SimpleGPT2VQVAE
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE

TRACK_BATCH_MEMORY_USE = True

def visualize_token_latent_alignment(original_token_ids, latent_indices, reconstructed_token_ids, tokenizer, 
                                   mode_name="", max_tokens_per_line=20):
    """
    Visualize the alignment between original tokens, latent indices, and reconstructed tokens.
    
    Args:
        original_token_ids: Tensor of original token IDs
        latent_indices: Tensor of corresponding latent indices
        reconstructed_token_ids: Tensor of reconstructed token IDs
        tokenizer: Tokenizer to convert IDs to text tokens
        mode_name: Name of the mode (e.g., "Teacher-forced", "Auto-regressive")
        max_tokens_per_line: Maximum number of tokens to show per line
    """
    try:
        # Convert token IDs to text tokens
        original_tokens = tokenizer.convert_ids_to_tokens(original_token_ids.tolist())
        reconstructed_tokens = tokenizer.convert_ids_to_tokens(reconstructed_token_ids.tolist())
        
        # Convert latent indices to strings
        latent_indices_str = [f"[{idx}]" for idx in latent_indices.tolist()]
        
        # Find the maximum length for alignment
        max_length = max(len(original_tokens), len(latent_indices_str), len(reconstructed_tokens))

        # Find maximum width for alignment across lines
        orig_max_width = max(len(s) for s in original_tokens)
        latent_max_width = max(len(s) for s in latent_indices_str)
        recon_max_width = max(len(s) for s in reconstructed_tokens)
        
        print(f"\n{'='*80}")
        print(f"TOKEN-LATENT ALIGNMENT {mode_name}")
        print(f"{'='*80}")
        print(f"Original | Latent Indices | Reconstructed")
        print(f"{'='*80}")
        
        # Process tokens in chunks
        for i in range(0, max_length, max_tokens_per_line):
            end_idx = min(i + max_tokens_per_line, max_length)
            
            # Get chunks for each sequence
            orig_chunk = original_tokens[i:end_idx] if i < len(original_tokens) else []
            latent_chunk = latent_indices_str[i:end_idx] if i < len(latent_indices_str) else []
            recon_chunk = reconstructed_tokens[i:end_idx] if i < len(reconstructed_tokens) else []
            
            # Pad chunks to same length
            chunk_length = max(len(orig_chunk), len(latent_chunk), len(recon_chunk))
            orig_chunk.extend([''] * (chunk_length - len(orig_chunk)))
            latent_chunk.extend([''] * (chunk_length - len(latent_chunk)))
            recon_chunk.extend([''] * (chunk_length - len(recon_chunk)))
            
            # Print aligned tokens    
            for j in range(chunk_length):
                orig_token = orig_chunk[j] if orig_chunk[j] else ''
                latent_token = latent_chunk[j] if latent_chunk[j] else ''
                recon_token = recon_chunk[j] if recon_chunk[j] else ''
                
                print(f"{orig_token.ljust(orig_max_width)} | {latent_token.ljust(latent_max_width)} | {recon_token.ljust(recon_max_width)}")
            
            if end_idx < max_length:
                print("-" * 80)
        
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error in visualize_token_latent_alignment: {e}")
        import traceback
        traceback.print_exc()


def visualize_token_latent_alignment_multi_cot(
    all_original_token_ids, all_latent_indices, all_reconstructed_token_ids, tokenizer, 
    mode_name="", max_tokens_per_line=20
):
    """
    Visualize the alignment for multiple CoTs side by side.
    all_original_token_ids, all_latent_indices, all_reconstructed_token_ids: lists of tensors, one per CoT
    """
    num_cots = len(all_original_token_ids)
    # Convert all to tokens/strings
    all_original_tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in all_original_token_ids]
    all_reconstructed_tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in all_reconstructed_token_ids]
    all_latent_strs = [[f"[{idx}]" for idx in indices.tolist()] for indices in all_latent_indices]
    # Find max length for each CoT
    max_lengths = [max(len(orig), len(latent), len(recon)) 
                   for orig, latent, recon in zip(all_original_tokens, all_latent_strs, all_reconstructed_tokens)]
    
    HEADER_ORIG = "CoTNN: Original"
    def header_orig(cot_num : int):
        return HEADER_ORIG.replace("NN", str(cot_num).ljust(2))
    HEADER_LATENT = "Latent"
    HEADER_RECON = "Recon"

    # Find max width for each column in each CoT
    orig_widths   = [max(len(s) for s in orig + [HEADER_ORIG])     
                     if orig   else 0 for orig in all_original_tokens]
    latent_widths = [max(len(s) for s in latent + [HEADER_LATENT]) 
                     if latent else 0 for latent in all_latent_strs]
    recon_widths  = [max(len(s) for s in recon + [HEADER_RECON]) 
                     if recon  else 0 for recon in all_reconstructed_tokens]
    
    # Print header
    print(f"\n{'='*120}")
    print(f"TOKEN-LATENT ALIGNMENT {mode_name} (All CoTs Side by Side)")
    print(f"{'='*120}")
    
    header = ""
    for c in range(num_cots):
        header += f"{header_orig(c+1)} | {HEADER_LATENT} | {HEADER_RECON}".ljust(
            orig_widths[c]+latent_widths[c]+recon_widths[c] + 6
        )
        if c < num_cots-1:
            header += " || "
    print(header)
    print(f"{'-'*120}")
    # Print rows
    for i in range(0, max(max_lengths), max_tokens_per_line):
        for row in range(i, min(i+max_tokens_per_line, max(max_lengths))):
            line = ""
            for c in range(num_cots):
                orig = all_original_tokens[c][row] if row < len(all_original_tokens[c]) else ""
                latent = all_latent_strs[c][row] if row < len(all_latent_strs[c]) else ""
                recon = all_reconstructed_tokens[c][row] if row < len(all_reconstructed_tokens[c]) else ""
                line += f"{orig.ljust(orig_widths[c])} | {latent.ljust(latent_widths[c])} | {recon.ljust(recon_widths[c])}"
                if c < num_cots-1:
                    line += " || "
            print(line)
        if i+max_tokens_per_line < max(max_lengths):
            print("-" * 120)
    print(f"{'='*120}")


def run_demonstration_on_split(model, tokenizer, num_examples, num_thoughts, num_embeddings, use_vq,
                              prompt_sequences, cot_sequences, prompt_mask, cot_mask, split_name, 
                              checkpoint_path, do_figure_analyses : bool=True, backpointers=None):
    print("\n" + "="*80)
    print(f"GENERATION DEMONSTRATION ON {split_name.upper()} DATA")
    print("="*80)
    all_indices_tf = []
    all_indices_ar = []

    device = next(model.parameters()).device

    with torch.no_grad():
        tf_mean_per_example_metric = {}
        ar_mean_per_example_metric = {}

        def decode_tokens(tokens, mask=None):
            if tokenizer is None:
                return f"[Tokens: {tokens.tolist()}]"
            if mask is not None:
                tokens = tokens[mask.bool()]
            try:
                return tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception as e:
                return f"[Decode error: {e}, tokens: {tokens.tolist()}]"

        def run_model_with_mode(
            inference_mode, 
            prompt, cot_gt, prompt_mask_ex, cot_mask_ex, backpointers_ex
        ):
            model_inputs = {
                'prompt': prompt,
                'cot_sequences': cot_gt,
                'cot_mask': cot_mask_ex,
                'prompt_mask': prompt_mask_ex,
                'inference': inference_mode,
                'quantize_cot_only': True
            }
            if backpointers_ex is not None:
                model_inputs['backpointers'] = backpointers_ex
            if hasattr(model, 'use_vq'):
                model_inputs['use_vq'] = use_vq
            else:
                model_inputs['no_vq'] = not use_vq
            out = model(**model_inputs)
            # Unpack outputs based on length
            if len(out) == 6:
                output_sequences, output_logits, vq_loss, perplexity, indices, debug_stats = out
                bp_logits = None
            elif len(out) == 7:
                output_sequences, output_logits, vq_loss, perplexity, indices, debug_stats, bp_logits = out
            else:
                output_sequences = output_logits = vq_loss = perplexity = indices = debug_stats = bp_logits = None
            return output_sequences, output_logits, vq_loss, perplexity, indices, debug_stats, bp_logits

        def process_mode(
            inference_mode, 
            prompt, cot_gt, prompt_mask_ex, cot_mask_ex, backpointers_ex, 
            num_thoughts, 
            indices_accumulator=None
        ):  
            try:
                output_sequences, output_logits, vq_loss, perplexity, indices, debug_stats, bp_logits = run_model_with_mode(
                    inference_mode, prompt, cot_gt, prompt_mask_ex, cot_mask_ex, backpointers_ex
                )
                if indices is not None and indices_accumulator is not None:
                    indices_accumulator.append(indices.flatten())
            except Exception as e:
                mode_str = "Teacher forcing" if not inference_mode else "Auto-regressive"
                print(f"{mode_str} generation failed: {e}")
                output_sequences = None
                output_logits = None
                vq_loss = None
                perplexity = None
                indices = None
                bp_logits = None

            cot_texts = []
            for j in range(num_thoughts):
                if output_sequences is not None:
                    cot_texts.append(
                        decode_tokens(output_sequences[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)  
                    )
                else:
                    cot_texts.append("FAILED")

            metrics = None
            recon_loss = None
            if output_sequences is not None and output_logits is not None:
                # Pass backpointer data if available for compress beam search mode
                metrics = compute_cot_reconstruction_metrics(
                    cot_gt, output_logits, cot_mask_ex, backpointers_ex, bp_logits)  
                if output_logits is not None and cot_gt is not None and cot_mask_ex is not None:  
                    recon_loss = compute_reconstruction_loss(output_logits, cot_gt, cot_mask_ex)  
            return {
                "output_sequences": output_sequences,
                "output_logits": output_logits,
                "vq_loss": vq_loss,
                "perplexity": perplexity,
                "indices": indices,
                "debug_stats": debug_stats if 'debug_stats' in locals() else None,
                "bp_logits": bp_logits,
                "cot_texts": cot_texts,
                "metrics": metrics,
                "recon_loss": recon_loss
            }

        for i in range(min(num_examples, len(prompt_sequences))):
            prompt = prompt_sequences[i:i+1].to(device)  # [1, K]
            cot_gt = cot_sequences[i:i+1].to(device)     # [1, M, L]
            prompt_mask_ex = prompt_mask[i:i+1].to(device) if prompt_mask is not None else None
            cot_mask_ex = cot_mask[i:i+1].to(device) if cot_mask is not None else None
            backpointers_ex = backpointers[i:i+1].to(device) if backpointers is not None else None

            prompt_text = decode_tokens(prompt[0], prompt_mask_ex[0] if prompt_mask_ex is not None else None)

            # Prepare ground truth CoT texts
            cot_gt_texts = [
                decode_tokens(cot_gt[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                for j in range(num_thoughts)
            ]

            metrics_to_store = ('token_level_accuracies', 'backpointer_accuracies')
            # Used at the very end of this function
            printing_names = ('token-level accuracy', 'backpointer accuracy')

            # Process teacher-forced mode
            tf_results = process_mode(
                inference_mode=False,
                prompt=prompt,
                cot_gt=cot_gt,
                prompt_mask_ex=prompt_mask_ex,
                cot_mask_ex=cot_mask_ex,  
                backpointers_ex=backpointers_ex,
                num_thoughts=num_thoughts,
                indices_accumulator=all_indices_tf
            )
            if tf_results["metrics"] is not None:
                for name in metrics_to_store: 
                    if name in tf_results["metrics"]:
                        metric_val = tf_results["metrics"][name][0].mean().item()
                        lst = tf_mean_per_example_metric.get(name, []).append(metric_val)
                        tf_mean_per_example_metric[name] = lst

            # Process auto-regressive mode
            ar_results = process_mode(
                inference_mode=True,
                prompt=prompt,
                cot_gt=cot_gt,
                prompt_mask_ex=prompt_mask_ex,
                cot_mask_ex=cot_mask_ex,  
                backpointers_ex=backpointers_ex,
                num_thoughts=num_thoughts,
                indices_accumulator=all_indices_ar
            )
            if ar_results["metrics"] is not None:
                for name in metrics_to_store: 
                    if name in ar_results["metrics"]:
                        metric_val = ar_results["metrics"][name][0].mean().item()
                        lst = ar_mean_per_example_metric.get(name, []).append(metric_val)
                        ar_mean_per_example_metric[name] = lst

            # Print results using the helper function
            print_demonstration_results(
                example_num=i+1,
                prompt_text=prompt_text,
                cot_gt_texts=cot_gt_texts,
                cot_tf_texts=tf_results["cot_texts"],
                cot_ar_texts=ar_results["cot_texts"],
                tf_metrics=tf_results["metrics"],
                ar_metrics=ar_results["metrics"],
                tf_indices=tf_results["indices"],
                ar_indices=ar_results["indices"],
                vq_loss_tf=tf_results["vq_loss"],
                perplexity_tf=tf_results["perplexity"],
                vq_loss_ar=ar_results["vq_loss"],
                perplexity_ar=ar_results["perplexity"],
                recon_loss_tf=tf_results["recon_loss"],
                recon_loss_ar=ar_results["recon_loss"],
                num_thoughts=num_thoughts
            )

            # Add latent visualization if available and for the first example
            if tokenizer is not None and do_figure_analyses:
                try:
                    print(f"\n{'='*80}")
                    print(f"GENERATING LATENT VISUALIZATION FOR EXAMPLE {i+1}")
                    print(f"{'='*80}")
                    analyzer = LatentVisualizationAnalyzer(model, tokenizer, device)
                    for j in range(num_thoughts):
                        print(f"\n--- Creating latent visualization for CoT {j+1} ---")
                        
                        def get_token_strings(tokens, mask, tokenizer):
                            if mask is not None:
                                tokens = tokens[mask.bool()]
                            return tokenizer.convert_ids_to_tokens(tokens.tolist())
                        
                        def get_latent_embeddings(indices, j, model):
                            latent_embeddings = model.vector_quantizer.embedding(indices)
                            if hasattr(model, 'chain_embeddings'):
                                chain_indices = torch.full(
                                    (latent_embeddings.size(0),), j,
                                    dtype=torch.long, device=latent_embeddings.device
                                )
                                chain_emb = model.chain_embeddings(chain_indices)
                                latent_embeddings = latent_embeddings + chain_emb
                            return latent_embeddings

                        original_tokens_raw = cot_gt[0, j]
                        original_mask = cot_mask_ex[0, j] if cot_mask_ex is not None else None  
                        original_token_strings = get_token_strings(original_tokens_raw, original_mask, tokenizer)

                        def run_latent_visualization(
                            mode_name,
                            token_tensor,
                            indices_tensor,
                            file_suffix,
                            fail_message
                        ):
                            if token_tensor is not None and indices_tensor is not None:
                                reconstructed_token_strings = get_token_strings(
                                    token_tensor[0, j], original_mask, tokenizer  
                                )
                                latent_embeddings = get_latent_embeddings(indices_tensor[0], j, model)
                                save_path = os.path.join(
                                    os.path.dirname(checkpoint_path),
                                    f"latent_alignment_cot{j+1}_{split_name}{file_suffix}.png"
                                )
                                
                                analyzer.visualize_cot_latent_alignment(
                                    original_tokens=original_token_strings,
                                    encoded_latents=latent_embeddings,
                                    reconstructed_tokens=reconstructed_token_strings,
                                    latent_indices=indices_tensor[0],
                                    title=f"CoT {j+1} Latent Alignment {mode_name} - {split_name.upper()} Data",
                                    save_path=save_path,
                                    tokens_per_chunk=8,
                                    max_chunks=None,
                                    show_token_boundaries=True,
                                    show_latent_stats=False,
                                    create_folder=True
                                )
                            else:
                                print(f"  {fail_message} for CoT {j+1}, skipping latent visualization")

                        # # Teacher-forced latent visualization
                        # run_latent_visualization(
                        #     mode_name="Teacher-forced",
                        #     token_tensor=tf_results["output_sequences"],
                        #     indices_tensor=tf_results["indices"],
                        #     file_suffix="_tf",
                        #     fail_message="Teacher forcing failed"
                        # )

                        # # Auto-regressive latent visualization
                        # run_latent_visualization(
                        #     mode_name="Auto-regressive",
                        #     token_tensor=ar_results["output_sequences"],
                        #     indices_tensor=ar_results["indices"],
                        #     file_suffix="_ar",
                        #     fail_message="Auto-regression failed"
                        # )

                    print(f"\nLatent visualization completed for {split_name} data!")

                    # Multi-CoT side-by-side alignment for teacher-forced
                    if tf_results["output_sequences"] is not None and tf_results["indices"] is not None:
                        visualize_token_latent_alignment_multi_cot(
                            all_original_token_ids=[cot_gt[0, j] for j in range(num_thoughts)],
                            all_latent_indices=[tf_results["indices"][0] for _ in range(num_thoughts)],
                            all_reconstructed_token_ids=[tf_results["output_sequences"][0, j] for j in range(num_thoughts)],
                            tokenizer=tokenizer,
                            mode_name="Teacher-forced",
                            max_tokens_per_line=15
                        )
                    # Multi-CoT side-by-side alignment for auto-regressive
                    if ar_results["output_sequences"] is not None and ar_results["indices"] is not None:
                        visualize_token_latent_alignment_multi_cot(
                            all_original_token_ids=[cot_gt[0, j] for j in range(num_thoughts)],
                            all_latent_indices=[ar_results["indices"][0] for _ in range(num_thoughts)],
                            all_reconstructed_token_ids=[ar_results["output_sequences"][0, j] for j in range(num_thoughts)],
                            tokenizer=tokenizer,
                            mode_name="Auto-regressive",
                            max_tokens_per_line=15
                        )
                    
                    if do_figure_analyses:
                        try:
                            print(f"\n{'='*80}")
                            print(f"GENERATING CHAIN EMBEDDINGS VISUALIZATION FOR {split_name.upper()} DATA")
                            print(f"{'='*80}")
                            chain_embeddings_path = os.path.join(os.path.dirname(checkpoint_path), 
                                                            f"chain_embeddings_{split_name}.png")
                            analyzer.visualize_chain_embeddings(
                                title=f"Chain Embeddings - {split_name.upper()} Data",
                                save_path=chain_embeddings_path,
                                show_stats=True
                            )
                            print(f"Chain embeddings visualization completed for {split_name} data!")
                        
                        except Exception as e:
                            print(f"Error during chain embeddings visualization: {e}")
                            import traceback
                            traceback.print_exc()
                        
                    if do_figure_analyses:
                        try:
                            print(f"\n{'='*80}")
                            print(f"GENERATING WORD-TO-LATENT MAPPING ANALYSIS FOR {split_name.upper()} DATA")
                            print(f"{'='*80}")
                            word_mapping_dir = os.path.join(os.path.dirname(checkpoint_path), 
                                                            f"word_mapping_analysis_{split_name}")
                            os.makedirs(word_mapping_dir, exist_ok=True)
                            analyzer.analyze_word_to_latent_mapping(
                                prompt_sequences=prompt_sequences,
                                cot_sequences=cot_sequences,
                                prompt_mask=prompt_mask,
                                cot_mask=cot_mask,
                                output_dir=word_mapping_dir,
                                sample_size=min(100, len(prompt_sequences)),
                                top_k_words=15,
                                top_k_codes=25
                            )
                            print(f"Word-to-latent mapping analysis completed for {split_name} data!")
                        except Exception as e:
                            print(f"Error during word-to-latent mapping analysis: {e}")
                            import traceback
                            traceback.print_exc()
                        
                except Exception as e:
                    print(f"Error during latent visualization: {e}")
                    import traceback
                    traceback.print_exc()
        
        # After all examples, report the overall mean per-example accuracy for each mode
        if tf_mean_per_example_metric:
            for print_name, vals in zip(printing_names, tf_mean_per_example_metric.values()):
                overall_val = sum(vals) / len(vals)
                print(f"\n[Teacher-forced] Overall mean {print_name} over all examples: {overall_val:.4f}")
        if ar_mean_per_example_metric:
            for print_name, vals in zip(printing_names, ar_mean_per_example_metric.values()):
                overall_val = sum(vals) / len(vals)
                print(f"[Auto-regressive] Overall mean {print_name} over all examples: {overall_val:.4f}")

        
        if use_vq and do_figure_analyses:
            print(f"\n{'='*80}")
            print(f"GENERATING COMPREHENSIVE CODEBOOK USAGE HEATMAPS FOR {split_name.upper()} DATA")
            print(f"{'='*80}")
            def generate_codebook_usage_heatmap(all_indices, mode_name, file_suffix):
                if all_indices:
                    combined_indices = torch.cat(all_indices, dim=0)
                    print(f"{mode_name} - Total indices: {combined_indices.numel()}")
                    print(f"{mode_name} - Unique indices: {torch.unique(combined_indices).numel()}")
                    heatmap_path = os.path.join(
                        os.path.dirname(checkpoint_path),
                        f"codebook_usage_{file_suffix}_comprehensive_{split_name}.png"
                    )
                    counts = torch.bincount(
                        combined_indices[combined_indices < num_embeddings],
                        minlength=num_embeddings
                    )
                    create_codebook_usage_heatmap(
                        counts.cpu().numpy(),
                        num_embeddings=num_embeddings,
                        title=f"{mode_name} Codebook Usage - All {num_examples} Examples ({split_name})",
                        save_path=heatmap_path
                    )

            generate_codebook_usage_heatmap(all_indices_tf, "Teacher Forcing", "tf")
            generate_codebook_usage_heatmap(all_indices_ar, "Auto-regressive", "ar")
        print("\nDemonstration completed for {} data!".format(split_name))


def compute_cot_reconstruction_metrics(
    ground_truth_cots, predicted_logits, cot_mask=None, 
    backpointer_gt=None, backpointer_logits=None, thresholds=[1.0, 0.95, 0.9]
):
    """
    Compute reconstruction loss, accuracy, and additional metrics for each CoT sequence using vectorized operations.
    Args:
        ground_truth_cots: Tensor of shape [batch, num_thoughts, seq_len]
        predicted_logits: Tensor of shape [batch, num_thoughts, seq_len, vocab_size]
        cot_mask: Optional mask tensor of shape [batch, num_thoughts, seq_len]
        backpointer_gt: Optional backpointer ground truth tensor of shape [batch, num_thoughts, seq_len]
        backpointer_logits: Optional backpointer logits tensor of shape [batch, num_thoughts, seq_len, num_thoughts]
        thresholds: List of floats for sequence-level accuracy thresholds (e.g., [1.0, 0.95, 0.9])
    Returns:
        metrics: dict with keys:
            - reconstruction_losses: [batch, num_thoughts]
            - token_level_accuracies: [batch, num_thoughts]
            - sequence_level_accuracies: {threshold: float}
            - perplexities: [batch, num_thoughts]
            - backpointer_accuracies: [batch, num_thoughts] (if backpointer data provided)
            - backpointer_losses: [batch, num_thoughts] (if backpointer data provided)
    """
    def compute_mask(ground_truth_cots, cot_mask):
        if cot_mask is not None:
            mask = cot_mask.bool()  # [B, M, L]
        else:
            mask = torch.ones_like(ground_truth_cots, dtype=torch.bool)  # [B, M, L]
        return mask
    
    def average_over_valid_positions(metric_arr, mask):
        """
        Computes the average of metric_arr over valid positions indicated by mask, 
        avoiding division by zero.
        Args:
            metric_arr: Tensor of shape [B, M, L]
            mask: Bool tensor of shape [B, M, L]
        Returns:
            avg: Tensor of shape [B, M]
        """
        mask_f = mask.float()  # [B, M, L]
        num_valid = mask_f.sum(dim=2)  # [B, M]
        num_valid = num_valid + (num_valid == 0)  # [B, M]
        avg = (metric_arr * mask_f).sum(dim=2) / num_valid  # [B, M]
        return avg  # [B, M]

    def compute_reconstruction_losses(ground_truth_cots, predicted_logits, mask):
        batch_size, num_thoughts, seq_len = ground_truth_cots.shape
        flat_logits = predicted_logits.reshape(-1, predicted_logits.size(-1))  # [B*M*L, V]  # [B*M*L, V]
        flat_targets = ground_truth_cots.reshape(-1)  # [B*M*L]  # [B*M*L]
        flat_mask = mask.reshape(-1)  # [B*M*L]  # [B*M*L]
        per_token_loss = torch.zeros_like(flat_targets, dtype=predicted_logits.dtype, device=ground_truth_cots.device)  # [B*M*L]
        if flat_mask.sum() > 0:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=50256, reduction='none')
            per_token_loss[flat_mask] = criterion(flat_logits[flat_mask], flat_targets[flat_mask])
        per_token_loss = per_token_loss.view(batch_size, num_thoughts, seq_len)  # [B, M, L]
        reconstruction_losses = average_over_valid_positions(per_token_loss, mask)  # [B, M]
        return reconstruction_losses  # [B, M]

    def compute_token_level_accuracies(ground_truth_cots, predicted_logits, mask):
        predicted_tokens = torch.argmax(predicted_logits, dim=-1)  # [B, M, L]
        correct = ((predicted_tokens == ground_truth_cots) & mask).float()  # [B, M, L]
        token_level_accuracies = average_over_valid_positions(correct, mask)  # [B, M]
        return token_level_accuracies  # [B, M]

    def compute_sequence_level_accuracies(token_level_accuracies, thresholds):
        sequence_level_accuracies = {}
        for thresh in thresholds:
            sequence_level_accuracies[thresh] = (token_level_accuracies >= thresh).sum().item()
        return sequence_level_accuracies

    def compute_perplexities(ground_truth_cots, predicted_logits, mask):
        log_probs = F.log_softmax(predicted_logits, dim=-1)  # [B, M, L, V]
        gt_log_probs = log_probs.gather(-1, ground_truth_cots.unsqueeze(-1)).squeeze(-1)  # [B, M, L]
        nll = torch.zeros_like(gt_log_probs)  # [B, M, L]
        nll[mask] = -gt_log_probs[mask]
        nll_mean = average_over_valid_positions(nll, mask)  # [B, M]
        perplexities = torch.exp(nll_mean)  # [B, M]
        return perplexities  # [B, M]

    def compute_backpointer_accuracies(backpointer_gt, backpointer_logits, mask):
        """
        Compute backpointer prediction accuracy for compress beam search mode.
        Args:
            backpointer_gt: Ground truth backpointers [B, M, L]
            backpointer_logits: Predicted backpointer logits [B, M, L, num_thoughts]
            mask: Mask tensor [B, M, L]
        Returns:
            backpointer_accuracies: [B, M]
        """
        predicted_backpointers = torch.argmax(backpointer_logits, dim=-1)  # [B, M, L]
        correct = ((predicted_backpointers == backpointer_gt) & mask).float()  # [B, M, L]
        backpointer_accuracies = average_over_valid_positions(correct, mask)  # [B, M]
        return backpointer_accuracies  # [B, M]

    def compute_backpointer_losses(backpointer_gt, backpointer_logits, mask):
        """
        Compute backpointer prediction loss for compress beam search mode.
        Args:
            backpointer_gt: Ground truth backpointers [B, M, L]
            backpointer_logits: Predicted backpointer logits [B, M, L, num_thoughts]
            mask: Mask tensor [B, M, L]
        Returns:
            backpointer_losses: [B, M]
        """
        batch_size, num_thoughts, seq_len = backpointer_gt.shape
        flat_logits = backpointer_logits.reshape(-1, backpointer_logits.size(-1))  # [B*M*L, num_thoughts]
        flat_targets = backpointer_gt.reshape(-1)  # [B*M*L]
        flat_mask = mask.reshape(-1)  # [B*M*L]
        per_token_loss = torch.zeros_like(flat_targets, dtype=backpointer_logits.dtype, device=backpointer_gt.device)  # [B*M*L]
        if flat_mask.sum() > 0:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss[flat_mask] = criterion(flat_logits[flat_mask], flat_targets[flat_mask])
        per_token_loss = per_token_loss.view(batch_size, num_thoughts, seq_len)  # [B, M, L]
        backpointer_losses = average_over_valid_positions(per_token_loss, mask)  # [B, M]
        return backpointer_losses  # [B, M]
    

    mask = compute_mask(ground_truth_cots, cot_mask)  # [B, M, L]
    reconstruction_losses = compute_reconstruction_losses(ground_truth_cots, predicted_logits, mask)  # [B, M]
    token_level_accuracies = compute_token_level_accuracies(ground_truth_cots, predicted_logits, mask)  # [B, M]
    sequence_level_accuracies = compute_sequence_level_accuracies(token_level_accuracies, thresholds)
    perplexities = compute_perplexities(ground_truth_cots, predicted_logits, mask)  # [B, M]

    metrics = {
        'reconstruction_losses': reconstruction_losses,
        'token_level_accuracies': token_level_accuracies,
        'sequence_level_accuracies': sequence_level_accuracies,
        'perplexities': perplexities
    }

    # Add backpointer metrics if data is provided
    if backpointer_gt is not None and backpointer_logits is not None:
        backpointer_accuracies = compute_backpointer_accuracies(backpointer_gt, backpointer_logits, mask)  # [B, M]
        backpointer_losses = compute_backpointer_losses(backpointer_gt, backpointer_logits, mask)  # [B, M]
        metrics['backpointer_accuracies'] = backpointer_accuracies
        metrics['backpointer_losses'] = backpointer_losses

    return metrics


def compute_word_latent_mapping_on_dataset(
    model, tokenizer,
    prompt_sequences=None, cot_sequences=None, prompt_mask=None, cot_mask=None,
    split_name=None, checkpoint_path=None,
    data_dir=None, num_thoughts=None, seed=42,
    sample_size=None
):
    """
    Compute the word-to-latent mapping analysis for the entire dataset and save the results.
    Args:
        model: The model instance
        tokenizer: The tokenizer instance
        prompt_sequences: Tensor of prompt sequences (optional if data_dir is provided)
        cot_sequences: Tensor of CoT sequences (optional if data_dir is provided)
        prompt_mask: Mask for prompt sequences (optional if data_dir is provided)
        cot_mask: Mask for CoT sequences (optional if data_dir is provided)
        split_name: Name of the data split (e.g., 'train', 'test')
        checkpoint_path: Path to the model checkpoint (used for output directory)
        data_dir: Path to data directory (optional, alternative to providing sequences directly)
        num_thoughts: Number of CoT sequences per example (required if data_dir is provided)
        seed: Random seed for data loading (default: 42)
        sample_size: Number of samples to analyze (default: None, meaning all samples)
    """
    # If data_dir is provided, load data
    if data_dir is not None:
        if num_thoughts is None:
            raise ValueError("num_thoughts must be provided when loading data from data_dir.")
        print(f"Loading data from {data_dir} for split '{split_name}'...")
        try:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = load_training_data(
                data_dir=data_dir, max_samples=None, num_thoughts=num_thoughts, seed=seed
            )
            if split_name == "train":
                prompt_sequences, cot_sequences, prompt_mask, cot_mask = \
                    train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask
            elif split_name == "test":
                prompt_sequences, cot_sequences, prompt_mask, cot_mask = \
                    test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask
            else:
                raise ValueError(f"Unknown split_name: {split_name}. Must be 'train' or 'test'.")
        except Exception as e:
            print(f"Error loading data from {data_dir}: {e}")
            return
    # Check that required tensors are available
    if prompt_sequences is None or cot_sequences is None:
        raise ValueError("Either prompt_sequences/cot_sequences or data_dir and split_name must be provided.")
    device = next(model.parameters()).device
    analyzer = LatentVisualizationAnalyzer(model, tokenizer, device)
    word_mapping_dir = os.path.join(os.path.dirname(checkpoint_path), f"word_mapping_analysis_{split_name}")
    os.makedirs(word_mapping_dir, exist_ok=True)
    print(f"Computing word-to-latent mapping analysis for {split_name} data...")
    analyzer.analyze_word_to_latent_mapping(
        prompt_sequences=prompt_sequences,
        cot_sequences=cot_sequences,
        prompt_mask=prompt_mask,
        cot_mask=cot_mask,
        output_dir=word_mapping_dir,
        sample_size=sample_size,
        top_k_words=15,
        top_k_codes=25
    )
    print(f"Word-to-latent mapping analysis completed for {split_name} data! Results saved to {word_mapping_dir}.")


def compute_dataset_reconstruction_metrics_with_examples(
    model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, 
    batch_size=256, use_vq=True, ar_gen=True, k=None, tokenizer=None,
    backpointers=None
):
    """
    Compute dataset-level reconstruction metrics by processing the dataset in batches and keep track of specific examples.
    
    Args:
        model: The model instance
        prompt_sequences: Tensor of prompt sequences
        cot_sequences: Tensor of CoT sequences  
        prompt_mask: Mask for prompt sequences
        cot_mask: Mask for CoT sequences
        batch_size: Batch size for processing (default: 256)
        use_vq: Whether to use vector quantization (default: True)
        ar_gen: Whether to generate auto-regressively or using teacher-forcing (default: True)
        k: If provided, the number of examples to keep for each category (default: None, none is kept)
        tokenizer: Tokenizer for decoding text (optional)
        backpointers: Optional backpointer tensor of shape [batch, num_thoughts, seq_len] for compress beam search mode
    
    Returns:
        dict: Dataset-level average metrics and tracked examples containing:
            - avg_reconstruction_loss: float
            - avg_token_level_accuracy: float  
            - avg_perplexity: float
            - sequence_level_accuracies: dict mapping threshold to count
            - tracked_examples: dict with examples for each category (only if k is not None)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Initialize GPU memory monitoring
    gpu_monitor = GPUMemoryMonitor(device_id=device.index if device.type == 'cuda' else None)
    memory_usage_data = []
    start_time = time.time()
    
    def log_memory_point(stage: str, batch_idx: int = None):
        """Log memory usage at a specific point"""
        timestamp = time.time() - start_time
        mem_info = gpu_monitor.log_memory_usage(stage, print_info=False)
        memory_usage_data.append({
            'timestamp': timestamp,
            'stage': stage,
            'batch_idx': batch_idx,
            'used_gb': mem_info['used_gb'],
            'free_gb': mem_info['free_gb'],
            'total_gb': mem_info['total_gb'],
            'utilization_percent': mem_info['utilization_percent']
        })
    
    def plot_memory_usage():
        """Plot GPU memory usage over time"""
        if not memory_usage_data:
            print("No memory usage data to plot.")
            return
        
        try:
            # timestamps = [data['timestamp'] for data in memory_usage_data]
            used_gb = [data['used_gb'] for data in memory_usage_data]
            utilization = [data['utilization_percent'] for data in memory_usage_data]
            stages = [data['stage'] for data in memory_usage_data]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot memory usage in GB
            ax1.plot(range(len(stages)), used_gb, 'b-', label='Used Memory (GB)', linewidth=2)
            ax1.set_ylabel('Memory (GB)')
            ax1.set_title('GPU Memory Usage Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(len(stages)))
            ax1.set_xticklabels(stages, rotation=45, ha='right')    
            
            # Plot utilization percentage
            ax2.plot(range(len(stages)), utilization, 'r-', label='Utilization (%)', linewidth=2)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Utilization (%)')
            ax2.set_title('GPU Memory Utilization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(len(stages)))
            ax2.set_xticklabels(stages, rotation=45, ha='right')
            
            plt.tight_layout()
            savepath = "gpu_memory_usage.png"
            while os.path.exists(savepath):
                savepath = savepath.replace(".png", "D.png")
            plt.savefig(savepath)
            plt.close(fig)
            
            # Print summary statistics
            print(f"\n{'='*60}")
            print("GPU MEMORY USAGE SUMMARY")
            print(f"{'='*60}")
            print(f"Peak memory usage: {max(used_gb):.2f} GB")
            print(f"Average memory usage: {sum(used_gb)/len(used_gb):.2f} GB")
            print(f"Peak utilization: {max(utilization):.1f}%")
            print(f"Average utilization: {sum(utilization)/len(utilization):.1f}%")
            print(f"Number of memory measurements: {len(memory_usage_data)}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error plotting memory usage: {e}")
            import traceback
            traceback.print_exc()
    
    # Log initial memory state
    log_memory_point("function_start")
    
    total_samples = len(prompt_sequences)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    # Accumulators for averaging
    total_reconstruction_loss = 0.0
    total_token_accuracy = 0.0
    total_perplexity = 0.0
    total_sequences = 0
    
    # For sequence-level accuracies, we need to track all individual accuracies
    sequence_level_accuracies = {}
    thresholds = [1.0, 0.95, 0.9, 0.8, 0.7]
    
    # Track best and worst examples using lists
    best_loss_list = []  # List for lowest loss (low to high)
    worst_loss_list = []  # List for highest loss (high to low)
    best_perplexity_list = []  # List for lowest perplexity (low to high)
    worst_perplexity_list = []  # List for highest perplexity (high to low)
    
    print(f"Computing dataset reconstruction metrics over {total_samples} samples in {num_batches} batches...")
    
    def update_tracked_list(tracked_list, precopy_example, k, is_best, use_loss):
        """Update a tracked list to maintain top k examples efficiently"""
        # Determine the key for comparison
        key = precopy_example['avg_loss'] if use_loss else precopy_example['avg_perplexity']
        
        # Check if we should insert this new example
        worst_key = tracked_list[-1][0] if len(tracked_list) > 0 else None
        # For loss & perplexity, the lower the better!
        if len(tracked_list) < k or (is_best and key < worst_key) or (not is_best and key > worst_key):
            copy_example = dict([(k, (v.clone() if type(v) == torch.Tensor else v)) 
                                 for (k, v) in precopy_example.items()])

            if len(tracked_list) >= k:
                _, deleted = tracked_list.pop()
                for v in deleted.values(): del v
                del deleted
                        
            tracked_list.append((key, copy_example))
            tracked_list.sort(key=lambda x: x[0], reverse=not is_best)
        
        return tracked_list
    
    with torch.no_grad():
        for batch_idx in tqdm.tqdm(range(num_batches)):
            # Log memory before each batch
            if TRACK_BATCH_MEMORY_USE:
                log_memory_point(f"batch_{batch_idx}_start", batch_idx)
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            
            # Get batch data
            batch_prompts = prompt_sequences[start_idx:end_idx].to(device)
            batch_cots = cot_sequences[start_idx:end_idx].to(device)
            batch_prompt_mask = prompt_mask[start_idx:end_idx].to(device) if prompt_mask is not None else None
            batch_cot_mask = cot_mask[start_idx:end_idx].to(device) if cot_mask is not None else None
            batch_backpointers = backpointers[start_idx:end_idx].to(device) if backpointers is not None else None
            
            # Log memory after data loading
            if TRACK_BATCH_MEMORY_USE:
                log_memory_point(f"batch_{batch_idx}_data_loaded", batch_idx)
            
            # Forward pass (teacher forcing)
            model_inputs = {
                'prompt': batch_prompts,
                'cot_sequences': batch_cots,
                'cot_mask': batch_cot_mask,
                'prompt_mask': batch_prompt_mask,
                'inference': ar_gen,
                'quantize_cot_only': True
            }
            if batch_backpointers is not None:
                model_inputs['backpointers'] = batch_backpointers
            if hasattr(model, 'use_vq'):
                model_inputs['use_vq'] = use_vq
            else:
                model_inputs['no_vq'] = not use_vq
            
            if TRACK_BATCH_MEMORY_USE:
                log_memory_point(f"batch_{batch_idx}_after_input_ready", batch_idx)
            
            out = model(**model_inputs)
            # Unpack outputs based on length
            if len(out) == 6:
                _, output_logits, _, _, indices, _ = out
                batch_bp_logits = None
            elif len(out) == 7:
                _, output_logits, _, _, indices, _, batch_bp_logits = out
            else:
                output_logits = indices = batch_bp_logits = None

            for v in model_inputs.values(): del v
            del model_inputs
            
            # Log memory after forward pass
            if TRACK_BATCH_MEMORY_USE:
                log_memory_point(f"batch_{batch_idx}_forward_pass", batch_idx)
            
            # Compute metrics for this batch
            batch_metrics = compute_cot_reconstruction_metrics(
                batch_cots, output_logits, batch_cot_mask, batch_backpointers, batch_bp_logits
            )
            
            # Log memory after metrics computation
            if TRACK_BATCH_MEMORY_USE:
                log_memory_point(f"batch_{batch_idx}_metrics_computed", batch_idx)
            
            # Accumulate metrics
            batch_size_actual = batch_cots.size(0)
            total_sequences += batch_size_actual * batch_cots.size(1)  # batch_size * num_thoughts
            
            # Average over batch and num_thoughts dimensions
            total_reconstruction_loss += batch_metrics['reconstruction_losses'].mean().item() * batch_size_actual
            total_token_accuracy += batch_metrics['token_level_accuracies'].mean().item() * batch_size_actual
            total_perplexity += batch_metrics['perplexities'].mean().item() * batch_size_actual
            
            # Compute and store sequence-level accuracies
            token_accuracies = batch_metrics['token_level_accuracies'].flatten()
            for thresh in thresholds:
                sequence_level_accuracies[thresh] = (token_accuracies >= thresh).sum().item()
            del token_accuracies
            
            if k is not None:
                # Track examples (multiple CoTs) with their average metrics
                for i in range(batch_size_actual):
                    example_idx = start_idx + i
                    
                    # Compute average metrics across all CoTs for this example
                    avg_loss = batch_metrics['reconstruction_losses'][i].mean().item()
                    avg_perplexity = batch_metrics['perplexities'][i].mean().item()
                    
                    # Get original prompt and all cots for this example
                    prompt = batch_prompts[i]
                    cots = batch_cots[i]  # [num_thoughts, seq_len]
                    recon_logits = output_logits[i] # [num_thoughts, seq_len]
                    prompt_mask_ex = batch_prompt_mask[i] if batch_prompt_mask is not None else None
                    cot_mask_ex = batch_cot_mask[i] if batch_cot_mask is not None else None  # [num_thoughts, seq_len]
                    
                    example_data_precopy = { # torch.tensor or int
                        'example_idx': example_idx,
                        'avg_loss': avg_loss,
                        'avg_perplexity': avg_perplexity,
                        'prompt': prompt,
                        'cots': cots,
                        'recon_logits' : recon_logits,
                        'prompt_mask': prompt_mask_ex,
                        'cot_mask': cot_mask_ex,
                        'individual_losses': batch_metrics['reconstruction_losses'][i],
                        'individual_perplexities': batch_metrics['perplexities'][i],
                        'individual_token_accuracies': batch_metrics['token_level_accuracies'][i]
                    }
                    
                    # Add backpointer data if available
                    # TODO DO SOMETHING WITH IT, CURRENTLY NOTHING DONE
                    if batch_backpointers is not None:
                        example_data_precopy['backpointers'] = batch_backpointers[i]
                    if batch_bp_logits is not None:
                        example_data_precopy['bp_logits'] = batch_bp_logits[i]
                    
                    # Update tracked lists for best/worst examples
                    best_loss_list        = update_tracked_list(best_loss_list,        example_data_precopy, k, is_best=True,  use_loss=True)
                    worst_loss_list       = update_tracked_list(worst_loss_list,       example_data_precopy, k, is_best=False, use_loss=True)
                    best_perplexity_list  = update_tracked_list(best_perplexity_list,  example_data_precopy, k, is_best=True,  use_loss=False)
                    worst_perplexity_list = update_tracked_list(worst_perplexity_list, example_data_precopy, k, is_best=False, use_loss=False)

                    del prompt, cots, recon_logits, prompt_mask_ex, cot_mask_ex
                    for v in example_data_precopy.values(): del v
                    del example_data_precopy

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{num_batches} batches...")
                    
            del batch_prompts, batch_cots, batch_prompt_mask, batch_cot_mask, output_logits, batch_metrics
            if batch_backpointers is not None:
                del batch_backpointers, batch_bp_logits
            
            # Log memory after cleanup
            if TRACK_BATCH_MEMORY_USE:
                log_memory_point(f"batch_{batch_idx}_cleanup_complete", batch_idx)
    
    # Log memory after main processing
    log_memory_point("main_processing_complete")
    
    # Compute final averages
    avg_reconstruction_loss = total_reconstruction_loss / total_samples
    avg_token_accuracy = total_token_accuracy / total_samples
    avg_perplexity = total_perplexity / total_samples
    
    # Prepare tracked examples from lists
    # Extract examples from tracked lists
    def extract_from_tracked_list(tracked_list):
        """Extract examples from tracked list (already sorted)"""
        return [item[1] for item in tracked_list]  # Extract example data from (key, example) tuples
    
    if k is not None:
        tracked_examples = {
            'highest_loss': extract_from_tracked_list(worst_loss_list),
            'lowest_loss': extract_from_tracked_list(best_loss_list),
            'highest_perplexity': extract_from_tracked_list(worst_perplexity_list),
            'lowest_perplexity': extract_from_tracked_list(best_perplexity_list)
        }

        del worst_loss_list, best_loss_list, worst_perplexity_list, best_perplexity_list
        # Log memory before random examples processing
        log_memory_point("random_examples_start")
        
        # Sample k random examples by selecting random indices and recomputing
        # Use torch.randperm for reproducibility and efficiency
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_samples, generator=generator)[:min(k, total_samples)]
        random_examples = []
        batch_size_random = min(32, batch_size)  # Use a reasonable batch size for random examples

        with torch.no_grad():
            for batch_start in range(0, len(indices), batch_size_random):
                batch_indices = indices[batch_start:batch_start + batch_size_random]
                    
                # Advanced indexing to get batch of random examples
                batch_prompts = prompt_sequences[batch_indices].to(device)
                batch_cots = cot_sequences[batch_indices].to(device)
                batch_prompt_mask = prompt_mask[batch_indices].to(device) if prompt_mask is not None else None
                batch_cot_mask = cot_mask[batch_indices].to(device) if cot_mask is not None else None
                batch_backpointers = backpointers[batch_indices].to(device) if backpointers is not None else None

                model_inputs = {
                    'prompt': batch_prompts,
                    'cot_sequences': batch_cots,
                    'cot_mask': batch_cot_mask,
                    'prompt_mask': batch_prompt_mask,
                    'inference': ar_gen,
                    'quantize_cot_only': True
                }
                if batch_backpointers is not None:
                    model_inputs['backpointers'] = batch_backpointers

                if hasattr(model, 'use_vq'):
                    model_inputs['use_vq'] = use_vq
                else:
                    model_inputs['no_vq'] = not use_vq

                out = model(**model_inputs)
                # Unpack outputs based on length
                if len(out) == 6:
                    _, output_logits, _, _, indices, _ = out
                    batch_bp_logits = None
                elif len(out) == 7:
                    _, output_logits, _, _, indices, _, batch_bp_logits = out
                else:
                    output_logits = indices = batch_bp_logits = None

                for v in model_inputs.values(): del v
                del model_inputs

                # Compute metrics for this batch
                batch_metrics = compute_cot_reconstruction_metrics(
                    batch_cots, output_logits, batch_cot_mask, batch_backpointers, batch_bp_logits
                )

                for i in range(batch_prompts.size(0)):
                    example_idx = batch_indices[i].item()
                    prompt = batch_prompts[i]
                    cots = batch_cots[i]
                    recon_logits = output_logits[i]
                    prompt_mask_ex = batch_prompt_mask[i] if batch_prompt_mask is not None else None
                    cot_mask_ex = batch_cot_mask[i] if batch_cot_mask is not None else None

                    avg_loss = batch_metrics['reconstruction_losses'][i].mean().item()
                    avg_perplexity = batch_metrics['perplexities'][i].mean().item()

                    example_data = {
                        'example_idx': example_idx,
                        'avg_loss': avg_loss,
                        'avg_perplexity': avg_perplexity,
                        'prompt': prompt.clone(),
                        'cots': cots.clone(),
                        'recon_logits' : recon_logits.clone(),
                        'prompt_mask': prompt_mask_ex.clone(),
                        'cot_mask': cot_mask_ex.clone(),
                        'individual_losses': batch_metrics['reconstruction_losses'][i].clone(),
                        'individual_perplexities': batch_metrics['perplexities'][i].clone(),
                        'individual_token_accuracies': batch_metrics['token_level_accuracies'][i].clone()
                    }
                    
                    # Add backpointer data if available
                    if batch_backpointers is not None:
                        example_data['backpointers'] = batch_backpointers[i].clone()
                    if batch_bp_logits is not None:
                        example_data['bp_logits'] = batch_bp_logits[i].clone()
                    
                    random_examples.append(example_data)

                    del prompt, cots, recon_logits, prompt_mask_ex, cot_mask_ex

                del batch_metrics, batch_prompts, batch_cots, batch_prompt_mask, batch_cot_mask
                if batch_backpointers is not None:
                    del batch_backpointers, batch_bp_logits
                
        tracked_examples['random'] = random_examples
        
        # Log memory after random examples processing
        log_memory_point("random_examples_complete")
    else:
        tracked_examples = {}
    
    results = {
        'avg_reconstruction_loss': avg_reconstruction_loss,
        'avg_token_level_accuracy': avg_token_accuracy,
        'avg_perplexity': avg_perplexity,
        'sequence_level_accuracies': sequence_level_accuracies,
        'total_sequences': total_sequences,
        'tracked_examples': tracked_examples
    }
    
    print(f"Dataset reconstruction metrics computed successfully!")
    print(f"Average reconstruction loss: {avg_reconstruction_loss:.4f}")
    print(f"Average token-level accuracy: {avg_token_accuracy:.4f}")
    print(f"Average perplexity: {avg_perplexity:.4f}")
    print(f"Sequence-level accuracies:")
    for thresh, count in sequence_level_accuracies.items():
        percentage = (count / total_sequences) * 100
        print(f"  Threshold {int(thresh*100)}%: {count}/{total_sequences} sequences ({percentage:.1f}%)")
    
    # Plot memory usage at the end
    plot_memory_usage()
    
    return results

def compute_dataset_reconstruction_metrics_from_checkpoint_and_keep_examples(
    checkpoint_path: str,
    data_dir: str = None,
    split_name: str = "both",  # "train", "test", or "both"
    num_examples_train: int = None,
    num_examples_test: int = None,
    batch_size: int = 256,
    ar_gen: bool=True,
    k: int = 5,  # Number of examples to keep for each category
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_vq: bool = True,
    seed: int = 42
):
    """
    Compute dataset-level reconstruction metrics for EnhancedGPT2VQVAE from a checkpoint and keep track of specific examples.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_dir: Path to data directory (optional, will use GSM8K default if not provided)
        split_name: Which split to analyze: 'train', 'test', or 'both'
        num_examples_train: Number of random samples from train split (optional)
        num_examples_test: Number of random samples from test split (optional)
        batch_size: Batch size for processing
        ar_gen: Whether to generate auto-regressively or with teacher-forcing
        k: Number of examples to keep for each category (highest/lowest loss/perplexity, random). If None, do not track
        device: Device to run on
        use_vq: Whether to use vector quantization
        seed: Random seed for data loading
    """
    print(f"Loading EnhancedGPT2VQVAE from checkpoint: {checkpoint_path}")
    try:
        model = EnhancedGPT2VQVAE.from_checkpoint(checkpoint_path, device=device)
        print(f"Successfully loaded EnhancedGPT2VQVAE model")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Make sure the checkpoint file exists and contains the required model configuration.")
        return
    
    model.eval()
    num_thoughts = getattr(model, 'num_thoughts', None)
    if num_thoughts is None:
        raise AttributeError("Loaded model does not have 'num_thoughts' attribute.")
    print(f"Detected num_thoughts from model: {num_thoughts}")
    
    # Set data_dir if not provided
    if data_dir is None:
        data_dir = f"data/GSM8K/128_128/batch_{num_thoughts}"
        print(f"No data_dir provided. Using default: {data_dir}")
    else:
        print(f"Using provided data_dir: {data_dir}")
    
    print("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")
    
    # Load data
    print(f"Loading data from {data_dir}...")
    try:
        data_loaded = load_training_data(
            data_dir=data_dir, max_samples=None, num_thoughts=num_thoughts, seed=seed
        )
        # Handle both cases: with and without backpointers
        if len(data_loaded) == 8:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = data_loaded
            train_backpointers = test_backpointers = None
        elif len(data_loaded) == 10:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, train_backpointers, \
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, test_backpointers = data_loaded
        else:
            raise ValueError(f"Unexpected number of return values from load_training_data: {len(data_loaded)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Function to sample data if num_examples is specified
    def sample_data(prompt_sequences, cot_sequences, prompt_mask, cot_mask, num_examples, sample_seed, backpointers=None):
        if num_examples is None or num_examples >= len(prompt_sequences):
            if backpointers is not None:
                return prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers
            else:
                return prompt_sequences, cot_sequences, prompt_mask, cot_mask
        
        # Use torch.randperm with seed for reproducible random sampling
        torch.manual_seed(sample_seed)
        indices = torch.randperm(len(prompt_sequences))[:num_examples]
        result = (prompt_sequences[indices], cot_sequences[indices], 
                prompt_mask[indices] if prompt_mask is not None else None,
                cot_mask[indices] if cot_mask is not None else None)
        if backpointers is not None:
            result += (backpointers[indices],)
        return result
    
    results = {}
    
    if split_name in ["train", "both"]:
        print(f"\n{'='*80}")
        print(f"COMPUTING RECONSTRUCTION METRICS FOR TRAIN SPLIT")
        print(f"{'='*80}")
        
        train_data = sample_data(train_prompt_sequences, train_cot_sequences, 
                               train_prompt_mask, train_cot_mask, num_examples_train, seed, train_backpointers)
        
        # Extract data from sample_data result
        if len(train_data) == 5:  # With backpointers
            train_prompts, train_cots, train_prompts_mask, train_cots_mask, train_bps = train_data
        else:  # Without backpointers
            train_prompts, train_cots, train_prompts_mask, train_cots_mask = train_data
            train_bps = None
            
        train_results = compute_dataset_reconstruction_metrics_with_examples(
            model=model,
            prompt_sequences=train_prompts,
            cot_sequences=train_cots,
            prompt_mask=train_prompts_mask,
            cot_mask=train_cots_mask,
            batch_size=batch_size,
            use_vq=use_vq,
            ar_gen=ar_gen,
            k=k,
            tokenizer=tokenizer,
            backpointers=train_bps
        )
        results['train'] = train_results
    
    if split_name in ["test", "both"]:
        print(f"\n{'='*80}")
        print(f"COMPUTING RECONSTRUCTION METRICS FOR TEST SPLIT")
        print(f"{'='*80}")
        
        test_data = sample_data(test_prompt_sequences, test_cot_sequences,
                              test_prompt_mask, test_cot_mask, num_examples_test, seed, test_backpointers)
        
        # Extract data from sample_data result
        if len(test_data) == 5:  # With backpointers
            test_prompts, test_cots, test_prompts_mask, test_cots_mask, test_bps = test_data
        else:  # Without backpointers
            test_prompts, test_cots, test_prompts_mask, test_cots_mask = test_data
            test_bps = None
            
        test_results = compute_dataset_reconstruction_metrics_with_examples(
            model=model,
            prompt_sequences=test_prompts,
            cot_sequences=test_cots,
            prompt_mask=test_prompts_mask,
            cot_mask=test_cots_mask,
            batch_size=batch_size,
            use_vq=use_vq,
            ar_gen=ar_gen,
            k=k,
            tokenizer=tokenizer,
            backpointers=test_bps
        )
        results['test'] = test_results
    
    if split_name == "both":
        print(f"\n{'='*80}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(f"{'Metric':<25} {'Train':<15} {'Test':<15}")
        print(f"{'-'*80}")
        print(f"{'Reconstruction Loss':<25} {results['train']['avg_reconstruction_loss']:<15.4f} {results['test']['avg_reconstruction_loss']:<15.4f}")
        print(f"{'Token Accuracy':<25} {results['train']['avg_token_level_accuracy']:<15.4f} {results['test']['avg_token_level_accuracy']:<15.4f}")
        print(f"{'Perplexity':<25} {results['train']['avg_perplexity']:<15.4f} {results['test']['avg_perplexity']:<15.4f}")
        print(f"{'='*80}")
    
    print(f"\nDataset reconstruction metrics computation completed!")
    
    # Decode and display tracked examples
    if tokenizer is not None and k is not None:
        print(f"\n{'='*80}")
        print(f"TRACKED EXAMPLES ANALYSIS")
        print(f"{'='*80}")
        
        def decode_tokens(tokens, mask=None):
            if mask is not None:
                tokens = tokens[mask.bool()]
            try:
                return tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception as e:
                return f"[Decode error: {e}, tokens: {tokens.tolist()}]"
        
        def display_examples(category_name, examples, split_name):
            print(f"\n{category_name.upper()} EXAMPLES ({split_name.upper()}):")
            print(f"{'='*60}")
            for i, example in enumerate(examples):
                prompt_text = decode_tokens(example['prompt'], example['prompt_mask'])
                
                print(f"\nExample {i+1} (Example {example['example_idx']}):")
                print(f"Average Reconstruction Loss: {example['avg_loss']:.4f}")
                print(f"Average Perplexity: {example['avg_perplexity']:.4f}")
                print(f"Average Token-Level Accuracy: {example['individual_token_accuracies'].mean().item():.4f}")
                
                # Prepare CoT texts for side-by-side comparison
                num_thoughts = example['cots'].size(0)
                cot_gt_texts = []
                cot_recon_texts = []
                
                for j in range(num_thoughts):
                    cot = example['cots'][j]
                    recon_cot = torch.argmax(example['recon_logits'][j], dim=-1)
                    cot_mask = example['cot_mask'][j] if example['cot_mask'] is not None else None
                    cot_text = decode_tokens(cot, cot_mask)
                    recon_cot_text = decode_tokens(recon_cot, cot_mask)
                    cot_gt_texts.append(cot_text)
                    cot_recon_texts.append(recon_cot_text)
                
                # Compute metrics using available data
                backpointer_gt = example.get('backpointers', None)
                backpointer_logits = example.get('bp_logits', None)
                if backpointer_gt is not None:
                    backpointer_gt = backpointer_gt.unsqueeze(0)
                if backpointer_logits is not None:
                    backpointer_logits = backpointer_logits.unsqueeze(0)
                
                metrics = compute_cot_reconstruction_metrics(
                    example['cots'].unsqueeze(0), example['recon_logits'].unsqueeze(0), example['cot_mask'].unsqueeze(0), 
                    backpointer_gt, backpointer_logits
                )

                # Use print_demonstration_results for side-by-side comparison
                print_demonstration_results(
                    example_num=i+1,
                    prompt_text=prompt_text,
                    cot_gt_texts=cot_gt_texts,
                    cot_tf_texts=cot_recon_texts if not ar_gen else None,
                    cot_ar_texts=cot_recon_texts if     ar_gen else None,  # No auto-regressive in this context
                    tf_metrics=metrics if not ar_gen else None,
                    ar_metrics=metrics if     ar_gen else None,
                    ar_indices=None,
                    tf_indices=None,
                    vq_loss_tf=None,
                    perplexity_tf=None,
                    vq_loss_ar=None,
                    perplexity_ar=None,
                    recon_loss_tf=example['avg_loss'] if not ar_gen else None,
                    recon_loss_ar=example['avg_loss'] if     ar_gen else None,
                    num_thoughts=num_thoughts
                )
        
        # Display examples for each split
        for split in ['train', 'test']:
            if split in results:
                split_results = results[split]
                if 'tracked_examples' in split_results:
                    tracked_examples = split_results['tracked_examples']
                    
                    for category in ['highest_loss', 'lowest_loss', 'highest_perplexity', 'lowest_perplexity', 'random']:
                        if category in tracked_examples:
                            display_examples(category, tracked_examples[category], split)
    
    return results


def print_demonstration_results(
    example_num, prompt_text, cot_gt_texts, cot_tf_texts, cot_ar_texts,
    tf_metrics=None, ar_metrics=None, tf_indices=None, ar_indices=None,
    vq_loss_tf=None, perplexity_tf=None, vq_loss_ar=None, perplexity_ar=None,
    recon_loss_tf=None, recon_loss_ar=None, num_thoughts=None
):
    """
    Print demonstration results in a formatted way.
    
    Args:
        example_num: Example number (1-indexed)
        prompt_text: The prompt text
        cot_gt_texts: List of ground truth CoT texts
        cot_tf_texts: List of teacher-forced CoT texts (can be None)
        cot_ar_texts: List of auto-regressive CoT texts (can be None)
        tf_metrics: Teacher-forced metrics dict (optional)
        ar_metrics: Auto-regressive metrics dict (optional)
        tf_indices: Teacher-forced indices (optional)
        ar_indices: Auto-regressive indices (optional)
        vq_loss_tf: Teacher-forced VQ loss (optional)
        perplexity_tf: Teacher-forced perplexity (optional)
        vq_loss_ar: Auto-regressive VQ loss (optional)
        perplexity_ar: Auto-regressive perplexity (optional)
        recon_loss_tf: Teacher-forced reconstruction loss (optional)
        recon_loss_ar: Auto-regressive reconstruction loss (optional)
        num_thoughts: Number of thoughts (if None, inferred from cot_gt_texts)
    """
    if num_thoughts is None:
        num_thoughts = len(cot_gt_texts)
    
    print(f"\n--- Example {example_num} ---")
    print(f"Prompt: {prompt_text}")
    
    print(f"\n{'='*120}")
    print(f"SIDE-BY-SIDE COMPARISON FOR EXAMPLE {example_num}")
    print(f"{'='*120}")
    
    # Print metrics for teacher-forced
    if vq_loss_tf is not None and perplexity_tf is not None:
        print(f"Teacher Forcing - VQ Loss: {vq_loss_tf.item():.4f}, Perplexity: {perplexity_tf.item():.2f}")
        if tf_indices is not None:
            unique_indices = torch.unique(tf_indices).numel()
            print(f"  Codebook usage: {unique_indices} unique indices out of {tf_indices.numel()} total")
        if recon_loss_tf is not None:
            print(f"  Reconstruction Loss: {recon_loss_tf.item():.4f}")
    
    # Print metrics for auto-regressive
    if vq_loss_ar is not None and perplexity_ar is not None:
        print(f"Auto-regressive - VQ Loss: {vq_loss_ar.item():.4f}, Perplexity: {perplexity_ar.item():.2f}")
        if ar_indices is not None:
            unique_indices = torch.unique(ar_indices).numel()
            print(f"  Codebook usage: {unique_indices} unique indices out of {ar_indices.numel()} total")
        if recon_loss_ar is not None:
            print(f"  Reconstruction Loss: {recon_loss_ar.item():.4f}")
    
    # Print detailed metrics for teacher-forced
    if tf_metrics is not None:
        tf_token_acc = tf_metrics['token_level_accuracies'][0]
        tf_seq_accs = tf_metrics['sequence_level_accuracies']
        print(f"\n[Teacher-forced] Token-level accuracy (per CoT, per example):")
        print(" | ".join(
            [f"CoT {i}: {tf_token_acc[i].item():.4f}" for i in range(tf_token_acc.size(0))]
        ))
        print(f"[Teacher-forced] Sequence-level accuracy (thresholds):")
        for thresh, count in tf_seq_accs.items():
            print(f"  Threshold {int(thresh*100)}%: {int(count)} sequences correct")
        mean_per_example_tf = tf_token_acc.mean().item() 
        print(f"[Teacher-forced] Mean token-level accuracy per example:")
        print(f"{mean_per_example_tf:.4f}")
        print(f"[Teacher-forced] Perplexity (per CoT, per example):")
        tf_ppl = tf_metrics['perplexities'][0]
        print(" | ".join([f"CoT {i}: {tf_ppl[i]:.4f}" for i in range(tf_ppl.size(0))]))
        
        # Print backpointer metrics if available (for compress beam search mode)
        if 'backpointer_accuracies' in tf_metrics:
            tf_bp_acc = tf_metrics['backpointer_accuracies'][0]
            tf_bp_loss = tf_metrics['backpointer_losses'][0]
            print(f"[Teacher-forced] Backpointer accuracy (per CoT, per example):")
            print(" | ".join([f"CoT {i}: {tf_bp_acc[i]:.4f}" for i in range(tf_bp_acc.size(0))]))
            print(f"[Teacher-forced] Backpointer loss (per CoT, per example):")
            print(" | ".join([f"CoT {i}: {tf_bp_loss[i]:.4f}" for i in range(tf_bp_loss.size(0))]))
    
    # Print detailed metrics for auto-regressive
    if ar_metrics is not None:
        ar_token_acc = ar_metrics['token_level_accuracies'][0]
        ar_seq_accs = ar_metrics['sequence_level_accuracies']
        print(f"\n[Auto-regressive] Token-level accuracy (per CoT, per example):")
        print(" | ".join(
            [f"CoT {i}: {ar_token_acc[i].item():.4f}" for i in range(ar_token_acc.size(0))]
        ))
        print(f"[Auto-regressive] Sequence-level accuracy (thresholds):")
        for thresh, count in ar_seq_accs.items():
            print(f"  Threshold {int(thresh*100)}%: {int(count)} sequences correct")
        mean_per_example_ar = ar_token_acc.mean().item()
        print(f"[Auto-regressive] Mean token-level accuracy per example:")
        print(f"{mean_per_example_ar:.4f}")
        print(f"[Auto-regressive] Perplexity (per CoT, per example):")
        ar_ppl = ar_metrics['perplexities'][0]
        print(" | ".join([f"CoT {i}: {ar_ppl[i]:.4f}" for i in range(ar_ppl.size(0))]))
        
        # Print backpointer metrics if available (for compress beam search mode)
        if 'backpointer_accuracies' in ar_metrics:
            ar_bp_acc = ar_metrics['backpointer_accuracies'][0]
            ar_bp_loss = ar_metrics['backpointer_losses'][0]
            print(f"[Auto-regressive] Backpointer accuracy (per CoT, per example):")
            print(" | ".join([f"CoT {i}: {ar_bp_acc[i]:.4f}" for i in range(ar_bp_acc.size(0))]))
            print(f"[Auto-regressive] Backpointer loss (per CoT, per example):")
            print(" | ".join([f"CoT {i}: {ar_bp_loss[i]:.4f}" for i in range(ar_bp_loss.size(0))]))
    
    # Helper function for chunking text
    def chunk_text(text, chunk_size=20):
        if not text:
            return []
        lines = text.split('\n')
        chunks = []
        for line in lines:
            line = line.strip()
            if not line: continue
            words = line.split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk:
                    chunks.append(chunk)
        return chunks
    
    # Print side-by-side text comparison for each CoT
    for j in range(num_thoughts):
        print(f"\n--- CoT {j+1} ---")
        cot_gt_text = cot_gt_texts[j] if j < len(cot_gt_texts) else ""
        cot_tf_text = cot_tf_texts[j] if cot_tf_texts and j < len(cot_tf_texts) else "FAILED"
        cot_ar_text = cot_ar_texts[j] if cot_ar_texts and j < len(cot_ar_texts) else "FAILED"
        
        gt_chunks = chunk_text(cot_gt_text)
        tf_chunks = chunk_text(cot_tf_text)
        ar_chunks = chunk_text(cot_ar_text)
        max_chunks = max(len(gt_chunks), len(tf_chunks), len(ar_chunks))
        
        for chunk_idx in range(max_chunks):
            gt_chunk = gt_chunks[chunk_idx] if chunk_idx < len(gt_chunks) else ""
            tf_chunk = tf_chunks[chunk_idx] if chunk_idx < len(tf_chunks) else ""
            ar_chunk = ar_chunks[chunk_idx] if chunk_idx < len(ar_chunks) else ""
            max_length = max(len(gt_chunk), len(tf_chunk), len(ar_chunk))
            gt_chunk_padded = gt_chunk.ljust(max_length)
            tf_chunk_padded = tf_chunk.ljust(max_length)
            ar_chunk_padded = ar_chunk.ljust(max_length)
            print(f"Original:        {gt_chunk_padded}")
            if cot_tf_texts:
                print(f"Teacher Forced:  {tf_chunk_padded}")
            if cot_ar_texts:
                print(f"Auto-regressive: {ar_chunk_padded}")
            if chunk_idx < max_chunks - 1:
                print("-" * 60)
    
    print("\n" + "="*120)


def demonstrate_model_from_checkpoint(checkpoint_path: str, 
                                    data_dir: str,
                                    num_examples: int = 3,
                                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                    use_vq: bool = True,
                                    model_type: str = "GPT2VQVAE",
                                    seed: int = 42,
                                    do_figure_analyses: bool = True,
                                    **kwargs):
    """
    Demonstrate GPT2VQVAE, EnhancedGPT2VQVAE, or SimpleGPT2VQVAE model generation capabilities from a checkpoint.
    """
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    try:
        if model_type == "EnhancedGPT2VQVAE":
            model = EnhancedGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded EnhancedGPT2VQVAE model")
        elif model_type == "SimpleGPT2VQVAE":
            model = SimpleGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded SimpleGPT2VQVAE model with use_vq={use_vq}")
        else:
            model = GPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded GPT2VQVAE model")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Make sure the checkpoint file exists and contains the required model configuration.")
        return
    model.eval()
    num_embeddings = model.vector_quantizer.num_embeddings
    num_thoughts = 1 if model_type == "SimpleGPT2VQVAE" else model.num_thoughts
    print(f"Model configured for {num_thoughts} parallel CoT sequences and {num_embeddings} embeddings")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded GPT2 tokenizer")
    if data_dir == "data/GSM8K":
        data_dir = f"data/GSM8K/128_128/batch_{num_thoughts}"
    print(f"Loading example data from: {data_dir}")
    try:
        out = load_training_data(
            data_dir=data_dir, max_samples=num_examples, num_thoughts=num_thoughts, seed=seed
        )
        # without backpointers
        if len(out) == 8:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
                test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = out
            train_backpointers, test_backpointers = None, None
        # with backpointers (required for compress_beam_search mode)
        else: # len(out) == 10:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, train_backpointers, \
                test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, test_backpointers = out
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    run_demonstration_on_split(model, tokenizer, num_examples, num_thoughts, num_embeddings, use_vq,
                              train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, "train", checkpoint_path, 
                              do_figure_analyses, train_backpointers)
    run_demonstration_on_split(model, tokenizer, num_examples, num_thoughts, num_embeddings, use_vq,
                              test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, "test", checkpoint_path, 
                              do_figure_analyses, test_backpointers)

def demonstrate_custom_prompt_cot(checkpoint_path: str,
                                 prompt_file: str = "test_prompt.txt",
                                 cot_file: str = "test_cot.txt",
                                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                 use_vq: bool = True,
                                 model_type: str = "GPT2VQVAE",
                                 do_figure_analyses: bool = True,
                                 **kwargs):
    """
    Demonstrate GPT2VQVAE, EnhancedGPT2VQVAE, or SimpleGPT2VQVAE model generation capabilities from a checkpoint
    using custom prompt and CoT from plain text files.
    """
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    try:
        if model_type == "EnhancedGPT2VQVAE":
            model = EnhancedGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded EnhancedGPT2VQVAE model")
        elif model_type == "SimpleGPT2VQVAE":
            model = SimpleGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded SimpleGPT2VQVAE model with use_vq={use_vq}")
        else:
            model = GPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded GPT2VQVAE model")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Make sure the checkpoint file exists and contains the required model configuration.")
        return
    model.eval()
    num_embeddings = model.vector_quantizer.num_embeddings
    num_thoughts = 1 if model_type == "SimpleGPT2VQVAE" else model.num_thoughts
    print(f"Model configured for {num_thoughts} parallel CoT sequences and {num_embeddings} embeddings")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded GPT2 tokenizer")
    print(f"Loading prompt from: {prompt_file}")
    print(f"Loading CoT from: {cot_file}")
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        with open(cot_file, 'r', encoding='utf-8') as f:
            cot_text = f.read().strip()
        print(f"Prompt: {prompt_text}")
        print(f"CoT: {cot_text}")
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    if tokenizer is None:
        print("Error: Tokenizer not available")
        return
    try:
        print("For now, we will pad both the prompt and cot to length 128.")
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt', padding='max_length', max_length=128)
        prompt_length = prompt_tokens.shape[1]
        cot_tokens = tokenizer.encode(cot_text, return_tensors='pt', padding='max_length', max_length=128)
        cot_length = cot_tokens.shape[1]
        print(f"Prompt tokens: {prompt_tokens.shape} (length: {prompt_length})")
        print(f"CoT tokens: {cot_tokens.shape} (length: {cot_length})")
        n_positions = model.encoder_config.n_positions
        total_length = prompt_length + cot_length
        if total_length > n_positions:
            print(f"Warning: Total sequence length {total_length} exceeds model's n_positions {n_positions}")
            print("Truncating sequences...")
            max_cot_length = n_positions - prompt_length
            if max_cot_length <= 0:
                print("Error: Prompt is too long, cannot fit CoT")
                return
            cot_tokens = cot_tokens[:, :max_cot_length]
            cot_length = max_cot_length
            print(f"Truncated CoT to {cot_length} tokens")
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return
    prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
    cot_mask = torch.ones_like(cot_tokens, dtype=torch.bool)
    cot_tokens_expanded = cot_tokens.unsqueeze(0).repeat(1, num_thoughts, 1)
    cot_mask_expanded = cot_mask.unsqueeze(0).repeat(1, num_thoughts, 1)
    prompt = prompt_tokens.to(device)
    cot_gt = cot_tokens_expanded.to(device)
    prompt_mask_ex = prompt_mask.to(device)
    cot_mask_ex = cot_mask_expanded.to(device)
    run_demonstration_on_split(
        model, tokenizer, 1, num_thoughts, num_embeddings, use_vq,
        prompt, cot_gt, prompt_mask_ex, cot_mask_ex, "custom", checkpoint_path, 
        do_figure_analyses # won't add backpointers for now coz I'm lazy
    )
    print("\nCustom prompt-CoT demonstration completed!")

if __name__ == "__main__":
    two_paths = [
        # r"checkpoints/asw_embsum/big/two_thoughts/512/best_model_normal_recon_epoch_22.pt",
        r"checkpoints/asw_embsum/big/two_thoughts/512/checkpoint_epoch_40.pt",
        
        # r"checkpoints/asw_embsum/big/two_thoughts/1024/best_model_normal_recon_epoch_30.pt",
        r"checkpoints/asw_embsum/big/two_thoughts/1024/checkpoint_epoch_40.pt",
        
        # r"checkpoints/asw_embsum/big/two_thoughts/2048/best_model_normal_recon_epoch_30.pt",
        r"checkpoints/asw_embsum/big/two_thoughts/2048/checkpoint_epoch_40.pt",
        
        # r"checkpoints/asw_embsum/big/two_thoughts/4096/best_model_normal_recon_epoch_24.pt"
        r"checkpoints/asw_embsum/big/two_thoughts/4096/checkpoint_epoch_40.pt"
    ]

    four_paths = [
        # r"checkpoints/asw_embsum/big/four_thoughts/512/best_model_normal_recon_epoch_39.pt",
        # r"checkpoints/asw_embsum/big/four_thoughts/512/checkpoint_epoch_40.pt",
        
        # # r"checkpoints/asw_embsum/big/four_thoughts/1024/best_model_normal_recon_epoch_25.pt",
        # r"checkpoints/asw_embsum/big/four_thoughts/1024/checkpoint_epoch_40.pt",
        
        # # r"checkpoints/asw_embsum/big/four_thoughts/2048/best_model_normal_recon_epoch_23.pt",
        # r"checkpoints/asw_embsum/big/four_thoughts/2048/checkpoint_epoch_40.pt",
        
        # r"checkpoints/asw_embsum/big/four_thoughts/4096/best_model_reinitialization_recon_epoch_19.pt"
        r"checkpoints/asw_embsum/big/four_thoughts/4096/checkpoint_epoch_40.pt"
    ]
    
    TRAIN_SAMPLES = 75
    TEST_SAMPLES = 75
    BATCH_SIZE_2 = 200
    BATCH_SIZE_4 = 25  # Reduced from 25 to 10
    K = 3
    AR_GEN = True

    if False:
        for path in two_paths:
            splitted = path.split('/')
            print(f"Processing {splitted[3]}, code size {splitted[4]}.")
            compute_dataset_reconstruction_metrics_from_checkpoint_and_keep_examples(
                checkpoint_path=path,
                data_dir=None, #assigns default
                split_name="both",  # "train", "test", or "both"
                num_examples_train=TRAIN_SAMPLES, #all
                num_examples_test=TEST_SAMPLES, #all
                batch_size=BATCH_SIZE_2,
                ar_gen=AR_GEN,
                k=K,  # Number of examples to keep for each category
                device= "cuda:0",
                use_vq=True,
                seed=42
            )
    else:
        for path in four_paths:
            splitted = path.split('/')
            print(f"Processing {splitted[3]}, code size {splitted[4]}.")
            compute_dataset_reconstruction_metrics_from_checkpoint_and_keep_examples(
                checkpoint_path=path,
                data_dir=None, #assigns default
                split_name="both",  # "train", "test", or "both"
                num_examples_train=TRAIN_SAMPLES, #all
                num_examples_test=TEST_SAMPLES, #all
                batch_size=BATCH_SIZE_4,
                ar_gen=AR_GEN,
                k=K,  # Number of examples to keep for each category
                device= "cuda:1",
                use_vq=True,
                seed=42
            )