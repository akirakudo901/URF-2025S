# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import os
import sys

import torch
from transformers import GPT2Tokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from latent_visualization_analysis import LatentVisualizationAnalyzer
from trainer.train_utils import compute_reconstruction_loss, create_codebook_usage_heatmap, load_training_data
from vqvae_gpt2 import GPT2VQVAE
from vqvae_gpt2_simple import SimpleGPT2VQVAE
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE


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
                              prompt_sequences, cot_sequences, prompt_mask, cot_mask, split_name, checkpoint_path, 
                              do_figure_analyses : bool=True):
    print("\n" + "="*80)
    print(f"GENERATION DEMONSTRATION ON {split_name.upper()} DATA")
    print("="*80)
    all_indices_tf = []
    all_indices_ar = []

    device = next(model.parameters()).device

    with torch.no_grad():
        tf_mean_per_example_accs = []
        ar_mean_per_example_accs = []
        for i in range(min(num_examples, len(prompt_sequences))):
            print(f"\n--- Example {i+1} ---")
            prompt = prompt_sequences[i:i+1].to(device)  # [1, K]
            cot_gt = cot_sequences[i:i+1].to(device)     # [1, M, L]
            prompt_mask_ex = prompt_mask[i:i+1].to(device) if prompt_mask is not None else None
            cot_mask_ex = cot_mask[i:i+1].to(device) if cot_mask is not None else None
            def decode_tokens(tokens, mask=None):
                if tokenizer is None:
                    return f"[Tokens: {tokens.tolist()}]"
                if mask is not None:
                    tokens = tokens[mask.bool()]
                try:
                    return tokenizer.decode(tokens, skip_special_tokens=True)
                except Exception as e:
                    return f"[Decode error: {e}, tokens: {tokens.tolist()}]"
            prompt_text = decode_tokens(prompt[0], prompt_mask_ex[0] if prompt_mask_ex is not None else None)
            print(f"Prompt: {prompt_text}")
            # Teacher forcing
            try:
                model_inputs = {
                    'prompt': prompt,
                    'cot_sequences': cot_gt, 
                    'cot_mask': cot_mask_ex,
                    'prompt_mask': prompt_mask_ex,
                    'inference': False,  # Teacher forcing
                    'quantize_cot_only': True
                }
                if hasattr(model, 'use_vq'):
                    model_inputs['use_vq'] = use_vq
                _, output_logits_tf, vq_loss_tf, perplexity_tf, indices_tf = model(**model_inputs)
                predicted_tokens_tf = torch.argmax(output_logits_tf, dim=-1)  # [B, M, L]
                if indices_tf is not None:
                    all_indices_tf.append(indices_tf.flatten())
            except Exception as e:
                print(f"Teacher forcing generation failed: {e}")
                predicted_tokens_tf = None
                vq_loss_tf = None
                perplexity_tf = None
                indices_tf = None
                raise # TODO REMOVE
            # Auto-regressive
            try:
                model_inputs = {
                    'prompt': prompt,
                    'cot_sequences': cot_gt,
                    'cot_mask': cot_mask_ex,
                    'prompt_mask': prompt_mask_ex,
                    'inference': True,  # Auto-regressive
                    'quantize_cot_only': True
                }
                if hasattr(model, 'use_vq'):
                    model_inputs['use_vq'] = use_vq
                output_sequences_ar, output_logits_ar, vq_loss_ar, perplexity_ar, indices_ar = model(**model_inputs)
                if indices_ar is not None:
                    all_indices_ar.append(indices_ar.flatten())
            except Exception as e:
                print(f"Auto-regressive generation failed: {e}")
                output_sequences_ar = None
                vq_loss_ar = None
                perplexity_ar = None
                indices_ar = None
            print(f"\n{'='*120}")
            print(f"SIDE-BY-SIDE COMPARISON FOR EXAMPLE {i+1}")
            print(f"{'='*120}")
            if vq_loss_tf is not None and perplexity_tf is not None:
                print(f"Teacher Forcing - VQ Loss: {vq_loss_tf.item():.4f}, Perplexity: {perplexity_tf.item():.2f}")
                if indices_tf is not None:
                    unique_indices = torch.unique(indices_tf).numel()
                    print(f"  Codebook usage: {unique_indices} unique indices out of {indices_tf.numel()} total")
                if output_logits_tf is not None and cot_gt is not None and cot_mask_ex is not None:
                    recon_loss_tf = compute_reconstruction_loss(output_logits_tf, cot_gt, cot_mask_ex)
                    print(f"  Reconstruction Loss: {recon_loss_tf.item():.4f}")
            if vq_loss_ar is not None and perplexity_ar is not None:
                print(f"Auto-regressive - VQ Loss: {vq_loss_ar.item():.4f}, Perplexity: {perplexity_ar.item():.2f}")
                if indices_ar is not None:
                    unique_indices = torch.unique(indices_ar).numel()
                    print(f"  Codebook usage: {unique_indices} unique indices out of {indices_ar.numel()} total")
                if output_logits_ar is not None and cot_gt is not None and cot_mask_ex is not None:
                    recon_loss_ar = compute_reconstruction_loss(output_logits_ar, cot_gt, cot_mask_ex)
                    print(f"  Reconstruction Loss: {recon_loss_ar.item():.4f}")
            # Compute and print reconstruction accuracies for all CoTs and examples
            if predicted_tokens_tf is not None and output_logits_tf is not None:
                _, tf_accuracies = compute_cot_reconstruction_metrics(cot_gt, output_logits_tf, cot_mask_ex)
                tf_accuracies = tf_accuracies[0]
                print(f"\n[Teacher-forced] Sequence-level reconstruction accuracy (per CoT, per example):")
                print(" | ".join(
                    [f"CoT {i}: {tf_accuracies[i].item():.4f}" for i in range(tf_accuracies.size(0))]
                    ))
                mean_per_example_tf = tf_accuracies.mean().item() 
                print(f"[Teacher-forced] Mean sequence-level reconstruction accuracy per example:")
                print(f"{mean_per_example_tf:.4f}")
                tf_mean_per_example_accs.append(mean_per_example_tf)
                
            if output_sequences_ar is not None and output_logits_ar is not None:
                _, ar_accuracies = compute_cot_reconstruction_metrics(cot_gt, output_logits_ar, cot_mask_ex)
                ar_accuracies = ar_accuracies[0]
                print(f"\n[Auto-regressive] Sequence-level reconstruction accuracy (per CoT, per example):")
                print(" | ".join(
                    [f"CoT {i}: {ar_accuracies[i].item():.4f}" for i in range(ar_accuracies.size(0))]
                    ))
                mean_per_example_ar = ar_accuracies.mean().item()
                print(f"[Auto-regressive] Mean sequence-level reconstruction accuracy per example:")
                print(f"{mean_per_example_ar:.4f}")
                ar_mean_per_example_accs.append(mean_per_example_ar)

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
            for j in range(num_thoughts):
                print(f"\n--- CoT {j+1} ---")
                cot_gt_text = decode_tokens(cot_gt[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                if predicted_tokens_tf is not None:
                    cot_tf_text = decode_tokens(predicted_tokens_tf[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                else:
                    cot_tf_text = "FAILED"
                if output_sequences_ar is not None:
                    cot_ar_text = decode_tokens(output_sequences_ar[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                else:
                    cot_ar_text = "FAILED"
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
                    print(f"Teacher Forced:  {tf_chunk_padded}")
                    print(f"Auto-regressive: {ar_chunk_padded}")
                    if chunk_idx < max_chunks - 1:
                        print("-" * 60)
            print("\n" + "="*120)
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
                                
                                # Add token-latent alignment visualization
                                # visualize_token_latent_alignment(
                                #     original_token_ids=original_tokens_raw,
                                #     latent_indices=indices_tensor[0],
                                #     reconstructed_token_ids=token_tensor[0, j],
                                #     tokenizer=tokenizer,
                                #     mode_name=f"CoT {j+1} {mode_name}",
                                #     max_tokens_per_line=15
                                # )
                            else:
                                print(f"  {fail_message} for CoT {j+1}, skipping latent visualization")

                        # # Teacher-forced latent visualization
                        # run_latent_visualization(
                        #     mode_name="Teacher-forced",
                        #     token_tensor=predicted_tokens_tf,
                        #     indices_tensor=indices_tf,
                        #     file_suffix="_tf",
                        #     fail_message="Teacher forcing failed"
                        # )

                        # # Auto-regressive latent visualization
                        # run_latent_visualization(
                        #     mode_name="Auto-regressive",
                        #     token_tensor=output_sequences_ar,
                        #     indices_tensor=indices_ar,
                        #     file_suffix="_ar",
                        #     fail_message="Auto-regression failed"
                        # )

                    print(f"\nLatent visualization completed for {split_name} data!")

                    # Multi-CoT side-by-side alignment for teacher-forced
                    if predicted_tokens_tf is not None and indices_tf is not None:
                        visualize_token_latent_alignment_multi_cot(
                            all_original_token_ids=[cot_gt[0, j] for j in range(num_thoughts)],
                            all_latent_indices=[indices_tf[0] for _ in range(num_thoughts)],
                            all_reconstructed_token_ids=[predicted_tokens_tf[0, j] for j in range(num_thoughts)],
                            tokenizer=tokenizer,
                            mode_name="Teacher-forced",
                            max_tokens_per_line=15
                        )
                    # Multi-CoT side-by-side alignment for auto-regressive
                    if output_sequences_ar is not None and indices_ar is not None:
                        visualize_token_latent_alignment_multi_cot(
                            all_original_token_ids=[cot_gt[0, j] for j in range(num_thoughts)],
                            all_latent_indices=[indices_ar[0] for _ in range(num_thoughts)],
                            all_reconstructed_token_ids=[output_sequences_ar[0, j] for j in range(num_thoughts)],
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
        if tf_mean_per_example_accs:
            overall_tf = sum(tf_mean_per_example_accs) / len(tf_mean_per_example_accs)
            print(f"\n[Teacher-forced] Overall mean sequence-level reconstruction accuracy over all examples: {overall_tf:.4f}")
        if ar_mean_per_example_accs:
            overall_ar = sum(ar_mean_per_example_accs) / len(ar_mean_per_example_accs)
            print(f"[Auto-regressive] Overall mean sequence-level reconstruction accuracy over all examples: {overall_ar:.4f}")
        
        if use_vq and do_figure_analyses:
            print(f"\n{'='*80}")
            print(f"GENERATING COMPREHENSIVE CODEBOOK USAGE HEATMAPS FOR {split_name.upper()} DATA")
            print(f"{'='*80}")
            if all_indices_tf:
                combined_indices_tf = torch.cat(all_indices_tf, dim=0)
                print(f"Teacher Forcing - Total indices: {combined_indices_tf.numel()}")
                print(f"Teacher Forcing - Unique indices: {torch.unique(combined_indices_tf).numel()}")
                heatmap_path_tf = os.path.join(os.path.dirname(checkpoint_path), f"codebook_usage_tf_comprehensive_{split_name}.png")
                counts_tf = torch.bincount(combined_indices_tf[combined_indices_tf < num_embeddings], 
                                          minlength=num_embeddings)
                create_codebook_usage_heatmap(
                    counts_tf.cpu().numpy(), 
                    num_embeddings=num_embeddings,
                    title=f"Teacher Forcing Codebook Usage - All {num_examples} Examples ({split_name})",
                    save_path=heatmap_path_tf
                )
            if all_indices_ar:
                combined_indices_ar = torch.cat(all_indices_ar, dim=0)
                print(f"Auto-regressive - Total indices: {combined_indices_ar.numel()}")
                print(f"Auto-regressive - Unique indices: {torch.unique(combined_indices_ar).numel()}")
                heatmap_path_ar = os.path.join(os.path.dirname(checkpoint_path), f"codebook_usage_ar_comprehensive_{split_name}.png")
                counts_ar = torch.bincount(combined_indices_ar[combined_indices_ar < num_embeddings], 
                                          minlength=num_embeddings)
                create_codebook_usage_heatmap(
                    counts_ar.cpu().numpy(), 
                    num_embeddings=num_embeddings,
                    title=f"Auto-regressive Codebook Usage - All {num_examples} Examples ({split_name})",
                    save_path=heatmap_path_ar
                )
        print("\nDemonstration completed for {} data!".format(split_name))


def compute_cot_reconstruction_metrics(ground_truth_cots, predicted_logits, cot_mask=None):
    """
    Compute reconstruction loss and accuracy for each CoT sequence using vectorized operations.
    
    Args:
        ground_truth_cots: Tensor of shape [batch, num_thoughts, seq_len]
        predicted_logits: Tensor of shape [batch, num_thoughts, seq_len, vocab_size]
        cot_mask: Optional mask tensor of shape [batch, num_thoughts, seq_len]
        
    Returns:
        reconstruction_losses: Tensor of shape [batch, num_thoughts] with loss per sequence
        reconstruction_accuracies: Tensor of shape [batch, num_thoughts] with accuracy per sequence
    """
    batch_size, num_thoughts, seq_len = ground_truth_cots.shape
    
    # Get predicted tokens for all CoTs at once
    predicted_tokens = torch.argmax(predicted_logits, dim=-1)  # [batch, num_thoughts, seq_len]
    
    # Initialize tensors for results
    reconstruction_losses = torch.zeros(batch_size, num_thoughts, device=ground_truth_cots.device)
    reconstruction_accuracies = torch.zeros(batch_size, num_thoughts, device=ground_truth_cots.device)
    
    # Compute reconstruction loss and accuracy for each CoT
    for j in range(num_thoughts):
        for b in range(batch_size):
            # Extract the b-th batch, j-th CoT sequence
            cot_gt = ground_truth_cots[b:b+1, j:j+1, :]  # [1, 1, seq_len]
            cot_logits = predicted_logits[b:b+1, j:j+1, :, :]  # [1, 1, seq_len, vocab_size]
            cot_mask_j = cot_mask[b:b+1, j:j+1, :] if cot_mask is not None else None  # [1, 1, seq_len]
            
            # Compute reconstruction loss for this sequence
            recon_loss = compute_reconstruction_loss(cot_logits, cot_gt, cot_mask_j)
            reconstruction_losses[b, j] = recon_loss.item()
            
            # Compute reconstruction accuracy for this sequence
            cot_pred = predicted_tokens[b:b+1, j:j+1, :]  # [1, 1, seq_len]
            
            if cot_mask_j is not None:
                # Apply mask to both ground truth and predictions
                masked_gt = cot_gt[cot_mask_j.bool()]
                masked_pred = cot_pred[cot_mask_j.bool()]
            else:
                masked_gt = cot_gt.flatten()
                masked_pred = cot_pred.flatten()
            
            # Compute accuracy for this sequence
            if masked_gt.numel() > 0:
                accuracy = (masked_gt == masked_pred).float().mean().item()
            else:
                accuracy = 0.0
            reconstruction_accuracies[b, j] = accuracy
    
    return reconstruction_losses, reconstruction_accuracies


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
        train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
        test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = load_training_data(
            data_dir=data_dir, max_samples=num_examples, num_thoughts=num_thoughts, seed=seed
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    run_demonstration_on_split(model, tokenizer, num_examples, num_thoughts, num_embeddings, use_vq,
                              train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, "train", checkpoint_path, 
                              do_figure_analyses)
    run_demonstration_on_split(model, tokenizer, num_examples, num_thoughts, num_embeddings, use_vq,
                              test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, "test", checkpoint_path, 
                              do_figure_analyses)

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
        do_figure_analyses
    )
    print("\nCustom prompt-CoT demonstration completed!")