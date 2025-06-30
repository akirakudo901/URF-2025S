# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from typing import Optional

# Import the base class and shared functions from vqvae_gpt2.py
from vqvae_gpt2 import (
    GPT2VQVAE,
    CustomGPT2LMHeadModel,
    VectorQuantizer,
    create_cross_attention_mask
)

class SimpleGPT2VQVAE(GPT2VQVAE):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, n_positions=1024, 
                 use_pretrained_encoder=True, use_pretrained_decoder=True,
                 pretrained_model_name="gpt2",
                 # Unified parameters (applied to both encoder and decoder if specified)
                 n_layer=12, n_head=12, n_inner=None, dropout=0.1, activation_function="gelu",
                 # Encoder-specific parameters (take precedence over unified if specified)
                 encoder_n_layer=None, encoder_n_head=None, encoder_n_inner=None,
                 encoder_dropout=None, encoder_activation_function=None,
                 # Decoder-specific parameters (take precedence over unified if specified)
                 decoder_n_layer=None, decoder_n_head=None, decoder_n_inner=None,
                 decoder_dropout=None, decoder_activation_function=None):
        """
        Simple GPT2-based VQ-VAE model that uses GPT2 as both encoder and decoder.
        This version handles single chain-of-thought per prompt (no aggregation needed).
        
        Args:
            # Shared parameters (must be the same for both encoder and decoder)
            vocab_size (int): Vocabulary size
            d_model (int): Model dimension (default: 768 for GPT2)
            n_positions (int): Maximum sequence length for GPT2 (default: 1024)
            
            # VQ-VAE specific parameters
            num_embeddings (int): VQ codebook size
            commitment_cost (float): VQ commitment cost
            
            # Pretrained model settings
            use_pretrained_encoder (bool): Whether to load pretrained weights for encoder
            use_pretrained_decoder (bool): Whether to load pretrained weights for decoder
            pretrained_model_name (str): Name of pretrained model to load (default: "gpt2")
            
            # Unified parameters (applied to both encoder and decoder)
            n_layer (int): Number of hidden layers for both encoder and decoder (default: 12)
            n_head (int): Number of attention heads for both encoder and decoder (default: 12)
            n_inner (int, optional): Dimensionality of inner feed-forward layers for both encoder and decoder
            dropout (float): Dropout probability for both encoder and decoder (default: 0.1)
            activation_function (str): Activation function for both encoder and decoder (default: "gelu")
            
            # Encoder-specific parameters (take precedence over unified if specified)
            encoder_n_layer (int, optional): Number of hidden layers in the encoder
            encoder_n_head (int, optional): Number of attention heads for encoder
            encoder_n_inner (int, optional): Dimensionality of encoder inner feed-forward layers
            encoder_dropout (float, optional): Dropout probability for encoder
            encoder_activation_function (str, optional): Activation function for encoder
            
            # Decoder-specific parameters (take precedence over unified if specified)
            decoder_n_layer (int, optional): Number of hidden layers in the decoder
            decoder_n_head (int, optional): Number of attention heads for decoder
            decoder_n_inner (int, optional): Dimensionality of decoder inner feed-forward layers
            decoder_dropout (float, optional): Dropout probability for decoder
            decoder_activation_function (str, optional): Activation function for decoder
        """
        # Call nn.Module constructor directly instead of parent
        super(GPT2VQVAE, self).__init__()

        # Apply unified parameters as defaults, then override with specific parameters if provided
        final_encoder_n_layer = encoder_n_layer if encoder_n_layer is not None else n_layer
        final_encoder_n_head = encoder_n_head if encoder_n_head is not None else n_head
        final_encoder_n_inner = encoder_n_inner if encoder_n_inner is not None else n_inner
        final_encoder_dropout = encoder_dropout if encoder_dropout is not None else dropout
        final_encoder_activation_function = encoder_activation_function if encoder_activation_function is not None else activation_function
        
        final_decoder_n_layer = decoder_n_layer if decoder_n_layer is not None else n_layer
        final_decoder_n_head = decoder_n_head if decoder_n_head is not None else n_head
        final_decoder_n_inner = decoder_n_inner if decoder_n_inner is not None else n_inner
        final_decoder_dropout = decoder_dropout if decoder_dropout is not None else dropout
        final_decoder_activation_function = decoder_activation_function if decoder_activation_function is not None else activation_function
        
        # Handle None values for n_inner parameters
        if final_encoder_n_inner == "None": final_encoder_n_inner = None
        if final_decoder_n_inner == "None": final_decoder_n_inner = None

        # Create encoder config with encoder-specific parameters
        self.encoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_positions=n_positions,
            n_layer=final_encoder_n_layer,
            n_head=final_encoder_n_head,
            n_inner=final_encoder_n_inner,
            resid_pdrop=final_encoder_dropout,
            embd_pdrop=final_encoder_dropout,
            attn_pdrop=final_encoder_dropout,
            activation_function=final_encoder_activation_function,
        )
        
        # Create decoder config with decoder-specific parameters and cross-attention
        self.decoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_positions=n_positions,
            add_cross_attention=True,  # Enable cross-attention for decoder
            is_decoder=True,  # Mark as decoder
            n_layer=final_decoder_n_layer,
            n_head=final_decoder_n_head,
            n_inner=final_decoder_n_inner,
            resid_pdrop=final_decoder_dropout,
            embd_pdrop=final_decoder_dropout,
            attn_pdrop=final_decoder_dropout,
            activation_function=final_decoder_activation_function,
        )
        
        # Initialize encoder with or without pretrained weights
        if use_pretrained_encoder:
            print(f"\nLoading pretrained {pretrained_model_name} weights for encoder...")
            self.encoder = GPT2Model.from_pretrained(pretrained_model_name, config=self.encoder_config)
            # Ensure the encoder uses our config (in case vocab_size differs)
            if self.encoder.config.vocab_size != vocab_size:
                print(f"Warning: Pretrained model vocab_size ({self.encoder.config.vocab_size}) "
                      f"differs from specified vocab_size ({vocab_size}). "
                      f"Using specified vocab_size.")
                self.encoder.resize_token_embeddings(vocab_size)
        else:
            print("\nInitializing encoder with random weights...")
            self.encoder = GPT2Model(self.encoder_config)
        
        # Initialize decoder with or without pretrained weights
        if use_pretrained_decoder:
            print(f"Loading pretrained {pretrained_model_name} weights for decoder...")
            self.decoder = CustomGPT2LMHeadModel.from_pretrained(pretrained_model_name, config=self.decoder_config)
            # Ensure the decoder uses our config
            if self.decoder.config.vocab_size != vocab_size:
                print(f"Warning: Pretrained model vocab_size ({self.decoder.config.vocab_size}) "
                      f"differs from specified vocab_size ({vocab_size}). "
                      f"Using specified vocab_size.")
                self.decoder.resize_token_embeddings(vocab_size)
        else:
            print("Initializing decoder with random weights...")
            self.decoder = CustomGPT2LMHeadModel(self.decoder_config)
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
        
        # Chain-positional embeddings (single chain per prompt)
        self.chain_embeddings = nn.Embedding(1, d_model)  # Only 1 embedding since single CoT per prompt
        # Initialize with small values
        nn.init.normal_(self.chain_embeddings.weight, mean=0.0, std=0.02)
        
        self.d_model = d_model
        self.num_thoughts = 1  # Single chain per prompt
        
        # Store pretrained model information for checkpoint validation
        self._use_pretrained_encoder = use_pretrained_encoder
        self._use_pretrained_decoder = use_pretrained_decoder
        self._pretrained_model_name = pretrained_model_name
        
        # Store configuration for checkpoint validation
        self._encoder_config_params = {
            'n_layer': final_encoder_n_layer,
            'n_head': final_encoder_n_head,
            'n_inner': final_encoder_n_inner,
            'dropout': final_encoder_dropout,
            'activation_function': final_encoder_activation_function,
        }
        self._decoder_config_params = {
            'n_layer': final_decoder_n_layer,
            'n_head': final_decoder_n_head,
            'n_inner': final_decoder_n_inner,
            'dropout': final_decoder_dropout,
            'activation_function': final_decoder_activation_function,
        }
        
        # Initialize gradient checkpointing as disabled by default
        self._gradient_checkpointing_enabled = False
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing_enabled = True
        self.encoder.gradient_checkpointing_enable()
        self.decoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for SimpleGPT2VQVAE")
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()
        print("Gradient checkpointing disabled for SimpleGPT2VQVAE")
        
    def _create_combined_mask(self, prompt_mask, cot_mask, batch_size, K, L, device):
        """
        Create combined attention mask from prompt and COT masks.
        
        Args:
            prompt_mask (torch.Tensor, optional): Prompt attention mask [batch_size, K]
            cot_mask (torch.Tensor, optional): COT attention mask [batch_size, L]
            batch_size (int): Batch size
            K (int): Prompt length
            L (int): COT length
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Combined attention mask [batch_size, K + L]
        """
        if prompt_mask is not None and cot_mask is not None:
            combined_mask = torch.cat([prompt_mask, cot_mask], dim=1)
        elif prompt_mask is not None:
            cot_mask_ones = torch.ones(batch_size, L, device=device)
            combined_mask = torch.cat([prompt_mask, cot_mask_ones], dim=1)
        elif cot_mask is not None:
            prompt_mask_ones = torch.ones(batch_size, K, device=device)
            combined_mask = torch.cat([prompt_mask_ones, cot_mask], dim=1)
        else:
            combined_mask = None
            
        return combined_mask
        
    def encode(self, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None, 
               quantize_cot_only=True, use_vq=True):
        """
        Encodes prompt sequences and COT sequences, with or without caching based on gradient checkpointing status.
        
        Args:
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, L] (single CoT per prompt)
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            quantize_cot_only (bool): If True, only quantize COT positions (K to K+L-1). 
                                    If False, quantize all positions (0 to K+L-1).
            use_vq (bool): If True, apply vector quantization to encoder outputs.
                          If False, pass encoder outputs directly without quantization.
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, indices) 
                  - If use_vq=True: quantized tensor, vq_loss, perplexity, indices
                  - If use_vq=False: encoder outputs, zero loss, None perplexity, None indices
        """
        batch_size, K = prompt_sequences.shape
        _, L = cot_sequences.shape  # Now single CoT sequence per prompt
        
        # Check if gradient checkpointing is enabled to determine caching strategy
        use_caching = not self.is_gradient_checkpointing_enabled()
        
        if use_caching:
            # CACHING APPROACH: Encode prompt first, then COT with cached activations
            
            # Step 1: Encode prompt sequences with caching
            prompt_outputs = self.encoder(
                input_ids=prompt_sequences,
                attention_mask=prompt_mask,
                past_key_values=DynamicCache(),
                use_cache=True,
                return_dict=True
            )
            
            # Extract prompt activations and cache
            prompt_activations = prompt_outputs.last_hidden_state  # [batch_size, K, d_model]
            prompt_cache = prompt_outputs.past_key_values
            
            # Step 2: Process COT sequences with prompt cache
            # Create combined mask using helper method
            combined_mask = self._create_combined_mask(prompt_mask, cot_mask, batch_size, K, L, cot_sequences.device)
            
            # Continue encoding with COT sequences using cached prompt activations
            cot_outputs = self.encoder(
                input_ids=cot_sequences,  # Only COT sequences, not combined input
                attention_mask=combined_mask,
                past_key_values=prompt_cache,  # Use prompt cache
                use_cache=False,
                return_dict=True
            )
            
            # Get COT hidden states (only the newly computed activations)
            cot_memory = cot_outputs.last_hidden_state  # [batch_size, L, d_model]
            
            if quantize_cot_only:
                # Only use COT activations, no need to concatenate with prompt
                memory = cot_memory  # [batch_size, L, d_model]
            else:
                # Concatenate prompt activations with COT activations to get full sequence
                full_memory = torch.cat([prompt_activations, cot_memory], dim=1)  # [batch_size, K + L, d_model]
                memory = full_memory  # [batch_size, K + L, d_model]
        
        else:
            # NON-CACHING APPROACH: Process full sequence in one pass
            # Concatenate prompt and COT sequences
            full_sequences = torch.cat([prompt_sequences, cot_sequences], dim=1)  # [batch_size, K + L]
            
            # Create combined attention mask using helper method
            combined_mask = self._create_combined_mask(prompt_mask, cot_mask, batch_size, K, L, cot_sequences.device)
            
            # Encode full sequence in one pass
            full_outputs = self.encoder(
                input_ids=full_sequences,
                attention_mask=combined_mask,
                use_cache=False,  # No caching when gradient checkpointing is enabled
                return_dict=True
            )
            
            # Extract hidden states
            full_memory = full_outputs.last_hidden_state  # [batch_size, K + L, d_model]
            
            if quantize_cot_only:
                # Only use COT activations (positions K to K+L-1)
                memory = full_memory[:, K:, :]  # [batch_size, L, d_model]
            else:
                # Use all activations
                memory = full_memory  # [batch_size, K + L, d_model]

        if use_vq:
            # Apply VQ directly to the memory (no MLP aggregation needed)
            # Reshape for vector quantizer: [batch_size, seq_len, d_model] -> [batch_size * seq_len, d_model]
            memory_flat = memory.view(-1, memory.size(-1))  # [batch_size * seq_len, d_model]
            quantized_flat, vq_loss, perplexity, indices_flat = self.vector_quantizer(memory_flat)
            
            # Reshape back to original shape
            quantized = quantized_flat.view(memory.shape)  # [batch_size, seq_len, d_model]
            indices = indices_flat.view(batch_size, -1)  # [batch_size, seq_len]
        else:
            # Bypass VQ-VAE: pass encoder outputs directly
            quantized = memory  # [batch_size, seq_len, d_model]
            vq_loss = torch.tensor(0.0, device=memory.device)  # Zero loss when not using VQ
            perplexity = torch.tensor(0.0, device=memory.device)  # Zero perplexity when not using VQ
            indices = torch.tensor(0.0, device=memory.device)  # Zero tensor when not using VQ
        
        return quantized, vq_loss, perplexity, indices

    def decode(self, memory, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None, pad_token_id=0):
        """
        Decodes using GPT2 decoder with or without caching based on gradient checkpointing status.
        
        Args:
            memory (torch.Tensor): Encoded memory [batch_size, L, d_model] or [batch_size, K+L, d_model]
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, L] (single CoT per prompt)
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            pad_token_id (int): Token ID to use for padding when K=0, defaults to 0
            
        Returns:
            torch.Tensor: Decoded output logits [batch_size, L, vocab_size] (only COT positions)
        """
        batch_size, K = prompt_sequences.shape
        _, L = cot_sequences.shape  # Now single CoT sequence per prompt
        
        # Add chain-positional embeddings to memory (single chain per prompt)
        chain_indices = torch.zeros(batch_size, dtype=torch.long, device=memory.device)  # All zeros since single chain
        chain_emb = self.chain_embeddings(chain_indices).unsqueeze(1)  # [batch_size, 1, d_model]
        memory = memory + chain_emb  # Add to all positions in the sequence
        
        # Check if gradient checkpointing is enabled to determine caching strategy
        use_caching = not self.is_gradient_checkpointing_enabled()
        
        if use_caching:
            # CACHING APPROACH: Pre-compute KV cache for prompt tokens except the last one
            
            # Step 1: Pre-compute KV cache for prompt tokens except the last one
            if K > 1:
                # Process prompt tokens 0 to K-2 (excluding the last one)
                prompt_except_last = prompt_sequences[:, :K-1]  # [batch_size, K-1]
                prompt_mask_except_last = prompt_mask[:, :K-1] if prompt_mask is not None else None # [batch_size, K-1]
                
                # Get GPT2 decoder outputs for prompt except last token with caching
                prompt_outputs = self.decoder(
                    input_ids=prompt_except_last,
                    attention_mask=prompt_mask_except_last,
                    past_key_values=EncoderDecoderCache(DynamicCache(), DynamicCache()), # for cross-attention
                    use_cache=True,
                    return_dict=True
                )
                
                # Extract prompt cache
                prompt_cache = prompt_outputs.past_key_values
            else:
                # If K=1, start with empty cache
                prompt_cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
            
            # Step 2: Prepare the last prompt token and concatenate with COT sequences
            # Prepare last prompt token (either from prompt or padding)
            if K > 0:
                last_prompt_token = prompt_sequences[:, K-1:K]  # [batch_size, 1]
            else:
                last_prompt_token = torch.full((batch_size, 1), pad_token_id, 
                                             dtype=torch.long, device=cot_sequences.device)  # [batch_size, 1]

            # Concatenate sequences
            combined_sequences = torch.cat([last_prompt_token, cot_sequences], dim=1)  # [batch_size, L+1]
            
            # Step 3: Create combined mask using _create_combined_mask for full extent
            full_combined_mask = self._create_combined_mask(prompt_mask, cot_mask, batch_size, K, L, cot_sequences.device)
            
            # Step 4: Create cross-attention mask for memory attention
            # For each position i in the combined sequence, can attend to memory tokens 0 to i
            cross_attention_mask = create_cross_attention_mask(L+1, L, memory.device)
            
            # Expand to match batch and sequence dimensions for GPT2 
            cross_attention_mask = cross_attention_mask.unsqueeze(0).unsqueeze(0).expand(
                batch_size, -1, -1, -1) # [batch_size, 1, L+1, L]
            
            # Step 5: Compute all logits in one forward pass
            decoder_outputs = self.decoder(
                input_ids=combined_sequences,  # [batch_size, L+1]
                attention_mask=full_combined_mask,
                encoder_hidden_states=memory,  # [batch_size, L, d_model]
                extra_cross_attention_mask=cross_attention_mask,  # [batch_size, 1, L+1, L]
                past_key_values=prompt_cache,  # Use EncoderDecoderCache
                use_cache=False,
                return_dict=True
            )
            
            # Get all logits
            all_logits = decoder_outputs.logits  # [batch_size, L+1, vocab_size]
            
            # Extract only the COT logits (discard last position that predicts beyond COT length)
            cot_logits = all_logits[:, :-1, :]  # [batch_size, L, vocab_size]
            
        else:
            # NON-CACHING APPROACH: Process full sequence in one pass
            # Concatenate prompt and COT sequences
            combined_sequences = torch.cat([prompt_sequences, cot_sequences], dim=1)  # [batch_size, K + L]
            
            # Create combined attention mask using helper method
            combined_mask = self._create_combined_mask(prompt_mask, cot_mask, batch_size, K, L, cot_sequences.device)
            
            # Create cross-attention mask for memory attention
            cross_attention_mask = create_cross_attention_mask(K + L, L, memory.device)
            cross_attention_mask = cross_attention_mask.unsqueeze(0).unsqueeze(0).expand(
                batch_size, -1, -1, -1) # [batch_size, 1, K+L, L]
            
            # Decode full sequence in one pass
            decoder_outputs = self.decoder(
                input_ids=combined_sequences,  # [batch_size, K + L]
                attention_mask=combined_mask,
                encoder_hidden_states=memory,  # [batch_size, L, d_model]
                extra_cross_attention_mask=cross_attention_mask,  # [batch_size, 1, K+L, L]
                use_cache=False,  # No caching when gradient checkpointing is enabled
                return_dict=True
            )
            
            # Get all logits
            all_logits = decoder_outputs.logits  # [batch_size, K + L, vocab_size]
            
            # Extract only the COT logits (positions K-1 to K+L-2)
            cot_logits = all_logits[:, K-1:-1, :]  # [batch_size, L, vocab_size]
        
        return cot_logits

    def forward(self, prompt, cot_sequences, cot_mask=None, prompt_mask=None, inference=False, quantize_cot_only=True, pad_token_id=50256, use_vq=True):
        """
        Forward pass through the model.
        
        Args:
            prompt (torch.Tensor): Prompt sequences [batch_size, K] where K is prompt length
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, L] or [batch_size, 1, L]
            cot_mask (torch.Tensor, optional): Chain-of-thought attention mask for padding
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            inference (bool): If True, performs inference without teacher forcing
            quantize_cot_only (bool): If True, only quantize the COT portion of sequences
            pad_token_id (int): Token ID to use for padding when K=0, defaults to 50256
            use_vq (bool): If True, apply vector quantization to encoder outputs.
                          If False, pass encoder outputs directly without quantization.
            
        Returns:
            tuple: (output_sequences, output_logits, vq_loss, perplexity, indices)
                - output_sequences: Generated token sequences [batch_size, L]
                - output_logits: Token logits for each position [batch_size, L, vocab_size]
                - vq_loss: Vector quantization loss (zero if use_vq=False)
                - perplexity: Codebook usage perplexity (None if use_vq=False)
                - indices: Codebook usage indices [batch_size, L] or [batch_size, K+L] depending on quantize_cot_only (None if use_vq=False)
        """
        if cot_sequences.ndim == 3 and cot_sequences.size(1) != 1:
            raise Exception(f"cot_sequences must only hold one CoT per prompt input but has shape: {cot_sequences.size()}")
        elif cot_sequences.ndim == 3:
            cot_sequences = torch.squeeze(cot_sequences, 1)    
            if cot_mask is not None:
                cot_mask = torch.squeeze(cot_mask, 1)
        
        batch_size, K = prompt.shape
        _, L = cot_sequences.shape  # Now single CoT sequence per prompt
        
        # Encode using the new approach (with optional VQ)
        quantized, vq_loss, perplexity, indices = self.encode(
            prompt, cot_sequences, 
            prompt_mask, cot_mask, 
            quantize_cot_only=quantize_cot_only,
            use_vq=use_vq
        )
        
        # quantized shape depends on quantize_cot_only:
        if quantize_cot_only:          # if True: [batch_size, L, d_model] (only COT positions)
            cot_quantized = quantized
        else:                          # if False: [batch_size, K+L, d_model] (all positions)
            cot_quantized = quantized[:, K:, :]
        
        if not inference:
            # During training, use teacher forcing with single forward pass to get all logits
            output_logits = self.decode(cot_quantized, prompt, cot_sequences, prompt_mask, cot_mask, pad_token_id) # [batch_size, L, vocab_size]
            
            # Get the predicted tokens from logits
            output_sequences = torch.argmax(output_logits, dim=-1)
        else:
            # Initialize tensor to store generation results and logits
            output_sequences = torch.zeros((batch_size, L), dtype=torch.long, device=cot_sequences.device)
            output_logits = torch.empty((batch_size, L, self.decoder_config.vocab_size), device=cot_sequences.device)
        
            # TODO IF I FIND THE TIME : USE KV-CACHING TO SPEED UP AUTO-REGRESSIVE GENERATION
            # During inference, generate sequence auto-regressively
            for t in range(L):
                current_output = self.decode(cot_quantized, prompt, output_sequences, prompt_mask, cot_mask, pad_token_id)
                
                # Get next token predictions
                output_logits[:, t, :] = current_output[:, t, :]  # [batch_size, L, vocab_size]
                output_sequences[:, t] = torch.argmax(output_logits[:, t, :], dim=-1)  # [batch_size, L]

        # Finally expand the output to match that of GPT2VQVAE
        output_sequences, output_logits = output_sequences.unsqueeze(1), output_logits.unsqueeze(1)
        
        return output_sequences, output_logits, vq_loss, perplexity, indices

    def load_checkpoint(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load the checkpoint on (if None, uses model's device)
        """
        if device is None:
            device = str(next(self.parameters()).device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate model and training configurations
        def _check_config_mismatch(checkpoint_config, current_config, config_type):
            """Helper function to check and report configuration mismatches."""
            if checkpoint_config != current_config:
                print(f"Warning: {config_type} configuration mismatch detected!")
                print(f"Differences between checkpoint and current {config_type} config:")
                for key in set(checkpoint_config.keys()) | set(current_config.keys()):
                    if key not in checkpoint_config:
                        print(f"  {key}: missing in checkpoint, current: {current_config[key]}")
                    elif key not in current_config:
                        print(f"  {key}: missing in current, checkpoint: {checkpoint_config[key]}")
                    elif checkpoint_config[key] != current_config[key]:
                        print(f"  {key}: checkpoint={checkpoint_config[key]}, current={current_config[key]}")
                print(f"Continuing with current {config_type} configuration...")
        
        # Get current model configuration
        current_model_config = {
            'vocab_size': self.encoder_config.vocab_size,
            'd_model': self.d_model,
            'num_embeddings': self.vector_quantizer.num_embeddings,
            'commitment_cost': self.vector_quantizer.commitment_cost,
            'n_positions': self.encoder_config.n_positions,
            'use_pretrained_encoder': hasattr(self, '_use_pretrained_encoder'),
            'use_pretrained_decoder': hasattr(self, '_use_pretrained_decoder'),
            'pretrained_model_name': getattr(self, '_pretrained_model_name', 'gpt2'),
            # Encoder-specific configuration
            'encoder_config': self._encoder_config_params,
            # Decoder-specific configuration
            'decoder_config': self._decoder_config_params,
        }
        
        if 'model_config' in checkpoint:
            _check_config_mismatch(checkpoint['model_config'], current_model_config, "model")
        
        if 'training_config' in checkpoint:
            print("Note: Training configuration found in checkpoint but not validated for model-only loading.")
        
        # Load model state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print("Checkpoint loaded successfully. Configuration validation completed.")