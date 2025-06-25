# Author: Akira Kudo
# Created: 2025/06/12
# Last Updated: 2025/06/23

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, Cache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

def compute_perplexity(encodings: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute perplexity from encoding probabilities or counts.
    
    Perplexity measures the diversity of latent code usage in vector quantization.
    - High perplexity: uniform usage, no learning (all codes used equally)
    - Low perplexity: not all codes are being used effectively
    - Mid perplexity: good balance of code usage
    
    Args:
        encodings (torch.Tensor): Either:
                                 - One-hot encoding tensor of shape [N, num_embeddings] (probabilities)
                                 - Count tensor of shape [num_embeddings] (counts)
                                 where N is the number of encoded vectors
        eps (float): Small epsilon value to prevent log(0), defaults to 1e-10
        
    Returns:
        torch.Tensor: Perplexity value (scalar tensor)
        
    Example:
        >>> # Using probabilities (one-hot encodings)
        >>> encodings = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3 vectors, 3 codes
        >>> perplexity = compute_perplexity(encodings)
        >>> print(perplexity)  # Should be close to 3.0 (uniform usage)
        
        >>> # Using counts
        >>> counts = torch.tensor([10, 8, 12])  # Count of each code usage
        >>> perplexity = compute_perplexity(counts)
        >>> print(perplexity)  # Perplexity based on count distribution
    """
    # Infer input type from tensor shape
    if encodings.dim() == 2:
        # 2D tensor: [N, num_embeddings] - treat as probabilities (one-hot encodings)
        avg_probs = torch.mean(encodings, dim=0)
    elif encodings.dim() == 1:
        # 1D tensor: [num_embeddings] - treat as counts
        total_count = torch.sum(encodings)
        avg_probs = encodings / total_count
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got {encodings.dim()}D tensor")
    
    # Compute perplexity: exp(-sum(p * log(p)))
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))
    
    return perplexity

class CustomGPT2Model(GPT2Model):
    """
    Custom GPT2Model that adds an extra 4D mask on top of the encoder_attention_mask.
    This mask has the same shape and purpose as one generated using create_cross_attention_mask.
    """
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Tuple[Tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_cross_attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        extra_cross_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_heads, query_length, key_length)`, *optional*):
            An additional 4D attention mask that will be added to the `encoder_attention_mask` for cross-attention.
            This mask should have the same shape and purpose as one generated using `create_cross_attention_mask`.
            The mask uses the same convention as `encoder_attention_mask`: 0 for attended positions, 
            negative infinity for masked positions. This allows for custom cross-attention patterns
            beyond the standard encoder-decoder attention which is crucial for causal attendance to latent tokens.

        *CODE IS MODIFIED FROM THE BASE GPT2Model CODE FOR A SPECIFIC REGION - INDICATED IN COMMENTS
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        return_legacy_cache = False
        if use_cache:
            if past_key_values is None:
                return_legacy_cache = True
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `Cache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            
            # MODIFICATION COMPARED TO THE BASE CODE - START
            # * encoder_attention_mask after _prepare_4d_attention_mask_for_sdpa is None
            #   if an all-one mask was passed to the function.
            
            # Add the extra cross attention mask on top of the encoder_attention_mask
            if extra_cross_attention_mask is not None:
                if encoder_attention_mask is None:
                    encoder_attention_mask = extra_cross_attention_mask
                else:
                    # Ensure the extra mask has the right shape and device
                    if extra_cross_attention_mask.device != encoder_attention_mask.device:
                        extra_cross_attention_mask = extra_cross_attention_mask.to(encoder_attention_mask.device)
                    if extra_cross_attention_mask.dtype != encoder_attention_mask.dtype:
                        extra_cross_attention_mask = extra_cross_attention_mask.to(encoder_attention_mask.dtype)
                
                    # Combine the masks: add the extra mask to the encoder attention mask
                    # Both masks use the same convention: 0 for attended positions, negative infinity for masked positions
                    encoder_attention_mask = encoder_attention_mask + extra_cross_attention_mask

            # MODIFICATION COMPARED TO THE BASE CODE - END
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    past_key_values,
                    cache_position,
                    causal_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    past_key_value=past_key_values,
                    cache_position=cache_position,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if return_legacy_cache:
            past_key_values = (
                past_key_values.self_attention_cache.to_legacy_cache()
                if self.config.add_cross_attention
                else past_key_values.to_legacy_cache()
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    
    def __init__(self, config):
        super().__init__(config)
        # Replace the transformer with CustomGPT2Model to support extra_cross_attention_mask
        self.transformer = CustomGPT2Model(config)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_cross_attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        extra_cross_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_heads, query_length, key_length)`, *optional*):
            An additional 4D attention mask that will be added to the `encoder_attention_mask` for cross-attention.
            This mask should have the same shape and purpose as one generated using `create_cross_attention_mask`.
            The mask uses the same convention as `encoder_attention_mask`: 0 for attended positions, 
            negative infinity for masked positions. This allows for custom cross-attention patterns
            beyond the standard encoder-decoder attention which is crucial for causal attendance to latent tokens.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            extra_cross_attention_mask=extra_cross_attention_mask,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int, 
                 commitment_cost : float=0.25):
        """
        Vector quantizer initialization.

        :param int num_embeddings: Codebook size.
        :param int embedding_dim: Dimension of each embedding.
        :param float commitment_cost: Commitment loss multiplier, defaults to 0.25
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # TODO REVISIT
        # check if: 1) using weight.data instead of init makes sense
        # 2) if the code for initialization makes sense (especially dividing by num_embeddings)
        # for example, LATENT_PLANNING does this: module.weight.data.normal_(mean=0.0, std=0.02)
        # haven't found any mathematically elaborate explanation of why this initialization makes sense
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # Convert inputs [(batch_size, sequence_length) OR (batch_size x sequence_length), embedding_dim]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim) # [ (batch x seq_len), emb_dim ]

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)           # [ (batch x seq_len), emb_num ]    
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.T))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1) # [ (batch x seq_len) ]
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,  # [ (batch x seq_len), emb_num ]
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        # Perplexity: diversity of latent code usage, keep it mid (high=uniform, no learning, low=not used fully)
        perplexity = compute_perplexity(encodings)
        
        return quantized, loss, perplexity, encoding_indices.view(input_shape[:-1])

class GPT2VQVAE(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_embeddings=512, 
                 commitment_cost=0.25, aggregation_hidden_dim=1024, 
                 num_thoughts=32, n_positions=1024, 
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
        GPT2-based VQ-VAE model that uses GPT2 as both encoder and decoder.
        
        Args:
            # Shared parameters (must be the same for both encoder and decoder)
            vocab_size (int): Vocabulary size
            d_model (int): Model dimension (default: 768 for GPT2)
            n_positions (int): Maximum sequence length for GPT2 (default: 1024)
            
            # VQ-VAE specific parameters
            num_embeddings (int): VQ codebook size
            commitment_cost (float): VQ commitment cost
            aggregation_hidden_dim (int): Aggregation MLP hidden dimension
            num_thoughts (int): Number of parallel sequences
            
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
        super(GPT2VQVAE, self).__init__()

        # TODO ADD INITIALIZATION FOR ENCODER, DECODER AND MLP
        # THOUGHT: COULD ADD output_attentions=True FOR DEBUGGING (E.G. FOR HAND-MADE CROSS-ATTENTION MASK OF DECODER)
        
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
        
        # Aggregation MLP
        # TODO: Consider different options - might cause posterior collapse if too strong
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(num_thoughts * d_model, aggregation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # GPT2's default dropout
            nn.Linear(aggregation_hidden_dim, d_model)
        )
        
        # Chain-positional embeddings to differentiate M sequences
        self.chain_embeddings = nn.Embedding(num_thoughts, d_model)
        # Initialize with small values
        nn.init.normal_(self.chain_embeddings.weight, mean=0.0, std=0.02)
        
        self.d_model = d_model
        self.num_thoughts = num_thoughts
        
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
        print("Gradient checkpointing enabled for GPT2VQVAE")
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing_enabled = False
        self.encoder.gradient_checkpointing_disable()
        self.decoder.gradient_checkpointing_disable()
        print("Gradient checkpointing disabled for GPT2VQVAE")
        
    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self._gradient_checkpointing_enabled
    
    def get_gradient_checkpointing_status(self):
        """Get detailed gradient checkpointing status for all components."""
        return {
            'model_enabled': self._gradient_checkpointing_enabled,
            'encoder_enabled': self.encoder.is_gradient_checkpointing,
            'decoder_enabled': self.decoder.is_gradient_checkpointing,
            'encoder_training': self.encoder.training,
            'decoder_training': self.decoder.training
        }
        
    def aggregate(self, memory, mode="linear"):
        """
        Aggregates same-position embeddings from chains sharing the same prompt.
        
        Args:
            memory (torch.Tensor): Shape [batch_size*L, num_thoughts, d_model]
            mode (str): Aggregation mode, defaults to "linear"
            
        Returns:
            torch.Tensor: Aggregated embeddings [batch_size*L or batch_size*(K+L), d_model]
        """
        _, M, d_model = memory.shape
        if mode == "linear":
            # Reshape to pass through MLP
            memory = memory.reshape(-1, M*d_model)         # [batch_size*L, M*d_model]
            # Pass through MLP for non-linear transformation
            aggregated = self.aggregation_mlp(memory)   # [batch_size*L, d_model]
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
            
        return aggregated
        
    def _create_combined_mask(self, prompt_mask, cot_mask_flat, batch_size, M, K, L, device):
        """
        Create combined attention mask from prompt and COT masks.
        
        Args:
            prompt_mask (torch.Tensor, optional): Prompt attention mask [batch_size, K]
            cot_mask_flat (torch.Tensor, optional): Flattened COT attention mask [batch_size * M, L]
            batch_size (int): Batch size
            M (int): Number of COT sequences
            K (int): Prompt length
            L (int): COT length
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Combined attention mask [batch_size * M, K + L]
        """
        if prompt_mask is not None and cot_mask_flat is not None:
            prompt_mask_expanded = prompt_mask.unsqueeze(1).expand(-1, M, -1).reshape(batch_size * M, K)
            combined_mask = torch.cat([prompt_mask_expanded, cot_mask_flat], dim=1)
        elif prompt_mask is not None:
            prompt_mask_expanded = prompt_mask.unsqueeze(1).expand(-1, M, -1).reshape(batch_size * M, K)
            cot_mask_ones = torch.ones(batch_size * M, L, device=device)
            combined_mask = torch.cat([prompt_mask_expanded, cot_mask_ones], dim=1)
        elif cot_mask_flat is not None:
            prompt_mask_ones = torch.ones(batch_size * M, K, device=device)
            combined_mask = torch.cat([prompt_mask_ones, cot_mask_flat], dim=1)
        else:
            combined_mask = None
            
        return combined_mask
    
    def _pad_kv_cache(self, prompt_cache, batch_size, M):
        """
        Pad KV cache M times to match COT batch size.
        
        Args:
            prompt_cache: KV cache from prompt processing (Cache object, EncoderDecoderCache, or legacy tuple format)
            batch_size (int): Batch size
            M (int): Number of COT sequences
            K (int): Prompt length
            
        Returns:
            Padded KV cache in the same format as input (Cache object, EncoderDecoderCache, or legacy tuple)
        """
        # Check if it's an EncoderDecoderCache
        if hasattr(prompt_cache, 'self_attention_cache') and hasattr(prompt_cache, 'cross_attention_cache'):
            # It's an EncoderDecoderCache
            padded_self_cache = self._pad_kv_cache(prompt_cache.self_attention_cache, batch_size, M)
            padded_cross_cache = self._pad_kv_cache(prompt_cache.cross_attention_cache, batch_size, M)
            return EncoderDecoderCache(padded_self_cache, padded_cross_cache)
        
        # Check if it's a Cache object (like DynamicCache)
        elif hasattr(prompt_cache, 'key_cache') and hasattr(prompt_cache, 'value_cache'):
            # It's a Cache object (e.g., DynamicCache)
            padded_cache = type(prompt_cache)()  # Create new instance of same type
            
            # Pad each layer's key and value tensors
            for layer_idx in range(len(prompt_cache.key_cache)):
                key_tensor = prompt_cache.key_cache[layer_idx]  # [batch_size, num_heads, K, head_dim]
                value_tensor = prompt_cache.value_cache[layer_idx]  # [batch_size, num_heads, K, head_dim]
                
                # Expand from [batch_size, num_heads, K, head_dim] to [batch_size * M, num_heads, K, head_dim]
                padded_key = key_tensor.unsqueeze(1).expand(-1, M, -1, -1, -1)
                padded_key = padded_key.reshape([batch_size * M] + list(key_tensor.size())[1:])
                
                padded_value = value_tensor.unsqueeze(1).expand(-1, M, -1, -1, -1)
                padded_value = padded_value.reshape([batch_size * M] + list(key_tensor.size())[1:])
                
                # Update the cache with padded tensors
                padded_cache.update(padded_key, padded_value, layer_idx)

        # in case no cache is being used (e.g. when gradient checkpointing is disabled)
        elif prompt_cache is None:
            padded_cache = None

        else:
            # It's a legacy tuple format: Tuple[Tuple[torch.Tensor, torch.Tensor]]
            padded_cache = []
            for layer_cache in prompt_cache:
                # Each layer cache is a tuple of (key, value) for each layer
                # key/value shape: [batch_size, num_heads, seq_len, head_dim]
                padded_layer_cache = []
                for kv in layer_cache:
                    # Expand from [batch_size, num_heads, K, head_dim] to [batch_size * M, num_heads, K, head_dim]
                    padded_kv = kv.unsqueeze(1).expand(-1, M, -1, -1, -1)
                    padded_kv = padded_kv.reshape([batch_size * M] + list(key_tensor.size())[1:])
                    padded_layer_cache.append(padded_kv)
                padded_cache.append(tuple(padded_layer_cache))
            
            # Convert back to tuple format
            padded_cache = tuple(padded_cache)
            
        return padded_cache
        
    def encode(self, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None, 
               aggregate_mode="linear", quantize_cot_only=True):
        """
        Encodes prompt sequences first with caching, then processes COT sequences with padded prompt activations.
        
        Args:
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            aggregate_mode (str): Mode of aggregation
            quantize_cot_only (bool): If True, only quantize COT positions (K to K+L-1). 
                                    If False, quantize all positions (0 to K+L-1).
            
        Returns:
            tuple: (quantized, vq_loss, perplexity, indices) 
        """
        batch_size, K = prompt_sequences.shape
        _, M, L = cot_sequences.shape
        
        # Step 1: Encode prompt sequences with caching
        
        # Get GPT2 encoder outputs for prompt with caching
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
        
        # Step 2: Process COT sequences with padded prompt cache
        # Reshape COT sequences to [batch_size * M, L]
        cot_flat = cot_sequences.view(batch_size * M, L)  # [batch_size * M, L]
        if cot_mask is not None:
            cot_mask_flat = cot_mask.view(batch_size * M, L)  # [batch_size * M, L]
        else:
            cot_mask_flat = None
        
        # Pad prompt cache M times using helper method
        padded_cache = self._pad_kv_cache(prompt_cache, batch_size, M)
        
        # Step 3: Continue encoding with COT sequences using cached prompt activations
        # Create combined mask using helper method
        combined_mask = self._create_combined_mask(prompt_mask, cot_mask_flat, batch_size, M, K, L, cot_flat.device)
        
        # Continue encoding with cached activations - only pass COT sequences as input_ids
        cot_outputs = self.encoder(
            input_ids=cot_flat,  # Only COT sequences, not combined input
            attention_mask=combined_mask,
            past_key_values=padded_cache,  # Use padded prompt cache
            use_cache=False,
            return_dict=True
        )
        
        # Get COT hidden states (only the newly computed activations)
        cot_memory = cot_outputs.last_hidden_state  # [batch_size * M, L, d_model]
        
        if quantize_cot_only:
            # Only use COT activations, no need to concatenate with prompt
            # Reshape to group tokens at same positions across sequences
            cot_memory = cot_memory.view(batch_size, M, L, -1)
            memory = cot_memory.transpose(1, 2)  # [batch_size, L, M, d_model]
        else:
            # Pad prompt activations M times to match COT batch size
            # [batch_size, K, d_model] -> [batch_size * M, K, d_model]
            padded_prompt_activations = prompt_activations.unsqueeze(1).expand(-1, M, -1, -1)
            padded_prompt_activations = padded_prompt_activations.reshape(batch_size * M, K, -1)
            
            # Concatenate prompt activations with COT activations to get full sequence
            # padded_prompt_activations: [batch_size * M, K, d_model]
            # cot_memory: [batch_size * M, L, d_model]
            full_memory = torch.cat([padded_prompt_activations, cot_memory], dim=1)  # [batch_size * M, K + L, d_model]
            
            # Reshape to group tokens at same positions across sequences
            full_memory = full_memory.view(batch_size, M, K + L, -1)
            memory = full_memory.transpose(1, 2)  # [batch_size, K + L, M, d_model]

        # Reshape for aggregation
        memory = memory.reshape(-1, M, memory.size(-1))  # [batch_size*L or batch_size*(K+L), M, d_model]

        # Aggregate the memory content per-prompt into single chains 
        aggregated = self.aggregate(memory, mode=aggregate_mode) # [batch_size*L or batch_size*(K+L), d_model]
        
        # Apply VQ
        quantized, vq_loss, perplexity, indices = self.vector_quantizer(aggregated) # [batch_size*L or batch_size*(K+L), d_model]
        
        # Tile back to obtain the same shape and amount of info
        # Expand vs repeat: more memory efficient + no in-place change
        quantized = quantized.unsqueeze(1).expand(-1, M, -1)  # [batch_size*(L or K+L), M, d_model]
        
        # Reshape back using the appropriate length
        quantized = quantized.view(batch_size, -1, M, quantized.size(-1))
        indices = indices.view(batch_size, -1)
        
        return quantized, vq_loss, perplexity, indices
        
    def decode(self, memory, prompt_sequences, cot_sequences, prompt_mask=None, cot_mask=None, pad_token_id=0):
        """
        Decodes using GPT2 decoder:
        1. Pre-compute KV cache for prompt tokens except the last one
        2. Pad the last prompt token M times and concatenate with COT sequences
        3. Compute all logits in one forward pass
        
        Args:
            memory (torch.Tensor): Encoded memory [batch_size, L, M, d_model]
            prompt_sequences (torch.Tensor): Prompt sequences [batch_size, K]
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            cot_mask (torch.Tensor, optional): COT attention mask for padding
            pad_token_id (int): Token ID to use for padding when K=0, defaults to 0
            
        Returns:
            torch.Tensor: Decoded output logits [batch_size, M, L, vocab_size] (only COT positions)
        """
        batch_size, K = prompt_sequences.shape
        _, M, L = cot_sequences.shape
        
        # Reshape memory for decoder
        memory = memory.transpose(1, 2)  # [batch_size, M, L, d_model]
        memory = memory.reshape(batch_size * M, L, -1)  # [batch_size * M, L, d_model]
        
        # Add chain-positional embeddings to memory
        chain_indices = torch.arange(M, device=memory.device).repeat(batch_size)
        chain_emb = self.chain_embeddings(chain_indices).unsqueeze(1)  # [batch_size*M, 1, d_model]
        memory = memory + chain_emb  # Add to all positions in the sequence
        
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
        
        # Pad prompt cache M times using helper method
        padded_cache = self._pad_kv_cache(prompt_cache, batch_size, M)
        
        # Step 2: Pad the last prompt token M times and concatenate with COT sequences
        # Prepare last prompt token (either from prompt or padding)
        if K > 0:
            last_prompt_token = prompt_sequences[:, K-1:K].unsqueeze(1).expand(-1, M, -1)  # [batch_size, M, 1]
        else:
            last_prompt_token = torch.full((batch_size, M, 1), pad_token_id, 
                                         dtype=torch.long, device=cot_sequences.device)  # [batch_size, M, 1]

        # Concatenate sequences
        combined_sequences = torch.cat([last_prompt_token, cot_sequences], dim=2)  # [batch_size, M, L+1]
        combined_sequences_flat = combined_sequences.view(batch_size * M, L + 1)  # [batch_size * M, L+1]
        
        # Step 3: Create combined mask using _create_combined_mask for full extent
        if cot_mask is not None:
            cot_mask_flat = cot_mask.view(batch_size * M, L)  # [batch_size * M, L]
        else:
            cot_mask_flat = None
        
        # Use _create_combined_mask to get the full combined mask (K+L length)
        full_combined_mask = self._create_combined_mask(prompt_mask, cot_mask_flat, batch_size, M, K, L, cot_sequences.device)
        
        # Step 4: Create cross-attention mask for memory attention
        # For each position i in the combined sequence, can attend to memory tokens 0 to i
        cross_attention_mask = create_cross_attention_mask(L+1, L, memory.device)
        
        # Expand to match batch and sequence dimensions for GPT2 
        cross_attention_mask = cross_attention_mask.unsqueeze(0).unsqueeze(0).expand(
            batch_size * M, -1, -1, -1) # [batch_size*M, 1, L+1, L]
        
        # Step 5: Compute all logits in one forward pass
        decoder_outputs = self.decoder(
            input_ids=combined_sequences_flat,  # [batch_size * M, L+1]
            attention_mask=full_combined_mask,
            encoder_hidden_states=memory,  # [batch_size * M, L, d_model]
            extra_cross_attention_mask=cross_attention_mask,  # [batch_size * M, 1, L+1, L]
            past_key_values=padded_cache,  # Use padded EncoderDecoderCache
            use_cache=False,
            return_dict=True
        )
        
        # Get all logits
        all_logits = decoder_outputs.logits  # [batch_size * M, L+1, vocab_size]
        
        # Extract only the COT logits (discard last position that predicts beyond COT length)
        cot_logits = all_logits[:, :-1, :]  # [batch_size * M, L, vocab_size]
        
        # Reshape back to [batch_size, M, L, vocab_size]
        return cot_logits.view(batch_size, M, L, -1)
        
    def forward(self, prompt, cot_sequences, cot_mask=None, prompt_mask=None, inference=False, quantize_cot_only=True, pad_token_id=50256):
        """
        Forward pass through the model.
        
        Args:
            prompt (torch.Tensor): Prompt sequences [batch_size, K] where K is prompt length
            cot_sequences (torch.Tensor): Chain-of-thought sequences [batch_size, M, L]
            cot_mask (torch.Tensor, optional): Chain-of-thought attention mask for padding
            prompt_mask (torch.Tensor, optional): Prompt attention mask for padding
            inference (bool): If True, performs inference without teacher forcing
            quantize_cot_only (bool): If True, only quantize the COT portion of sequences
            pad_token_id (int): Token ID to use for padding when K=0, defaults to 50256
            
        Returns:
            tuple: (output_sequences, output_logits, vq_loss, perplexity, indices)
                - output_sequences: Generated token sequences [batch_size, M, L]
                - output_logits: Token logits for each position [batch_size, M, L, vocab_size]
                - vq_loss: Vector quantization loss
                - perplexity: Codebook usage perplexity
                - indices: Codebook usage indices [batch_size, L] or [batch_size, K+L] depending on quantize_cot_only
        """
        batch_size, K = prompt.shape
        _, M, L = cot_sequences.shape
        
        # Encode using the new separate prompt and COT approach
        quantized, vq_loss, perplexity, indices = self.encode(
            prompt, cot_sequences, 
            prompt_mask, cot_mask, 
            quantize_cot_only=quantize_cot_only
        )
        
        # quantized shape depends on quantize_cot_only:
        if quantize_cot_only:          # if True: [batch_size, L, M, d_model] (only COT positions)
            cot_quantized = quantized
        else:                          # if False: [batch_size, K+L, M, d_model] (all positions)
            cot_quantized = quantized[:, K:, :, :]
        
        if not inference:
            # During training, use teacher forcing with single forward pass to get all logits
            output_logits = self.decode(cot_quantized, prompt, cot_sequences, prompt_mask, cot_mask, pad_token_id) # [batch_size, M, L, vocab_size]
            
            # Get the predicted tokens from logits
            output_sequences = torch.argmax(output_logits, dim=-1)
        else:
            # Initialize tensor to store generation results and logits
            output_sequences = torch.zeros((batch_size, M, L), dtype=torch.long, device=cot_sequences.device)
            output_logits = torch.empty((batch_size, M, L, self.decoder_config.vocab_size), device=cot_sequences.device)
        
            # TODO IF I FIND THE TIME : USE KV-CACHING TO SPEED UP AUTO-REGRESSIVE GENERATION
            # During inference, generate sequence auto-regressively
            for t in range(L):
                current_output = self.decode(cot_quantized, prompt, output_sequences, prompt_mask, cot_mask, pad_token_id)
                
                # Get next token predictions
                output_logits[:, :, t, :] = current_output[:, :, t, :]  # [batch_size, M, L, vocab_size]
                output_sequences[:, :, t] = torch.argmax(output_logits[:, :, t, :], dim=-1)  # [batch_size, M, L]
        
        return output_sequences, output_logits, vq_loss, perplexity, indices

    def load_checkpoint(self, checkpoint_path: str, device: str = None):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load the checkpoint on (if None, uses model's device)
        """
        if device is None:
            device = next(self.parameters()).device
        
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
            'aggregation_hidden_dim': self.aggregation_mlp[0].out_features,
            'num_thoughts': self.num_thoughts,
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

def create_cross_attention_mask(query_length, key_length, device, dtype=torch.float32):
    """
    Create a cross-attention mask for memory attention.
    
    Args:
        query_length (int): Length of the query sequence (K+L)
        key_length (int): Length of the key sequence (L)
        device (torch.device): Device to create the mask on
        dtype (torch.dtype): Data type for the mask
        
    Returns:
        torch.Tensor: Cross-attention mask of shape (query_length, key_length)
                     The ith row can attend to key positions 0 to (i-(query_length-key_length)) + 1
                     Uses 0 for attended positions and torch.finfo(dtype).min for masked positions
    """
    # Calculate the offset: query_length - key_length = K
    offset = query_length - key_length
    
    # Use the same pattern as _prepare_4d_causal_attention_mask_with_cache_position
    min_dtype = torch.finfo(dtype).min
    mask = torch.full(
        (query_length, key_length), 
        fill_value=min_dtype, 
        dtype=dtype, 
        device=device
    )
    
    # Vectorized implementation: set positions that can be attended to as 0
    i = torch.arange(query_length, device=device).unsqueeze(1)  # [query_length, 1]
    j = torch.arange(key_length, device=device).unsqueeze(0)    # [1, key_length]
    attend_mask = (j <= i - offset + 1)  # [query_length, key_length]
    
    # Set attended positions to 0 (can attend) and keep masked positions as min_dtype (cannot attend)
    mask = torch.where(attend_mask, torch.tensor(0.0, dtype=dtype, device=device), mask)
    
    return mask