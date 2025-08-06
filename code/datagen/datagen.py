# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM
from vllm.sampling_params import BeamSearchParams
import numpy as np
import itertools
from tqdm.auto import tqdm
from typing import Optional, Union, Sequence, cast
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.beam_search import BeamSearchInstance, BeamSearchOutput, BeamSearchSequence, create_sort_beams_key_function
from vllm.entrypoints.llm import LLM as VLLMLLM

# first load some prompts from the GSM8K dataset (~100)
# then generate full beam search results of beam size 4
# store the result in the form of a data structure?
# - first four token, 


# Sample prompts.

INSTRUCTION = r"Please put your final answer within \boxed{}. Please also be concise."
THINKING_FORCE = r"<think>\n"

prompts = [
    "There are three buckets full of oranges. There are 22 oranges in the first bucket, 17 more oranges in the second bucket and 11 fewer oranges in the third bucket than in the second. How many oranges are there in all the buckets?",
]

prompts = [INSTRUCTION + p + THINKING_FORCE for p in prompts]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def beam_search_with_backpointers(
        llm: VLLMLLM,
        prompts: list[Union[TokensPrompt, TextPrompt]],
        params: BeamSearchParams,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        use_tqdm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate sequences using beam search and return token_ids and backpointers as numpy arrays.
        
        Args:
            llm: The LLM instance
            prompts: A list of prompts. Each prompt can be a string or a list of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            
        Returns:
            tuple: (token_ids, backpointers, debug_info)
                - token_ids: numpy array of shape (K, L) where K is beam width and L is max sequence length
                - backpointers: numpy array of shape (K, L) where (i,j) entry indicates which beam (i,j-1) came from
                - debug_info: dictionary containing debug information about the beam search
        """
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        length_penalty = params.length_penalty

        # Get lora requests
        if isinstance(lora_request, Sequence) and len(lora_request) != len(prompts):
            raise ValueError("Lora request list should be the same length as the prompts")

        if lora_request is None or isinstance(lora_request, LoRARequest):
            lora_requests = [lora_request] * len(prompts)
        else:
            lora_requests = lora_request

        tokenizer = llm.get_tokenizer()
        sort_beams_key = create_sort_beams_key_function(
            tokenizer.eos_token_id,
            length_penalty,
        )

        def create_tokens_prompt_from_beam(
                beam: BeamSearchSequence) -> TokensPrompt:
            token_prompt_kwargs: TokensPrompt = {
                "prompt_token_ids": beam.tokens
            }
            if beam.multi_modal_data is not None:
                token_prompt_kwargs["multi_modal_data"] = beam.multi_modal_data

            if beam.mm_processor_kwargs is not None:
                token_prompt_kwargs[
                    "mm_processor_kwargs"] = beam.mm_processor_kwargs
            return TokensPrompt(**token_prompt_kwargs)

        # generate 2 * beam_width candidates at each step
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: list[BeamSearchInstance] = []

        for lora_req, prompt in zip(lora_requests, prompts):
            # Add multimodal processor kwargs & data
            mm_kwargs = {}
            if "multi_modal_data" in prompt:
                mm_kwargs["multi_modal_data"] = prompt["multi_modal_data"]
            if "mm_processor_kwargs" in prompt:
                mm_kwargs["mm_processor_kwargs"] = prompt[
                    "mm_processor_kwargs"]

            if "prompt_token_ids" in prompt:
                prompt = cast(TokensPrompt, prompt)  # Needed for mypy
                prompt_tokens = prompt["prompt_token_ids"]
            else:
                prompt_tokens = tokenizer.encode(prompt["prompt"])

            instances.append(
                BeamSearchInstance(
                    prompt_tokens,
                    lora_request=lora_req,
                    logprobs=None,
                    **mm_kwargs,
                ), )

        token_iter = range(max_tokens)
        if use_tqdm:
            token_iter = tqdm(token_iter,
                                desc="Beam search",
                                unit="token",
                                unit_scale=False)

        # Initialize arrays for each instance
        token_ids_list = []
        backpointers_list = []
        
        for instance in instances:
            # Initialize arrays for this instance
            token_ids = np.full((beam_width, max_tokens), tokenizer.eos_token_id, dtype=np.int32)
            backpointers = np.full((beam_width, max_tokens), -1, dtype=np.int32)
            
            # Fill initial tokens (prompt tokens)
            backpointers[0, :] = 0  # No backpointers for prompt tokens
            
            # Store arrays for this instance
            token_ids_list.append(token_ids)
            backpointers_list.append(backpointers)
        
        for instance in instances:
            for beam in instance.beams:
                beam.rank = 0
                beam.parent_rank = 0
        
        for token_idx in token_iter:
            all_beams: list[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), []))
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances))
            instance_start_and_end: list[tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            # create the corresponding batch entries for prompt & optional lora
            prompts_batch, lora_req_batch = zip(
                *[(create_tokens_prompt_from_beam(beam), beam.lora_request)
                    for beam in all_beams])

            # only runs for one step
            output = llm.generate(prompts_batch,
                                    sampling_params=beam_search_params,
                                    use_tqdm=False,
                                    lora_request=lora_req_batch)

            for instance_idx, (start, end) in enumerate(instance_start_and_end):
                instance = instances[instance_idx]
                instance_new_beams = []
                
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                logprobs=current_beam.logprobs + [logprobs],
                                lora_request=current_beam.lora_request,
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob,
                                multi_modal_data=current_beam.multi_modal_data,
                                mm_processor_kwargs=current_beam.
                                mm_processor_kwargs)
                            
                            # Assign beam parent rank
                            new_beam.parent_rank = current_beam.rank
                            
                            if token_id == tokenizer.eos_token_id and \
                                not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                
                # Sort and select top beams
                sorted_beams = sorted(instance_new_beams,
                                        key=sort_beams_key,
                                        reverse=True)
                instance.beams = sorted_beams[:beam_width]
                
                # Assign beam ranks based on sorting order
                for rank, beam in enumerate(instance.beams):
                    beam.rank = rank
                
                # Update arrays for this instance
                token_ids = token_ids_list[instance_idx]
                backpointers = backpointers_list[instance_idx]
                
                # Fill arrays for this step
                for rank, beam in enumerate(instance.beams):
                    # Fill the latest token_ids
                    token_ids[rank, token_idx] = beam.tokens[-1]
                    # Fill backpointers
                    backpointers[rank, token_idx] = beam.parent_rank

        # Now finalize the arrays for each instance
        final_token_ids_list = []
        final_backpointers_list = []
        outputs = []
        
        for instance_idx, instance in enumerate(instances):
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                    key=sort_beams_key,
                                    reverse=True)
            best_beams = sorted_completed[:beam_width]
            
            if len(best_beams) == 0:
                final_token_ids_list.append(np.array([]))
                final_backpointers_list.append(np.array([]))
                continue
            
            # Find the maximum sequence length
            max_seq_len = max(len(beam.tokens) for beam in best_beams)
            
            # Get the arrays for this instance
            token_ids = token_ids_list[instance_idx]
            backpointers = backpointers_list[instance_idx]
            
            # Trim to actual size
            token_ids = token_ids[:, :max_seq_len]
            backpointers = backpointers[:, :max_seq_len]
            
            final_token_ids_list.append(token_ids)
            final_backpointers_list.append(backpointers)
            
            # Also store the best beams
            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        
        # Prepare debug information
        debug_info = {
            'max_seq_len': max(len(beam.tokens) for instance in instances for beam in instance.beams) if any(instance.beams for instance in instances) else 0,
            'beam_width': beam_width
        }
        
        return outputs, final_token_ids_list, final_backpointers_list, debug_info

def main():
    BEAM_WIDTH = 3
    MAX_TOKENS = 128

    llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    params = BeamSearchParams(beam_width=BEAM_WIDTH, max_tokens=MAX_TOKENS)
    
    # Test the new function
    outputs, token_ids_list, backpointers_list, debug_info = beam_search_with_backpointers(
        llm, 
        [{"prompt": p} for p in prompts], 
        params
    )
    
    print(f"Number of instances: {len(token_ids_list)}")
    for i, (token_ids, backpointers) in enumerate(zip(token_ids_list, backpointers_list)):
        print(f"\nInstance {i}:")
        print(f"  Token IDs shape: {token_ids.shape}")
        print(f"  Backpointers shape: {backpointers.shape}")
        print(f"  Token IDs (first few):")
        print(f"    {token_ids[:3, :10]}")  # Show first 3 beams, first 10 tokens
        print(f"  Backpointers (first few):")
        print(f"    {backpointers[:3, :10]}")  # Show first 3 beams, first 10 positions
    
    # Debug information
    print(f"Max sequence length: {debug_info['max_seq_len']}")
    print(f"Beam width: {debug_info['beam_width']}")
    
    # Show a few backpointer rows in detail for first instance
    if len(backpointers_list) > 0:
        print("\nDetailed backpointers for first instance:")
        for i in range(min(3, BEAM_WIDTH)):
            print(f"Beam {i}: {backpointers_list[0]}")
    
    print("\nGenerated text below.")
    for prompt, output in zip(prompts, outputs):
        for beam_idx in range(BEAM_WIDTH):
            generated_text = output.sequences[beam_idx].text
            print(f"Beam result {beam_idx}: \n{generated_text.replace(prompt, "")}")
            print("-"*60)

if __name__ == "__main__":
    main()