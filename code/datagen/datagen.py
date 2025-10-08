# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

from vllm import LLM
from vllm.sampling_params import BeamSearchParams
import numpy as np
import itertools
import torch
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer
from typing import Optional, Union, Sequence, cast
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.beam_search import BeamSearchInstance, BeamSearchOutput, BeamSearchSequence, create_sort_beams_key_function
from vllm.entrypoints.llm import LLM as VLLMLLM
import os
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.train_utils import load_training_data
from dartmath_prompt.get_prompt import get_prompt



def load_gsm8k_prompts(
    data_dir: str = "data/GSM8K/128_128/batch_1", 
    num_train_prompts: int = 20,
    num_test_prompts: int = 5
) -> tuple[list[str], list[str]]:
    """
    Load unique prompts from the GSM8K dataset into strings for training and testing.

    Args:
        data_dir: Directory containing GSM8K data files
        num_train_prompts: Number of unique prompts to load for training
        num_test_prompts: Number of unique prompts to load for testing

    Returns:
        tuple[list[str], list[str]]: (train_prompts, test_prompts)
    """
    total_prompts = num_train_prompts + num_test_prompts
    # fetch many prompts to ensure we get at least total_prompts unique ones
    train_prompt_sequences, _, train_prompt_mask, _, _, _, _, _ = load_training_data(
        data_dir=data_dir, max_samples=total_prompts*50, num_thoughts=None, seed=42
    )

    # train_prompt_sequences: [B, L], train_prompt_mask: [B, L]
    # Find unique prompts (by content, not by tensor identity), ignoring padding tokens as indicated by the mask.
    # Load the GPT2Tokenizer (assumes 'gpt2' model, adjust if needed)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    seen_prompt_strs = set()
    for seq, mask in zip(train_prompt_sequences, train_prompt_mask):
        if len(seen_prompt_strs) >= total_prompts: break
        seq = seq[mask.bool()]
        # Convert to string
        prompt_str = tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        if prompt_str in seen_prompt_strs: continue
        seen_prompt_strs.add(prompt_str)

    if len(seen_prompt_strs) < total_prompts:
        print(f"Could only find {len(seen_prompt_strs)} unique prompts ({total_prompts} requested), proceeding.")
        total_prompts = len(seen_prompt_strs)
    
    # Split into train and test prompts
    all_prompts = list(seen_prompt_strs)
    train_prompts = all_prompts[:num_train_prompts]
    test_prompts = all_prompts[num_train_prompts:num_train_prompts + num_test_prompts]
    
    return train_prompts, test_prompts

def generate_gsm8k_datasets(
    llm: VLLMLLM,
    pad_token_id: int = 151643,
    data_dir: str = "data/GSM8K/128_128/batch_1",
    beam_widths: list[int] = [2, 4],
    num_train_prompts: int = 20,
    num_test_prompts: int = 5,
    max_tokens: int = 128,
    results_per_prompt: int = 1,
    require_answerbox: bool = False,
    stop_at_answer: bool = False,
    keep_done_beams: bool = False,
    length_penalty: float = 1.0,
    temperature: float = 0,
    batch_size: int = 32
) -> dict:
    """
    Generate GSM8K datasets with beam search in format expected by load_training_data.
    
    Args:
        llm: The LLM instance
        pad_token_id: Token ID to use for padding (defaults to 151643, EOS token for DeepSeek-R1-Distill-Qwen-1.5B)
        data_dir: Directory containing GSM8K data files
        beam_widths: List of beam widths to use for generation
        num_train_prompts: Number of unique prompts to load for training
        num_test_prompts: Number of unique prompts to load for testing
        max_tokens: Maximum number of tokens to generate for the COT (not including the prompt)
        results_per_prompt: Number of results to generate per prompt TODO QUESTIONABLE SINCE VANILLA BEAM SEARCH IS
                            DETERMINISTIC HENCE SAMPLING MULTIPLE TIME ON THE SAME PROMPT DOESN'T YIELD DIFFERENT RESULTS.
                            CONSIDER STOCHASTIC BEAM SEARCH?
        require_answerbox: If True, excludes any beams not including the complete "\\boxed{ANSWER}" pattern.
        stop_at_answer: If True, enables special mode where each beam tracks if it has encountered
                        the complete "\\boxed{ANSWER}" pattern. This can be useful for monitoring
                        when sequences reach a complete answer format.
        keep_done_beams: If True, keep beams that are "done" (reached EOS or an answer box) inside
                         the active beam set. These beams are excluded from LLM calls and instead have
                         an EOS token appended at each subsequent step with probability 1.
        length_penalty: Penalizes longer sequences, score = (prob)**{1/(length ** length_penalty)}.
                        Handy to encourage shorter sequences. The higher the stronger the penalty, defaults to 1.0.
        temperature: Determines how variable the generation sequence is. Could be set to a decently high value provided that 
                     with this set to 0, we obtain the exact same result for all beams.
        batch_size: Batch size for processing prompts at once.
        
    Returns:
        Dictionary containing the generated datasets with structure:
        {
            'beam_width_2': {
                'train': {
                    'prompt_sequences': torch.Tensor,
                    'cot_sequences': torch.Tensor,
                    'prompt_mask': torch.Tensor,
                    'cot_mask': torch.Tensor,
                    'backpointers': torch.Tensor,
                    'metadata': dict
                },
                'test': {
                    'prompt_sequences': torch.Tensor,
                    'cot_sequences': torch.Tensor,
                    'prompt_mask': torch.Tensor,
                    'cot_mask': torch.Tensor,
                    'backpointers': torch.Tensor,
                    'metadata': dict
                }
            },
            'beam_width_4': {
                ...
            }
        }
    """
    if results_per_prompt != 1:
        raise Exception("Temporally deprecating results_per_prompt, has to be 1 to be ran as vanilla beam search is deterministic and thus it doesn't make sense.")
    # Load prompts from GSM8K dataset
    train_prompts, test_prompts = load_gsm8k_prompts(data_dir, num_train_prompts*10, num_test_prompts*10)
    
    # Add instruction and thinking force to prompts
    formatted_train_prompts = [get_prompt(query=q, n_shot=4) for q in train_prompts] # provide four examples before generating
    formatted_test_prompts = [get_prompt(query=q, n_shot=4) for q in test_prompts] # provide four examples before generating
    
    datasets = {}
    
    for beam_width in beam_widths:
        print(f"\nGenerating dataset with beam width {beam_width}...")

        # Initialize separate containers for train and test data
        train_data = {
            'prompt_sequences': [],
            'cot_sequences': [],
            'prompt_masks': [],
            'cot_masks': [],
            'backpointers': [],
            'prompts': []
        }
        
        test_data = {
            'prompt_sequences': [],
            'cot_sequences': [],
            'prompt_masks': [],
            'cot_masks': [],
            'backpointers': [],
            'prompts': []
        }

        def process_answers(answers : list[str], correct_val):
            """
            Returns None if:
            - Any answer doesn't include a \\boxed{} or \\fbox{} answer
            - None of the answers reach the correct solution provided
            Otherwise, returns the original answers.
            """
            def get_boxed_answer(ans : str):
                """Returns answer truncated to the boxed answer if it was in answer, or None otherwise."""
                # Look for the left most \\boxed{} expression
                idx = ans.find("\\boxed")
                if idx < 0:
                    idx = ans.find("\\fbox")
                    if idx < 0: return None
                
                # Find the right most brace
                i = idx
                right_brace_idx = None
                num_left_braces_open = 0
                while i < len(ans):
                    if ans[i] == "{":
                        num_left_braces_open += 1
                    if ans[i] == "}":
                        num_left_braces_open -= 1
                        if num_left_braces_open == 0:
                            right_brace_idx = i
                            break
                    i += 1

                if right_brace_idx is None: return None
                
                # return ans[:right_brace_idx+1]
                return ans

            ret = []
            for ans in answers:
                truncated_ans = get_boxed_answer(ans)
                if truncated_ans:
                    ret.append(truncated_ans)
                else:
                    return None
            return ret

        def process_prompts_for_dataset(formatted_prompts, original_prompts, dataset_name, data_container, 
                                        num_prompts_to_gather, batch_size=32):
            """Helper function to process prompts for either train or test dataset.
               Tries to generate as many as specified, stopping if the quota is met."""
            print(f"  Processing {len(formatted_prompts)} {dataset_name} prompts to gather {num_prompts_to_gather} prompts...")
            
            gathered = 0
            prompt_batch, original_prompt_token_id_batch, original_prompt_batch = [], [], []
            batch_acc_num = 0
            
            for i, (formatted_prompt, original_prompt) in enumerate(zip(formatted_prompts, original_prompts)):

                # Add this prompt to the batch until we accumulate enough prompts
                prompt_batch.append({"prompt": formatted_prompt})
                batch_acc_num += 1

                # Tokenize the ORIGINAL prompt to store in dataset
                original_prompt_token_ids = llm.get_tokenizer().encode(original_prompt)
                original_prompt_token_id_batch.append(original_prompt_token_ids)
                original_prompt_batch.append(original_prompt)

                # If we don't have enough prompts yet for batch (but gathered + current prompt number < goal), accumulate more
                if batch_acc_num < batch_size and (gathered + batch_acc_num < num_prompts_to_gather):
                    continue

                # TODO ADJUST TO BE USEFUL IN CASE WE START USING BEAM SEARCH WITH SOME STOCHASTICITY
                gen_count, stop_count = 0, 0
                while gen_count < (results_per_prompt * batch_acc_num):
                    try:
                        params = BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens, 
                                                  length_penalty=length_penalty, temperature=temperature)
                        outputs, token_ids_list, backpointers_list, mask_list, debug_info = beam_search_with_backpointers(
                            llm, prompt_batch, params, require_answerbox=require_answerbox, stop_at_answer=stop_at_answer, keep_done_beams=keep_done_beams)

                        redo_prompt_batch, redo_og_prompt_token_ids_batch, redo_og_prompt_batch = [], [], []
    
                        for out, cot_token_ids, cot_backpointers, cot_mask, og_prompt_token_ids, og_prompt, formatted_prompt_dict in zip(
                            outputs, token_ids_list, backpointers_list, mask_list, original_prompt_token_id_batch, original_prompt_batch, prompt_batch
                            ):
                            # If less than beam_width beams were found for this prompt, generate again until exceeding stop_count
                            if len(out.sequences) < beam_width:
                                redo_prompt_batch.append(formatted_prompt_dict)
                                redo_og_prompt_token_ids_batch.append(og_prompt_token_ids)
                                redo_og_prompt_batch.append(og_prompt)
                            
                            else:
                                # Store ORIGINAL prompt sequence (not the formatted one!)
                                data_container['prompt_sequences'].append(og_prompt_token_ids)
                                data_container['prompt_masks'].append([1] * len(og_prompt_token_ids))
                                
                                # Store COT sequences (one per beam)
                                data_container['cot_sequences'].append(cot_token_ids)
                                data_container['cot_masks'].append(cot_mask)
                                
                                # Store backpointers
                                data_container['backpointers'].append(cot_backpointers)
                                
                                # Store ORIGINAL prompt text (not the formatted one!)
                                data_container['prompts'].append(og_prompt)

                                gen_count += 1
                                gathered += 1
                        

                        prompt_batch = redo_prompt_batch 
                        original_prompt_batch = redo_og_prompt_batch
                        original_prompt_token_id_batch = redo_og_prompt_token_ids_batch
                        
                        stop_count += 1

                        if stop_count > results_per_prompt * 2:
                            print("Exceeded trial numbers, breaking...")
                            break

                    except Exception as e:
                        print(f"      Warning: Failed to generate for {dataset_name} prompt {i+1}, attempt {stop_count+1}: {e}")
                        raise # TODO REMOVE
                        continue
                
                batch_acc_num = 0
                prompt_batch, original_prompt_token_id_batch, original_prompt_batch = [], [], []
                print(f"    Processed {dataset_name.capitalize()} prompt {i+1}/{len(formatted_prompts)}, {gathered}/{num_prompts_to_gather} gathered so far...")
                
                if gathered >= num_prompts_to_gather:
                    break
            
            print(f" Gathered {gathered}/{num_prompts_to_gather} {dataset_name.capitalize()} prompts.")
            print('-'*60)

        # Process train and test prompts
        process_prompts_for_dataset(formatted_train_prompts, train_prompts, "train", train_data, 
                                    num_prompts_to_gather=num_train_prompts, batch_size=batch_size)
        process_prompts_for_dataset(formatted_test_prompts, test_prompts, "test", test_data, 
                                    num_prompts_to_gather=num_test_prompts, batch_size=batch_size)

        def convert_to_tensors(data_container, dataset_name):
            """Helper function to convert lists to tensors with proper padding"""
            if len(data_container['prompt_sequences']) == 0:
                return None
                
            print(f"    Converting {dataset_name} data to tensors...")
            
            # Pad prompt sequences
            max_prompt_len = max(len(seq) for seq in data_container['prompt_sequences'])
            padded_prompts = []
            padded_prompt_masks = []
            
            for seq, mask in zip(data_container['prompt_sequences'], data_container['prompt_masks']):
                # Pad sequence
                padded_seq = seq + [pad_token_id] * (max_prompt_len - len(seq))
                padded_prompts.append(padded_seq)
                
                # Pad mask
                padded_mask = mask + [0] * (max_prompt_len - len(mask))
                padded_prompt_masks.append(padded_mask)
            
            prompt_sequences = torch.tensor(padded_prompts, dtype=torch.long)
            prompt_mask = torch.tensor(padded_prompt_masks, dtype=torch.bool)
            
            # Pad COT sequences and backpointers
            max_cot_len = max(seq.shape[-1] for seq in data_container['cot_sequences'])
            padded_cots = []
            padded_cot_masks = []
            padded_backpointers = []
            
            for seq, mask, bp_seq in zip(data_container['cot_sequences'], data_container['cot_masks'], data_container['backpointers']):
                # Pad sequence
                if seq.ndim == 1: # when beam dimension is 1 and thus was ignored
                    seq = seq.reshape(1, -1)  # Add beam dimension
                    mask = mask.reshape(1, -1)
                    bp_seq = bp_seq.reshape(1, -1)
                
                beam_width_actual = seq.shape[0]
                padded_seq = np.full((beam_width_actual, max_cot_len), pad_token_id, dtype=seq.dtype)
                padded_mask = np.zeros((beam_width_actual, max_cot_len), dtype=mask.dtype)
                padded_bpseq = np.zeros((beam_width_actual, max_cot_len), dtype=seq.dtype)  # 0 for padding
                
                padded_seq[:, :seq.shape[1]] = seq
                padded_mask[:, :mask.shape[1]] = mask
                padded_bpseq[:, :bp_seq.shape[1]] = bp_seq
                
                padded_cots.append(padded_seq) # each is [BEAM_WIDTH, MAX_COT_LEN]
                padded_cot_masks.append(padded_mask)
                padded_backpointers.append(padded_bpseq)
            
            # Stack all COT sequences and reshape to [batch_size, beam_width, seq_len]
            # Each prompt generates beam_width sequences, so we need to group them
            all_cot_arrays = np.stack(padded_cots, axis=0) # [(NUM_PROMTS x RESULT_PER_PROMPT), BEAM_WIDTH, MAX_COT_LEN]
            all_cot_mask_arrays = np.stack(padded_cot_masks, axis=0) # [(NUM_PROMTS x RESULT_PER_PROMPT), BEAM_WIDTH, MAX_COT_LEN]
            all_backpointer_arrays = np.stack(padded_backpointers, axis=0)
            
            # Reshape to [num_prompts, beam_width, seq_len]
            num_prompts = len(data_container['prompts'])
            cot_sequences = torch.tensor(all_cot_arrays.reshape(num_prompts, beam_width, max_cot_len), dtype=torch.long)
            cot_mask = torch.tensor(all_cot_mask_arrays.reshape(num_prompts, beam_width, max_cot_len), dtype=torch.bool)
            backpointers = torch.tensor(all_backpointer_arrays.reshape(num_prompts, beam_width, max_cot_len), dtype=torch.long)
            
            # Create metadata
            metadata = {
                'num_prompts': len(data_container['prompts']),
                'num_sequences': len(data_container['cot_sequences']),
                'beam_width': beam_width,
                'max_prompt_len': max_prompt_len,
                'max_cot_len': max_cot_len,
                'pad_token_id': pad_token_id,
            }
            
            return {
                'prompt_sequences': prompt_sequences,
                'cot_sequences': cot_sequences,
                'prompt_mask': prompt_mask,
                'cot_mask': cot_mask,
                'backpointers': backpointers,
                'metadata': metadata
            }

        # Convert both train and test data to tensors
        train_dataset = convert_to_tensors(train_data, "train")
        test_dataset = convert_to_tensors(test_data, "test")
        
        # Store datasets for this beam width
        datasets[f'beam_width_{beam_width}'] = {}
        if train_dataset:
            datasets[f'beam_width_{beam_width}']['train'] = train_dataset
            print(f"    Generated {len(train_data['cot_sequences'])} train results for beam width {beam_width}")
        if test_dataset:
            datasets[f'beam_width_{beam_width}']['test'] = test_dataset
            print(f"    Generated {len(test_data['cot_sequences'])} test results for beam width {beam_width}")
    
    return datasets

def beam_search_with_backpointers(
        llm: VLLMLLM,
        prompts: list[Union[TokensPrompt, TextPrompt]],
        params: BeamSearchParams,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        use_tqdm: bool = False,
        require_answerbox: bool = False,
        stop_at_answer: bool = False,
        keep_done_beams: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate sequences using beam search and return token_ids and backpointers as numpy arrays.
        
        Args:
            llm: The LLM instance
            prompts: A list of prompts. Each prompt can be a string or a list of token IDs.
            params: The beam search parameters.
            lora_request: LoRA request to use for generation, if any.
            use_tqdm: Whether to use tqdm to display the progress bar.
            require_answerbox: If True, exclude sequence lacking "\\boxed{ANSWER}" pattern before returning best beams. 
            stop_at_answer: If True, track when sequences encounter "\\boxed{ANSWER}" pattern.
                            Each beam tracks if it has found the complete pattern. The final token_ids
                            / backpointers then gets rid of any entries past the pattern for each final beam.
            keep_done_beams: If True, keep beams that are "done" (reached EOS or an answer box) inside
                             the active beam set. These beams are excluded from LLM calls and instead have
                             an EOS token appended at each subsequent step with probability 1.
            
        Returns:
            tuple: (outputs, token_ids, backpointers, debug_info)
                - outputs: list of BeamSearchOutput corresponding to the final K beam obtained
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

        # If stop_at_answer or require_answerbox is enabled, get the token IDs for the special tokens
        if stop_at_answer or require_answerbox:
            # Get token IDs for "\\", "boxed", "{", "}"
            # Note: These are individual tokens that should be individually tokenized
            backslash_token_id = 1124 # for single backslash "\"
            boxed_token_id = tokenizer.encode("boxed")[1] if tokenizer.encode("boxed") else None
            open_brace_token_id = tokenizer.encode("{")[1] if tokenizer.encode("{") else None
            close_brace_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["}", "}.", "}\\"]] + \
                                    [532, 27275, 630, 3417, 47449, 11248, 31716] 
                                    # respectively for: "}\\n", "}.\\n", "}\\n\\n", "}}", "}}\\n\\n", "}}\\n", "}$"
            
            # print(f"backslash_token_id: {backslash_token_id}")
            # print(f"boxed_token_id: {boxed_token_id}")
            # print(f"open_brace_token_id: {open_brace_token_id}")
            # print(f"close_brace_token_ids: {close_brace_token_ids}")
            
            # Verify all required tokens are available
            if any(tid is None for tid in [backslash_token_id, boxed_token_id, open_brace_token_id, close_brace_token_ids]):
                print("Warning: Some required tokens for stop_at_answer/require_answerbox mode not found. Disabling both modes.")
                stop_at_answer = False
                require_answerbox = False

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
            backpointers = np.zeros((beam_width, max_tokens), dtype=np.int32) #empty areas are set as 0
            
            # Fill initial tokens (prompt tokens)
            backpointers[0, :] = 0  # No backpointers for prompt tokens
            
            # Store arrays for this instance
            token_ids_list.append(token_ids)
            backpointers_list.append(backpointers)
        
        for instance in instances:
            for beam in instance.beams:
                beam.rank = 0
                beam.parent_rank = 0
                # Track completion state for the new mode
                beam.done = False
                # Initialize answer tracking for stop_at_answer or require_answerbox mode
                if stop_at_answer or require_answerbox:
                    beam.has_answer = False
                    beam.answer_state = 0  # 0: waiting for \, 1: waiting for boxed, 2: waiting for {, 3: waiting for }, 4: complete
                    beam.last_pos = None # stores the position where the answer pattern was completed
        
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

            # Build batches for only non-done beams if keeping done beams; otherwise include all
            if keep_done_beams:
                active_beams = [beam for beam in all_beams if not getattr(beam, 'done', False)]
                if len(active_beams) > 0:
                    prompts_batch, lora_req_batch = zip(
                        *[(create_tokens_prompt_from_beam(beam), beam.lora_request)
                            for beam in active_beams])
                    output_active = llm.generate(prompts_batch,
                                                 sampling_params=beam_search_params,
                                                 use_tqdm=False,
                                                 lora_request=lora_req_batch)
                else:
                    output_active = []
                active_out_idx = 0
            else:
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

                    # If keeping done beams and this beam is already done, append EOS and carry forward
                    if keep_done_beams and getattr(current_beam, 'done', False):
                        new_beam = BeamSearchSequence(
                            tokens=current_beam.tokens + [tokenizer.eos_token_id],
                            logprobs=current_beam.logprobs + [{}],
                            lora_request=current_beam.lora_request,
                            cum_logprob=current_beam.cum_logprob + 0.0,
                            multi_modal_data=current_beam.multi_modal_data,
                            mm_processor_kwargs=current_beam.mm_processor_kwargs)

                        new_beam.parent_rank = current_beam.rank
                        new_beam.done = True

                        # Preserve answer tracking fields if present
                        if stop_at_answer or require_answerbox:
                            new_beam.has_answer = getattr(current_beam, 'has_answer', False)
                            new_beam.answer_state = getattr(current_beam, 'answer_state', 0)
                            new_beam.last_pos = getattr(current_beam, 'last_pos', None)

                        instance_new_beams.append(new_beam)
                        continue

                    # Otherwise expand using LLM outputs
                    result = output_active[active_out_idx] if keep_done_beams else output[i]
                    if keep_done_beams:
                        active_out_idx += 1

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

                            # Default done state
                            new_beam.done = getattr(current_beam, 'done', False)
                            
                            # Handle answer tracking for stop_at_answer or require_answerbox mode
                            if stop_at_answer or require_answerbox:
                                new_beam.has_answer = current_beam.has_answer
                                new_beam.answer_state = current_beam.answer_state
                                new_beam.last_pos = current_beam.last_pos
                                new_beam.done = current_beam.done
                                
                                # Update answer state based on the new token
                                # IMPORTANT: ASSUMES THAT '\\boxed{' WILL ONLY APPEAR WHEN THE FINAL ANSWER IS
                                # GENERATED, AND THUS DOESN'T CONSIDER WHAT'S BETWEEN '\\boxed{' AND '}' THAT FOLLOWS
                                if not new_beam.has_answer:
                                    if (new_beam.answer_state == 0 and token_id == backslash_token_id) or \
                                       (new_beam.answer_state == 1 and token_id == boxed_token_id) or \
                                       (new_beam.answer_state == 2 and token_id == open_brace_token_id):
                                        new_beam.answer_state += 1
                                    elif new_beam.answer_state == 3 and (token_id in close_brace_token_ids):
                                        new_beam.answer_state = 4
                                        new_beam.has_answer = True
                                        new_beam.done = True
                                    elif new_beam.answer_state < 3:
                                        # Reset state if we encounter a different token
                                        new_beam.answer_state = 0

                            # EOS handling
                            if token_id == tokenizer.eos_token_id and not ignore_eos:
                                if keep_done_beams:
                                    new_beam.done = True
                                    instance_new_beams.append(new_beam)
                                else:
                                    instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                
                # Sort and select top beams
                sorted_beams = sorted(instance_new_beams,
                                        key=sort_beams_key,
                                        reverse=True)
                next_beams = []
                num_stop_beam_seen = 0 # Use when stop_at_answer=True
                
                # Assign beam ranks based on sorting order
                for rank, beam in enumerate(sorted_beams):
                    # Since we might exclude beams reaching an answer, we discount rank for this
                    beam.rank = rank - num_stop_beam_seen
                    
                    if stop_at_answer or require_answerbox:
                        # Store the position where the answer pattern was completed excluding the bracket
                        if beam.has_answer and beam.last_pos is None:
                            beam.last_pos = (token_idx-1, beam.parent_rank)
                    
                            if stop_at_answer and not keep_done_beams:
                                # Move this to 'completed' and consider the next ranking beam for new_beams
                                instance.completed.append(beam)
                                num_stop_beam_seen += 1
                                continue

                    next_beams.append(beam)
                    if len(next_beams) >= beam_width: break
                
                instance.beams = next_beams
                
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
        final_mask_list = []
        outputs = []
        
        for instance_idx, instance in enumerate(instances):
            instance.completed.extend(instance.beams)

            # Apply require_answerbox filter if enabled
            if require_answerbox:
                instance.completed = [beam for beam in instance.completed if beam.last_pos is not None]
            
            sorted_completed = sorted(instance.completed,
                                    key=sort_beams_key,
                                    reverse=True)
            
            # Apply stop_at_answer deduplication if enabled
            if stop_at_answer:
                # Remove duplicate beams that share the same last_pos, excluding those with last_pos=None
                seen_positions = set()
                filtered_beams = []
                for beam in sorted_completed:
                    if beam.last_pos is None:
                        # Always include beams that never encountered the pattern
                        filtered_beams.append(beam)
                    elif beam.last_pos not in seen_positions:
                        # Include beams with new positions
                        seen_positions.add(beam.last_pos)
                        filtered_beams.append(beam)
                sorted_completed = filtered_beams
            
            best_beams = sorted_completed[:beam_width]

            if len(best_beams) == 0:
                outputs.append(BeamSearchOutput(sequences=[]))
                final_token_ids_list.append([ np.array([]) ])
                final_backpointers_list.append([ np.array([]) ])
                final_mask_list.append([ np.array([]) ])
                continue

            # Find the maximum sequence length
            max_seq_len = max(len(beam.tokens) for beam in best_beams)

            # Get the arrays for this instance
            token_ids = token_ids_list[instance_idx]
            backpointers = backpointers_list[instance_idx]

            # Trim to actual size
            token_ids = token_ids[:, :max_seq_len]
            backpointers = backpointers[:, :max_seq_len]

            # If stop_at_answer is enabled, process token_ids and backpointers arrays
            if stop_at_answer:
                # If last_pos isn't assigned, the beam has max length
                for beam in best_beams:
                    if beam.last_pos is None:
                        beam.last_pos = (max_tokens-1, beam.rank)
                
                # Create a boolean mask & terminal_parents of the same shape as the arrays
                mask = np.ones(token_ids.shape, dtype=bool)  # [beam_width, max_seq_len]
                terminal_parents = np.zeros_like(mask, dtype=bool)

                # Set terminal parents at each beam's final position (last_pos)
                leftmost_terminal_pos = mask.shape[1]  # start with max, will take min
                for beam in best_beams:
                    pos, rank = beam.last_pos  # (token_idx, rank)
                    terminal_parents[rank, pos] = True
                    if pos < leftmost_terminal_pos:
                        leftmost_terminal_pos = pos

                # Iterate forward through token positions, starting at the leftmost terminal parent position
                for pos in range(leftmost_terminal_pos, mask.shape[1] - 1):
                    for rank in range(mask.shape[0]):
                        # The parent of (rank, pos+1) is (parent_rank, pos)
                        parent_rank = backpointers[rank, pos + 1]
                        parent_pos = pos
                        # If the parent is a terminal parent, mask out this child and mark it as a new terminal parent
                        if terminal_parents[parent_rank, parent_pos]:
                            mask[rank, pos + 1] = False
                            terminal_parents[rank, pos + 1] = True

                # Fill the parts that aren't True in the mask with fillers
                eos_token_id = tokenizer.eos_token_id
                token_ids = np.where(mask, token_ids, eos_token_id)  # [beam_width, max_seq_len]

                # TODO DEBUG REMOVE
                # print(f"terminal_parents: \n{terminal_parents}")
                # print(f"mask: \n{mask}")
                # print(f"token_ids: \n{token_ids}")
                # print(f"backpointers: \n{backpointers}")
                # TODO END
            else:
                mask = np.ones_like(token_ids, dtype=bool)
            
            final_token_ids_list.append(token_ids)
            final_backpointers_list.append(backpointers)
            final_mask_list.append(mask)
            
            # Also store the best beams
            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        
        # Prepare debug information
        debug_info = {
            'max_seq_len': max(len(beam.tokens) for instance in instances for beam in instance.beams) if any(instance.beams for instance in instances) else 0,
            'beam_width': beam_width
        }
        
        # Add answer tracking information if stop_at_answer or require_answerbox is enabled
        if stop_at_answer:
            debug_info['stop_at_answer'] = True
        if require_answerbox:
            debug_info['require_answerbox'] = True
        
        return outputs, final_token_ids_list, final_backpointers_list, final_mask_list, debug_info



def save_generated_datasets(datasets: dict, save_path: str):
    """
    Save generated datasets in the format expected by load_training_data.
    
    Args:
        datasets: Dictionary containing the generated datasets with train/test splits
        save_path: Base directory to save the datasets
    """
    print("\nSaving datasets...")
    for beam_width_key, beam_datasets in datasets.items():
        print(f"\nProcessing {beam_width_key}...")
        
        # Create save directory structure
        base_save_dir = os.path.join(save_path, beam_width_key)
        train_save_dir = os.path.join(base_save_dir, "train")
        test_save_dir = os.path.join(base_save_dir, "test")
        
        os.makedirs(train_save_dir, exist_ok=True)
        os.makedirs(test_save_dir, exist_ok=True)
        
        train_dataset = beam_datasets.get('train')
        test_dataset = beam_datasets.get('test')
        
        # Save train data if available
        if train_dataset:
            print(f"  Saving train data to {train_save_dir}...")
            torch.save(train_dataset['prompt_sequences'], os.path.join(train_save_dir, "prompt_sequences.pt"))
            torch.save(train_dataset['cot_sequences'], os.path.join(train_save_dir, "cot_sequences_tensor.pt"))
            torch.save(train_dataset['prompt_mask'], os.path.join(train_save_dir, "prompt_mask.pt"))
            torch.save(train_dataset['cot_mask'], os.path.join(train_save_dir, "cot_mask.pt"))
            torch.save(train_dataset['backpointers'], os.path.join(train_save_dir, "backpointers.pt"))
        
        # Save test data if available
        if test_dataset:
            print(f"  Saving test data to {test_save_dir}...")
            torch.save(test_dataset['prompt_sequences'], os.path.join(test_save_dir, "prompt_sequences.pt"))
            torch.save(test_dataset['cot_sequences'], os.path.join(test_save_dir, "cot_sequences_tensor.pt"))
            torch.save(test_dataset['prompt_mask'], os.path.join(test_save_dir, "prompt_mask.pt"))
            torch.save(test_dataset['cot_mask'], os.path.join(test_save_dir, "cot_mask.pt"))
            torch.save(test_dataset['backpointers'], os.path.join(test_save_dir, "backpointers.pt"))
        
        # Save metadata and summary
        summary = {
            'beam_width': train_dataset['metadata']['beam_width'] if train_dataset else test_dataset['metadata']['beam_width'],
            'train_prompts': train_dataset['metadata']['num_prompts'] if train_dataset else 0,
            'test_prompts': test_dataset['metadata']['num_prompts'] if test_dataset else 0,
            'total_prompts': (train_dataset['metadata']['num_prompts'] if train_dataset else 0) + 
                           (test_dataset['metadata']['num_prompts'] if test_dataset else 0),
            'max_prompt_len': train_dataset['metadata']['max_prompt_len'] if train_dataset else test_dataset['metadata']['max_prompt_len'],
            'max_cot_len': train_dataset['metadata']['max_cot_len'] if train_dataset else test_dataset['metadata']['max_cot_len'],
            'pad_token_id': train_dataset['metadata']['pad_token_id'] if train_dataset else test_dataset['metadata']['pad_token_id'],
            'shapes': {}
        }
        
        if train_dataset:
            summary['shapes'].update({
                'train_prompt_sequences': train_dataset['prompt_sequences'].shape,
                'train_cot_sequences': train_dataset['cot_sequences'].shape,
                'train_prompt_mask': train_dataset['prompt_mask'].shape,
                'train_cot_mask': train_dataset['cot_mask'].shape,
                'train_backpointers': train_dataset['backpointers'].shape,
            })
        
        if test_dataset:
            summary['shapes'].update({
                'test_prompt_sequences': test_dataset['prompt_sequences'].shape,
                'test_cot_sequences': test_dataset['cot_sequences'].shape,
                'test_prompt_mask': test_dataset['prompt_mask'].shape,
                'test_cot_mask': test_dataset['cot_mask'].shape,
                'test_backpointers': test_dataset['backpointers'].shape,
            })
        
        with open(os.path.join(base_save_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary to {os.path.join(base_save_dir, 'summary.json')}")
    
    print("\nDataset saving completed!")
    total_examples = sum(
        (beam_datasets.get('train', {}).get('metadata', {}).get('num_prompts', 0) +
         beam_datasets.get('test', {}).get('metadata', {}).get('num_prompts', 0))
        for beam_datasets in datasets.values()
    )
    print(f"Total examples generated: {total_examples}")

def reconstruct_beam_sequences(token_ids, backpointers, mask):
    """
    Reconstruct the final sequences generated by beam search by following backpointers.
    Masked tokens are ignored when reconstructing.
    
    Args:
        token_ids: numpy array of shape (beam_width, seq_len) containing token IDs
        backpointers: numpy array of shape (beam_width, seq_len) where backpointers[i, j] 
                     indicates which beam position (i, j-1) came from
        mask: numpy array of shape (beam_width, seq_len) masking tokens not part of final beams.
    
    Returns:
        List of lists, where each inner list contains the reconstructed token sequence 
        for one beam, following the backpointers from the end and ignoring pad tokens.
    """
    beam_width, seq_len = token_ids.shape
    reconstructed_sequences = []

    # First identify the last tokens for beam_width beams
    last_token_positions = set()
    # There are beam_width such tokens that aren't masked, but which children are
    for pos in range(seq_len - 1, -1, -1):
        for rank in range(beam_width):
            parent_rank, parent_pos = backpointers[rank, pos].item(), pos - 1
            # Last unmasked column entry is a beam ending with max tokens generated
            if (pos == seq_len - 1) and mask[rank, pos]:
                if (rank, pos) not in last_token_positions:
                    last_token_positions.add((rank, pos))
            # Or child is masked but parent isn't, so it is a beam ending
            elif not mask[rank, pos] and mask[parent_rank, parent_pos]:
                if (parent_rank, parent_pos) not in last_token_positions:
                    last_token_positions.add((parent_rank, parent_pos))

            if len(last_token_positions) >= beam_width:
                break
    
    print(f"last_token_positions: {last_token_positions}")
    
    for (last_rank, last_pos) in last_token_positions:
        sequence = []
        current_beam_idx = last_rank
        
        # Work backwards through the sequence
        for pos in range(last_pos, -1, -1):
            # Check if this location is part of 
            # Add the token at this position
            sequence.append(token_ids[current_beam_idx, pos].item())
            
            # Follow the backpointer to the previous position
            if pos > 0:  # Don't follow backpointer for the first position
                current_beam_idx = backpointers[current_beam_idx, pos]
        
        # Reverse to get the sequence in correct order
        sequence.reverse()
        reconstructed_sequences.append(sequence)
    
    return reconstructed_sequences

def visualize_beam_search_tree(token_ids, backpointers, tokenizer=None, max_seq_len=None, show_token_id=True):
    """
    Create a visualization of the beam search tree showing token connections.
    
    Args:
        token_ids: numpy array of shape (beam_width, seq_len) containing token IDs
        backpointers: numpy array of shape (beam_width, seq_len) where backpointers[i, j] 
                     indicates which beam position (i, j-1) came from
        tokenizer: Optional tokenizer to decode token IDs to text
        max_seq_len: Optional maximum sequence length to display (for readability)
        show_token_id: if True, each visualized token string is followed by its token id (Default to False)
    
    Returns:
        String representation of the beam search tree visualization
    """
    beam_width, seq_len = token_ids.shape
    
    # Limit sequence length for readability
    if max_seq_len is not None:
        seq_len = min(seq_len, max_seq_len)
        token_ids = token_ids[:, :seq_len]
        backpointers = backpointers[:, :seq_len]
    
    # Helper function to decode token if tokenizer is available
    def token_to_str(token_id):
        if tokenizer is not None:
            try:
                decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True)
                if show_token_id:
                    decoded += f" [{int(token_id)}]"
                return decoded
            except:
                return f"[{int(token_id)}]"
        else:
            return f"[{int(token_id)}]"
    
    # Define unique symbols for each row
    row_symbols = ['*', '.', "'", '^', '~', '!', '@', '#', '$', '%', '&', '(', ')', '-', '=', '+', '[', ']', '{', '}', ':', ';', '<', '>', ',', '?', '/']
    
    visualization = []

    # Decode the symbols to be shown
    tokens, max_str_len = [], 0
    for beam_idx in range(beam_width):
        row_tokens = []
        for pos in range(seq_len):
            token_str = token_to_str(token_ids[beam_idx, pos])
            row_tokens.append(token_str)
            max_str_len = max(max_str_len, len(token_str))
        tokens.append(row_tokens)

    # Find which positions belong to the last beams
    last_beam_positions = find_last_beam_positions(backpointers)

    # Display header with positions
    header = "Pos:   "
    for pos in range(seq_len):
        header += f"{str(pos).center(max_str_len, ' ')}  |  "
    visualization.append(header)
    visualization.append("-" * len(header))
    
    # Display each beam with tokens and symbols
    for beam_idx in range(beam_width):
        line = f"Rank{beam_idx}: "
        for pos in range(seq_len):
            token_str = tokens[beam_idx][pos].replace('\n', r'\n') # unescape for displaying
            token_str = token_str.center(max_str_len, ' ')
            row_symbol = row_symbols[beam_idx % len(row_symbols)]
            
            # Add parent indicator if not the first position
            if pos < (seq_len - 1):
                nextcol_parent_beam = backpointers[beam_idx, pos+1]
                nextcol_parent_symbol = row_symbols[nextcol_parent_beam % len(row_symbols)]
                token_with_symbols = f"{token_str} {row_symbol}|{nextcol_parent_symbol} "
            else:
                token_with_symbols = f"{token_str}  |  "

            line += f"{token_with_symbols}"
        visualization.append(line)
        
        # Add underlines using the last_beam_positions information
        underline_line = "       "  # Align with "RankX: "
        for pos in range(seq_len):
            # Check if this position belongs to one of the last beams
            beam_ranks = last_beam_positions[beam_idx][pos]
            if type(beam_ranks) is tuple:
                beam_ranks = [str(i) for i in beam_ranks]
                # Position belongs to one of the last beams, add numbered underline
                underline_line += f"{'|'.join(beam_ranks).center(max_str_len, '-')}  |  "
            else:
                # Position doesn't belong to last beams, add simple dashes
                underline_line += f"{'-' * max_str_len}  |  "
        visualization.append(underline_line)
    
    # Add legend
    visualization.append("LEGEND:")
    visualization.append("- Each row shows one beam's tokens")
    visualization.append("- Numbers in [] are token IDs")
    visualization.append("- Text shows decoded tokens")
    visualization.append("- Format: token {row_symbol}{parent_symbol}")
    visualization.append("- {row_symbol} = unique symbol for current row")
    visualization.append("- {parent_symbol} = symbol corresponding to backpointer for next column")
    visualization.append("")
    return "\n".join(visualization)

def find_last_beam_positions(backpointers):
    """
    Find which positions in the backpointers belong to the last M beams by backtracking.
    
    Args:
        backpointers: numpy array of shape (beam_width, seq_len) where backpointers[i, j] 
                     indicates which beam position (i, j-1) came from
    
    Returns:
        List of list of tuple (beam_width, seq_len, variable) where:
        - If position [i, j] belongs to one or more of the last M beams, stores the rank of that beam in a tuple
        - Otherwise, stores -1
    """
    
    beam_width, seq_len = backpointers.shape
    result = np.full((beam_width, seq_len), -1, dtype=int).tolist()

    def fill_result(result, current_beam, current_pos, last_beam_idx):
        curr_val = result[current_beam][current_pos]
        if type(curr_val) is tuple:
            new_val = tuple(list(curr_val) + [last_beam_idx])
        else:
            new_val = (last_beam_idx, )
        result[current_beam][current_pos] = new_val
    
    # For each position in the last column, trace back to find all positions that belong to that beam
    for last_beam_idx in range(beam_width):
        current_beam = last_beam_idx
        current_pos = seq_len - 1
        
        # Mark the last position as belonging to this beam
        fill_result(result, current_beam, current_pos, last_beam_idx)
        
        # Backtrack through the sequence
        while current_pos > 0:
            # Find which beam this position came from
            parent_beam = backpointers[current_beam][current_pos]
            parent_pos = current_pos - 1
            
            # Mark the parent position as belonging to this beam
            fill_result(result, parent_beam, parent_pos, last_beam_idx)
            
            # Continue backtracking
            current_beam = parent_beam
            current_pos = parent_pos
    
    return result

def compute_minimal_edits(seq1, seq2):
    """
    Compute minimal edit operations to transform seq1 into seq2.
    
    Args:
        seq1: First sequence (list of tokens)
        seq2: Second sequence (list of tokens)
    
    Returns:
        List of edit operations, where each operation is a tuple:
        - ('keep', token, pos): Keep token at position pos
        - ('insert', token, pos): Insert token at position pos
        - ('delete', pos): Delete token at position pos
    """
    # Use dynamic programming to find LCS and minimal edits
    m, n = len(seq1), len(seq2)
    
    # dp[i][j] = length of LCS of seq1[:i] and seq2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find edit operations
    edits = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and seq1[i-1] == seq2[j-1]:
            # Tokens match, keep this token
            edits.append(('keep', seq1[i-1], i-1))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            # Insert token from seq2
            edits.append(('insert', seq2[j-1], i))
            j -= 1
        else:
            # Delete token from seq1
            edits.append(('delete', i-1))
            i -= 1
    
    # Reverse to get correct order
    edits.reverse()
    return edits

def format_beam_differences(seq1, seq2, tokenizer=None):
    """
    Format the differences between two sequences showing minimal edits.
    
    Args:
        seq1: First sequence (reference)
        seq2: Second sequence (to be described in terms of edits)
        tokenizer: Optional tokenizer to decode tokens
    
    Returns:
        String describing seq2 in terms of minimal edits relative to seq1
    """
    if seq1 == seq2:
        return "Identical to reference sequence"
    
    edits = compute_minimal_edits(seq1, seq2)
    
    # Group consecutive operations for cleaner display
    grouped_edits = []
    current_group = []
    
    for edit in edits:
        if not current_group or current_group[0][0] == edit[0]:
            current_group.append(edit)
        else:
            grouped_edits.append(current_group)
            current_group = [edit]
    
    if current_group:
        grouped_edits.append(current_group)
    
    cumul_seq1_text_len = []
    for i, token in enumerate(seq1):
        token_text_flat = tokenizer.decode([token], skip_special_tokens=True).replace("\n", "\\n")
        l = len(token_text_flat)
        if i == 0:
            cumul_seq1_text_len.append(l)
        else:
            cumul_seq1_text_len.append(l + cumul_seq1_text_len[-1])

    # Format the description; if tokenizer, we fill changes appropriately into description
    if tokenizer:
        description_insert = " " * cumul_seq1_text_len[-1]
        description_delete = " " * cumul_seq1_text_len[-1]
    else: 
        description = []
    for group in grouped_edits:
        op_type = group[0][0]
        
        if op_type == 'keep':
            # Show range of kept tokens
            start_pos = group[0][2]
            end_pos = group[-1][2]
            if start_pos == end_pos:
                pos_desc = f"position {start_pos}"
            else:
                pos_desc = f"positions {start_pos}-{end_pos}"
            
            if tokenizer is None:
                description.append(f"Keep {len(group)} tokens at {pos_desc}")
        
        elif op_type == 'insert':
            # Show inserted tokens
            if tokenizer:
                tokens = [edit[1] for edit in group]
                token_text_flat = tokenizer.decode(tokens, skip_special_tokens=True).replace("\n", "\\n")
                insert_pos = group[0][2]
                start = cumul_seq1_text_len[insert_pos-1]
                end = start + len(token_text_flat) + 1
                description_insert = description_insert[:start] + \
                                     f"^{token_text_flat}" + \
                                     description_insert[end:]
            else:
                pos_desc = f"position {group[0][2]}"
                description.append(f"Insert {len(group)} tokens at {pos_desc}")
        
        elif op_type == 'delete':
            # Show deleted positions
            positions = [edit[1] for edit in group]
            if len(positions) == 1:
                pos_desc = f"position {positions[0]}"
            else:
                pos_desc = f"positions {min(positions)}-{max(positions)}"
            
            if tokenizer:
                start = cumul_seq1_text_len[min(positions)-1]
                end   = cumul_seq1_text_len[max(positions)]
                description_delete = description_delete[:start] + \
                                     "x" * (end - start) + \
                                     description_delete[end:]
            else:
                description.append(f"Delete at {pos_desc}")
    
    if tokenizer:
        return description_insert, description_delete
    else:
        return "; ".join(description)

def display_beam_comparison(beam_sequences, tokenizer=None, max_beams_to_compare=3):
    """
    Display a comparison between multiple beam sequences, showing minimal differences.
    
    Args:
        beam_sequences: List of reconstructed sequences
        tokenizer: Optional tokenizer to decode tokens
        max_beams_to_compare: Maximum number of beams to compare (for readability)
    """
    if len(beam_sequences) < 2:
        print("Need at least 2 beams to compare")
        return
    
    # Limit number of beams to compare
    num_beams = min(len(beam_sequences), max_beams_to_compare)
    sequences_to_compare = beam_sequences[:num_beams]
    
    print(f"\n{'='*60}")
    print(f"BEAM COMPARISON (showing minimal differences, leaving equal parts empty)")
    print(f"{'='*60}")
    
    # Use first beam as reference
    reference_beam = sequences_to_compare[0]
    
    if tokenizer:
        reference_text = tokenizer.decode(reference_beam, skip_special_tokens=True)
        print(f"\nREFERENCE BEAM 0 (full sequence) + minimal edit descriptions; '^ TEXT' = insert TEXT here, 'x' = delete letter here.")
        print("-" * (len(reference_text) + 10))
        print(f"BEAM 0: {reference_text.replace('\n', '\\n')}")
    else:
        print(f"\nREFERENCE BEAM 0 (full sequence):")
        print("-" * (len(reference_text) + 10))
        print(f"BEAM 0 Tokens: {reference_beam}")
    
    # Compare other beams to reference
    for i in range(1, num_beams):
        print("-" * (len(reference_text) + 50))
        print(f"BEAM {i}: ", end="")
        
        current_beam = sequences_to_compare[i]
        
        if current_beam == reference_beam:
            print("Identical to reference beam")
        else:
            if tokenizer:
                desc_insert, desc_delete = format_beam_differences(reference_beam, current_beam, tokenizer)
                edit_description = desc_insert + "\n      : " + desc_delete
            else:
                edit_description = format_beam_differences(reference_beam, current_beam, tokenizer)
            print(edit_description)
            
            # Also show the full sequence for reference
            # if tokenizer:
            #     current_text = tokenizer.decode(current_beam, skip_special_tokens=True)
            #     print(f"\nFull sequence: {current_text}")
            # else:
            #     print(f"\nFull sequence: {current_beam}"))

def load_and_display_dataset(
    dataset_path: str = "data/GSM8K/generate_test/beam_width_4/train",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_examples_to_show: int = 2
):
    """
    Load a saved dataset and display the decoded prompts and COTs with backpointers.
    
    Args:
        dataset_path: Path to the directory containing the saved dataset files
        model_name: Model name to use for tokenizer (should match the one used for generation)
        max_examples_to_show: Maximum number of examples to display
    """
    import os
    from transformers import AutoTokenizer
    
    print(f"\nLoading dataset from: {dataset_path}")
    print("="*80)
    
    # Check if dataset files exist
    required_files = [
        "prompt_sequences.pt",
        "cot_sequences_tensor.pt", 
        "prompt_mask.pt",
        "cot_mask.pt",
        "backpointers.pt"
    ]
    
    for file in required_files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            print(f"Error: Required file {file} not found at {file_path}")
            return
    
    # Load the dataset tensors
    prompt_sequences = torch.load(os.path.join(dataset_path, "prompt_sequences.pt"))  # [num_prompts, max_prompt_len]
    cot_sequences = torch.load(os.path.join(dataset_path, "cot_sequences_tensor.pt"))  # [num_prompts, beam_width, max_cot_len]
    prompt_mask = torch.load(os.path.join(dataset_path, "prompt_mask.pt"))  # [num_prompts, max_prompt_len]
    cot_mask = torch.load(os.path.join(dataset_path, "cot_mask.pt"))  # [num_prompts, beam_width, max_cot_len]
    backpointers = torch.load(os.path.join(dataset_path, "backpointers.pt"))  # [num_prompts, beam_width, max_cot_len]
    
    print(f"Dataset shapes:")
    print(f"  prompt_sequences: {prompt_sequences.shape}")
    print(f"  cot_sequences: {cot_sequences.shape}")
    print(f"  prompt_mask: {prompt_mask.shape}")
    print(f"  cot_mask: {cot_mask.shape}")
    print(f"  backpointers: {backpointers.shape}")
    
    # Load the tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Display examples
    num_examples = min(max_examples_to_show, prompt_sequences.shape[0])
    beam_width = cot_sequences.shape[1]
    
    for example_idx in range(num_examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {example_idx + 1}")
        print(f"{'='*80}")
        
        # Get prompt data
        prompt_seq = prompt_sequences[example_idx]  # [max_prompt_len]
        prompt_mask_seq = prompt_mask[example_idx]  # [max_prompt_len]
        
        # Decode prompt (only non-padded tokens)
        valid_prompt_tokens = prompt_seq[prompt_mask_seq]  # [actual_prompt_len]
        decoded_prompt = tokenizer.decode(valid_prompt_tokens.tolist(), skip_special_tokens=True)
        
        print(f"\nORIGINAL GSM8K PROMPT ({len(valid_prompt_tokens)} tokens):")
        print(f"{decoded_prompt}")
        print(f" *Note: This is the original GSM8K problem, not the prompt used for generation.")
        
        # Display COT sequences for each beam
        # for beam_idx in range(beam_width):
        #     cot_seq = cot_sequences[example_idx, beam_idx]  # [max_cot_len]
        #     cot_mask_seq = cot_mask[example_idx, beam_idx]  # [max_cot_len]
        #     backpointer_seq = backpointers[example_idx, beam_idx]  # [max_cot_len]
            
        #     # Get valid (non-padded) tokens
        #     valid_cot_tokens = cot_seq[cot_mask_seq]  # [actual_cot_len]
            
        #     print(f"\nCOT BEAM {beam_idx + 1} ({len(valid_cot_tokens)} tokens):")

        #     if len(valid_cot_tokens) == 0:
        #         print(f"    No valid tokens in this beam")
        #         continue
                
        #     # Decode COT
        #     decoded_cot = tokenizer.decode(valid_cot_tokens.tolist(), skip_special_tokens=True)
        #     print("-"*60)
        #     print(f"{decoded_cot}")
        #     print("-"*60)
        
        # Get the full COT arrays for visualization (all beams together)
        example_cot_tokens = cot_sequences[example_idx]  # [beam_width, max_cot_len]
        example_cot_mask = cot_mask[example_idx]  # [beam_width, max_cot_len]
        example_backpointers = backpointers[example_idx]  # [beam_width, max_cot_len]
        
        # Only show valid tokens for each beam by finding the maximum valid length
        max_valid_len = 0
        for beam_idx in range(beam_width):
            valid_len = example_cot_mask[beam_idx].sum().item()
            max_valid_len = max(max_valid_len, valid_len)
        
        if max_valid_len > 0:
            # Create arrays with only valid tokens for visualization
            viz_tokens = example_cot_tokens[:, :max_valid_len].numpy()
            viz_backpointers = example_backpointers[:, :max_valid_len].numpy()

            # Also show reconstructed sequences
            reconstructed_seqs = reconstruct_beam_sequences(viz_tokens, viz_backpointers, mask=example_cot_mask)
            
            # CAN SHOW IT FULLY, OR DIFFERENCE ONLY AS BELOW
            # print(f"\nRECONSTRUCTED SEQUENCES:")
            # for i, seq in enumerate(reconstructed_seqs):
            #     decoded_seq = tokenizer.decode(seq, skip_special_tokens=True)
            #     print(f"Beam {i}:")
            #     print(f"-" * 40)
            #     print(decoded_seq)
            #     print(f"-" * 40)
            
            # Show beam comparison with minimal differences
            if len(reconstructed_seqs) > 1:
                display_beam_comparison(reconstructed_seqs, tokenizer=tokenizer, max_beams_to_compare=len(reconstructed_seqs))
            elif len(reconstructed_seqs) == 1:
                print(f"\nRECONSTRUCTED SEQUENCES (ONLY ONE):")
                seq = reconstructed_seqs[0]
                decoded_seq = tokenizer.decode(seq, skip_special_tokens=True)
                print(f"Beam 0:")
                print(f"-" * 40)
                print(decoded_seq)
                print(f"-" * 40)
            else:
                print("No beam to reconstruct...")

            
            # Show beam search tree visualization for this example
            print(f"\n{'='*60}")
            print(f"BEAM SEARCH TREE VISUALIZATION FOR EXAMPLE {example_idx + 1}")
            print(f"{'='*60}")
            
            # Generate and display the visualization
            tree_viz = visualize_beam_search_tree(
                viz_tokens, 
                viz_backpointers, 
                tokenizer=tokenizer, 
                max_seq_len=None
            )
            print(tree_viz)

        else:
            print("No valid tokens to visualize for this example.")

def test_beam_functions():
    """
    Test the beam search reconstruction and visualization functions with sample data.
    """
    print("\n" + "="*80)
    print("TESTING BEAM SEARCH HELPER FUNCTIONS")
    print("="*80)
    
    # Sample token IDs representing a small beam search
    # Beam width = 3, sequence length = 5
    sample_token_ids = np.array([
        [100, 101, 102, 103, 104],  # Beam 0
        [100, 105, 106, 107, 108],  # Beam 1  
        [100, 101, 109, 110, 111],  # Beam 2
    ])
    
    # Sample backpointers showing how beams branched
    sample_backpointers = np.array([
        [0, 0, 0, 0, 1],  # Beam 0: starts from beam 0, ends from beam 1
        [0, 0, 1, 1, 1],  # Beam 1: starts from beam 0, continues from beam 1
        [0, 0, 0, 2, 2],  # Beam 2: starts from beam 0, continues from beam 2
    ])
    
    print("SAMPLE DATA:")
    print(f"Token IDs shape: {sample_token_ids.shape}")
    print(f"Backpointers shape: {sample_backpointers.shape}")
    print()
    
    # Test reconstruction function
    print("TESTING RECONSTRUCTION FUNCTION:")
    print("-" * 40)
    reconstructed = reconstruct_beam_sequences(sample_token_ids, sample_backpointers,
                        mask=np.ones_like(sample_token_ids))
    
    for i, seq in enumerate(reconstructed):
        print(f"Reconstructed Beam {i}: {seq}")
    print()
    
    # Test visualization function
    print("TESTING VISUALIZATION FUNCTION:")
    print("-" * 40)
    
    # Create a simple mock tokenizer for testing
    class MockTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            # Simple mapping for testing
            token_map = {
                100: "The", 101: "cat", 102: "sat", 103: "on", 104: "mat",
                105: "dog", 106: "ran", 107: "fast", 108: "home",
                109: "jumped", 110: "high", 111: "up"
            }
            if isinstance(token_ids, list) and len(token_ids) == 1:
                return token_map.get(token_ids[0], f"[{token_ids[0]}]")
            else:
                return " ".join([token_map.get(tid, f"[{tid}]") for tid in token_ids])
    
    mock_tokenizer = MockTokenizer()
    
    visualization = visualize_beam_search_tree(
        sample_token_ids, 
        sample_backpointers, 
        tokenizer=mock_tokenizer
    )
    print(visualization)
    
    # Test the new beam comparison functionality
    print("\nTESTING BEAM COMPARISON FUNCTION:")
    print("-" * 40)
    
    # Create some sample sequences that are similar but have small differences
    sample_sequences = [
        [100, 101, 102, 103, 104],  # "The cat sat on mat"
        [100, 101, 102, 105, 103, 104],  # "The cat sat dog on mat" (inserted "dog")
        [100, 101, 106, 103, 104],  # "The cat ran on mat" (replaced "sat" with "ran")
    ]
    
    display_beam_comparison(sample_sequences, tokenizer=mock_tokenizer)

def test_dataset_loading():
    """
    Test function to load and display a generated dataset.
    """
    BEAM_SIZE = 4
    DATADIR_PATH = f"data/GSM8K/generate_test_hightemp/beam_width_{BEAM_SIZE}/"
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print("\n" + "="*80)
    print("TESTING DATASET LOADING")
    print("="*80)
    
    # Test loading the dataset
    load_and_display_dataset(
        dataset_path=DATADIR_PATH+"train",
        model_name=MODEL_NAME,
        max_examples_to_show=20
    )

    load_and_display_dataset(
        dataset_path=DATADIR_PATH+"test",
        model_name=MODEL_NAME,
        max_examples_to_show=20
    )

def main():
    # Generate GSM8K datasets with beam search
    BEAM_WIDTHS = [4]
    MAX_TOKENS = 128
    NUM_TRAIN_PROMPTS = 20
    NUM_TEST_PROMPTS = 20
    RESULTS_PER_PROMPT = 1
    SAVE_PATH = "data/GSM8K/generate_test_hightemp"
    REQUIRE_ANSWERBOX = True
    STOP_AT_ANSWER = True
    KEEP_DONE_BEAMS = True
    LENGTH_PENALTY = 1.0
    TEMPERATURE = 1.0
    BATCH_SIZE = 20

    llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    # Generate datasets
    datasets = generate_gsm8k_datasets(
        llm=llm,
        pad_token_id=151643, # <- for qwen, 50256 for GPT2 eos_token
        data_dir="data/GSM8K/128_128/batch_1",
        beam_widths=BEAM_WIDTHS,
        num_train_prompts=NUM_TRAIN_PROMPTS,
        num_test_prompts=NUM_TEST_PROMPTS,
        max_tokens=MAX_TOKENS,
        results_per_prompt=RESULTS_PER_PROMPT,
        require_answerbox=REQUIRE_ANSWERBOX,
        stop_at_answer=STOP_AT_ANSWER,
        keep_done_beams=KEEP_DONE_BEAMS,
        length_penalty=LENGTH_PENALTY,
        temperature=TEMPERATURE, 
        batch_size=BATCH_SIZE
    )
    
    # Print summary of generated datasets
    print("\n" + "="*60)
    print("DATASET GENERATION SUMMARY")
    print("="*60)
    
    for beam_width_key, beam_datasets in datasets.items():
        print(f"\n{beam_width_key.upper()}:")
        
        train_dataset = beam_datasets.get('train')
        test_dataset = beam_datasets.get('test')
        
        if train_dataset:
            print(f"  TRAIN:")
            print(f"    Number of examples: {train_dataset['metadata']['num_prompts']}")
            
            # Show sample shapes
            sample_prompt_seq = train_dataset['prompt_sequences'][0]
            sample_cot_seq = train_dataset['cot_sequences'][0]
            sample_prompt_mask = train_dataset['prompt_mask'][0]
            sample_cot_mask = train_dataset['cot_mask'][0]
            sample_backpointers = train_dataset['backpointers'][0]

            print(f"    Sample prompt_sequences shape: {sample_prompt_seq.shape}")
            print(f"    Sample cot_sequences shape: {sample_cot_seq.shape}")
            print(f"    Sample prompt_mask shape: {sample_prompt_mask.shape}")
            print(f"    Sample cot_mask shape: {sample_cot_mask.shape}")
            print(f"    Sample backpointers shape: {sample_backpointers.shape}")
            
            # Show sample data
            print(f"    Sample prompt_sequences (first prompt, first 10 tokens):")
            print(f"      {sample_prompt_seq[:10]}")
            print(f"    Sample cot_sequences (first COT, first 10 tokens):")
            print(f"      {sample_cot_seq[0, :10]}")
            print(f"    Sample backpointers (first beam, first 10 positions):")
            print(f"      {sample_backpointers[0, :10]}")
        
        if test_dataset:
            print(f"  TEST:")
            print(f"    Number of examples: {test_dataset['metadata']['num_prompts']}")
            
            # Show sample shapes
            sample_prompt_seq = test_dataset['prompt_sequences'][0]
            sample_cot_seq = test_dataset['cot_sequences'][0]
            sample_prompt_mask = test_dataset['prompt_mask'][0]
            sample_cot_mask = test_dataset['cot_mask'][0]
            sample_backpointers = test_dataset['backpointers'][0]

            print(f"    Sample prompt_sequences shape: {sample_prompt_seq.shape}")
            print(f"    Sample cot_sequences shape: {sample_cot_seq.shape}")
            print(f"    Sample prompt_mask shape: {sample_prompt_mask.shape}")
            print(f"    Sample cot_mask shape: {sample_cot_mask.shape}")
            print(f"    Sample backpointers shape: {sample_backpointers.shape}")
    
    # Save datasets to files
    save_generated_datasets(datasets, SAVE_PATH)

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for different test modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_load":
            test_dataset_loading()
        elif sys.argv[1] == "test_beam":
            test_beam_functions()
        elif sys.argv[1] == "test_compare":
            # Test just the beam comparison functionality
            print("\n" + "="*80)
            print("TESTING BEAM COMPARISON FUNCTIONALITY")
            print("="*80)
            
            # Create a simple mock tokenizer for testing
            class MockTokenizer:
                def decode(self, token_ids, skip_special_tokens=True):
                    # Simple mapping for testing
                    token_map = {
                        100: "The", 101: "cat", 102: "sat", 103: "on", 104: "mat",
                        105: "dog", 106: "ran", 107: "fast", 108: "home",
                        109: "jumped", 110: "high", 111: "up", 112: "quickly"
                    }
                    if isinstance(token_ids, list) and len(token_ids) == 1:
                        return token_map.get(token_ids[0], f"[{token_ids[0]}]")
                    else:
                        return " ".join([token_map.get(tid, f"[{tid}]") for tid in token_ids])
            
            mock_tokenizer = MockTokenizer()
            
            # Test with various types of differences
            test_sequences = [
                [100, 101, 102, 103, 104],  # "The cat sat on mat"
                [100, 101, 102, 105, 103, 104],  # "The cat sat dog on mat" (inserted "dog")
                [100, 101, 106, 103, 104],  # "The cat ran on mat" (replaced "sat" with "ran")
                [100, 101, 102, 103, 104, 112],  # "The cat sat on mat quickly" (added "quickly")
                [100, 101, 102, 103],  # "The cat sat on" (deleted "mat")
            ]
            
            display_beam_comparison(test_sequences, tokenizer=mock_tokenizer, max_beams_to_compare=5)
            
        elif sys.argv[1] == "test_all":
            test_beam_functions()
            test_dataset_loading()
        else:
            print("Unknown command. Available options:")
            print("  python datagen.py                 # Run main generation")
            print("  python datagen.py test_load       # Test dataset loading")
            print("  python datagen.py test_beam       # Test beam search functions")
            print("  python datagen.py test_compare    # Test beam comparison only")
            print("  python datagen.py test_all        # Test all functions")
    else:
        main()
        
        # To test the functions, you can run:
        # python datagen.py test_beam    # Test beam search helper functions
        # python datagen.py test_load    # Test dataset loading
        # python datagen.py test_all     # Test everything