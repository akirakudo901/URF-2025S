# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM
from vllm.sampling_params import BeamSearchParams

# first load some prompts from the GSM8K dataset (~100)
# then generate full beam search results of beam size 4
# store the result in the form of a data structure?
# - first four token, 


# Sample prompts.

INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}. Please also be concise."
THINKING_FORCE = r"<think>\n"

prompts = [
    "There are three buckets full of oranges. There are 22 oranges in the first bucket, 17 more oranges in the second bucket and 11 fewer oranges in the third bucket than in the second. How many oranges are there in all the buckets?",
]

prompts = [INSTRUCTION + p + THINKING_FORCE for p in prompts]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    BEAM_WIDTH = 5
    MAX_TOKENS = 256

    llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    params = BeamSearchParams(beam_width=BEAM_WIDTH, max_tokens=MAX_TOKENS)
    outputs = llm.beam_search([ { "prompt": p } for p in prompts], params)

    print("Generated text below.")
    for prompt, output in zip(prompts, outputs):
        for beam_idx in range(BEAM_WIDTH):
            generated_text = output.sequences[beam_idx].text
            print(f"Beam result {beam_idx}: \n{generated_text.replace(prompt, "")}")
            print("-"*60)

if __name__ == "__main__":
    main()