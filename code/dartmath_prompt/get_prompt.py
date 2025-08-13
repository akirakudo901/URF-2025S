# ADDED BY AKIRA: USE AS THE FOLLOWING

import os
import sys

sys.path.append(os.path.dirname(__file__))

from utils_modified import PromptTemplate
from data_modified import ICL_EGS, extract_ans_from_math_sol

PROMPT_TYPE = "deepseek-distill-qwen"
DATASET_TYPE = "gsm8k/test"
N_SHOT = 4

def get_prompt(query : str, n_shot : int):
    if n_shot > N_SHOT:
        print(f"n_shot maximum is {N_SHOT}, adjusting...")
        n_shot = N_SHOT
    prompt_template = PromptTemplate.load_from_id_or_path(PROMPT_TYPE)
    icl_egs = ICL_EGS[DATASET_TYPE][:n_shot]
    input_str = prompt_template.make_full_prompt(query, icl_egs)
    return input_str

if __name__ == "__main__":    
    print(get_prompt("Test, test.", n_shot=2))
    # print("="*60)
    # print("Extracting answer from examples:")


    # for (q, a) in ICL_EGS[DATASET_TYPE]:
    #     print(f"- Query : {q}")
    #     print(f"- CoT : {a}")
    #     print(f"- Answer : {extract_ans_from_math_sol(a)}")
    