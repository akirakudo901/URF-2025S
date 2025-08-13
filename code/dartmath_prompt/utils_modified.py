import logging
import os

# Prompt
PROMPT_TEMPLATE_ID2DICT = {
    "deepseekmath": dict(  # c.f. https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct
        id="deepseekmath",
        sys_prompt="",
        query_prompt="User:" + " ",
        # {query}
        prompt_after_query="\n"
        + "Please reason step by step, and put your final answer within \\boxed{}."
        + "\n\n",
        resp_prompt="Assistant:" + " ",
        prompt_before_resp="",
        # {resp}
        delim="<｜end▁of▁sentence｜>",
    ),
    "deepseek-distill-qwen": dict(
        id="deepseek-distill-qwen",
        sys_prompt="",
        query_prompt="User:" + " ",
        # {query}
        prompt_after_query="\n"
        + "Please reason step by step, and put your final answer within \\boxed{} followed by <｜end▁of▁sentence｜>. Do not repeat what I have described."
        + "\n\n",
        resp_prompt="Assistant:" + " ",
        prompt_before_resp="",
        # {resp}
        delim="<｜end▁of▁sentence｜>",
    ),
    "alpaca": dict(
        id="alpaca",
        sys_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request."
        + "\n\n",
        query_prompt="### Instruction:" + "\n",
        # {query}
        prompt_after_query="\n\n",
        resp_prompt="### Response:" + "\n",
        prompt_before_resp="",
        # {resp}
        delim="\n\n",
    ),
}


# %% ../nbs/99_utils.ipynb 0
class PromptTemplate:
    """Prompt template.
    The complete prompt is in the form `{sys_prompt}{eg_qa1}{delim}{eg_qa2}{delim}...{delim}{eg_qaN}{delim}{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}`.
    default: PROMPT_TEMPLATE_ID2DICT["alpaca"]

    Parameters
    ----------
    id : str
        Short name as ID of the prompt template, like "alpaca".
    sys_prompt : str
        System prompt as the beginning of the full prompt.
    query_prompt : str
        Simple prompt as delimiter between response and new query.
    prompt_after_query : str
        Prompt to append after the raw query, like "Let's think step by step.".
    resp_prompt : str
        Simple prompt as delimiter between query and response.
    delim : str
        Delimiter between query-response pairs.
    """

    def __init__(
        self,
        id: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["id"],
        sys_prompt: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["sys_prompt"],
        query_prompt: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["query_prompt"],
        prompt_after_query: str = PROMPT_TEMPLATE_ID2DICT["alpaca"][
            "prompt_after_query"
        ],
        resp_prompt: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["resp_prompt"],
        prompt_before_resp: str = PROMPT_TEMPLATE_ID2DICT["alpaca"][
            "prompt_before_resp"
        ],
        delim: str = PROMPT_TEMPLATE_ID2DICT["alpaca"]["delim"],
    ):

        self.id = id
        self.sys_prompt = sys_prompt
        self.query_prompt = query_prompt
        self.prompt_after_query = prompt_after_query
        self.resp_prompt = resp_prompt
        self.prompt_before_resp = prompt_before_resp
        self.delim = delim

    @staticmethod
    def load_from_id_or_path(prompt_template: str = "alpaca") -> "PromptTemplate":
        """Load prompt template from ID or file path."""
        if prompt_template in PROMPT_TEMPLATE_ID2DICT:  # ID
            return PromptTemplate(
                **{
                    k: v
                    for k, v in PROMPT_TEMPLATE_ID2DICT[prompt_template].items()
                    if k != "model_ids"
                }
            )
        else:  # Default
            logging.warning("Unknown prompt template, using the default 'alpaca'.")
            return PromptTemplate(**PROMPT_TEMPLATE_ID2DICT["alpaca"])

    def make_prefix_prompt(self, query: str) -> str:
        """Make a prefix prompt of `{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}.rstrip(" ")`.
        NOTE: `.rstrip(" ")` is important for correct tokenization, while some cases need "\\n" at the end.
        """
        return f"{self.query_prompt}{query}{self.prompt_after_query}{self.resp_prompt}{self.prompt_before_resp}".rstrip(
            " "
        )

    def make_qa_pair(self, query: str, response: str) -> str:
        """Make a QA pair of `{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}{response}`."""
        return f"{self.query_prompt}{query}{self.prompt_after_query}{self.resp_prompt}{self.prompt_before_resp}{response}"

    def make_full_prompt(self, query: str, eg_qas: list[tuple[str, str]] = []) -> str:
        """Make full prompt as input to the model.
        Format: f"{sys_prompt}{eg_qa1}{eg_qa2}...{eg_qaN}{query_prompt}{query}{prompt_after_query}{resp_prompt}{prompt_before_resp}".
        """
        eg_qa_strs = [self.make_qa_pair(q, a) for q, a in eg_qas]
        prefix_prompt = self.make_prefix_prompt(query)
        return self.sys_prompt + self.delim.join(eg_qa_strs + [prefix_prompt])