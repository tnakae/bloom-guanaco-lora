from typing import Any, Callable, Dict

from transformers import BloomTokenizerFast


class PromptGenerator:
    def __init__(self,
                 tokenizer: BloomTokenizerFast,
                 cutoff_len: int):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

    def generate(self,
                 data_point: Dict[str, str],
                 training: bool = True) -> Dict[str, Any]:
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        user_prompt = (
            (
                "Below is an instruction that describes a task, "
                "paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data_point['instruction']}\n\n"
                "### Input:\n"
                f"{data_point['input']}\n\n"
                "### Response:"
            )
            if data_point["input"] else (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data_point['instruction']}\n\n"
                "### Response:"
            )
        )

        if training:
            len_user_prompt_tokens = (
                len(
                    self.tokenizer(
                        user_prompt,
                        truncation=True,
                        max_length=self.cutoff_len + 1,
                    )["input_ids"]
                )
                - 1
            )  # no eos token

            full_tokens = self.tokenizer(
                user_prompt + data_point["output"],
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )["input_ids"][:-1]

            return {
                "input_ids": full_tokens,
                "labels": [-100] * len_user_prompt_tokens
                + full_tokens[len_user_prompt_tokens:],
                "attention_mask": [1] * (len(full_tokens)),
            }
        else:
            inputs = self.tokenizer(user_prompt, return_tensors="pt")
            return {
                "input_ids": inputs["input_ids"]
            }

    def get_generate_method(self) -> Callable[[Dict[str, str]], Dict[str, Any]]:
        def generator_method(data_point):
            return self.generate(data_point)
        return generator_method
