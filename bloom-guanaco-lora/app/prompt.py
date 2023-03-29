from typing import Any, Callable, Dict, List

from transformers import BloomTokenizerFast


class PromptGenerator:
    def __init__(self, tokenizer: BloomTokenizerFast, cutoff_len: int):
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len

    def generate(
        self, data_point: Dict[str, str], training: bool = True
    ) -> Dict[str, Any]:
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
                "### Response:\n"
            )
            if data_point["input"]
            else (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{data_point['instruction']}\n\n"
                "### Response:\n"
            )
        )

        if training:
            result: Dict[str, List[Any]] = self.tokenizer(
                user_prompt + data_point["output"],
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
                return_tensors=None,
            )

            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result
        else:
            output: Dict[str, Any] = self.tokenizer(user_prompt, return_tensors="pt")
            return {"input_ids": output["input_ids"]}

    def get_generate_method(self) -> Callable[[Dict[str, str]], Dict[str, Any]]:
        def generator_method(data_point: Dict[str, str]) -> Dict[str, Any]:
            return self.generate(data_point)

        return generator_method
