import sys
from typing import Dict, Optional

import torch
import transformers
from datasets import DatasetDict, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import BloomForCausalLM, BloomTokenizerFast, GenerationConfig

from .config import ModelConfig
from .prompt import PromptGenerator


class BloomLoRa:
    def __init__(self, config: ModelConfig, training: bool = True):
        device_map: str = "auto"

        self.config = config
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = BloomForCausalLM.from_pretrained(
            config.training.model_name,
            load_in_8bit=config.llm.load_in_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        self.tokenizer = BloomTokenizerFast.from_pretrained(config.training.model_name)

        if training:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.gradient_checkpointing_enable()  # reduce number of stored activations
            self.model.enable_input_require_grads()

            self.model = get_peft_model(self.model, LoraConfig(**config.lora.dict()))
            if not config.llm.load_in_8bit:
                self.model = self.model.half()

            # unk. we want this to be different from the eos token
            self.tokenizer.pad_token_id = 0
            self.data: DatasetDict = load_dataset(
                "json", data_files=[config.file_path.data_path]
            )

            self.prompt_generator = PromptGenerator(
                self.tokenizer, config.llm.cutoff_len
            )
        else:
            self.model = get_peft_model(self.model, config.file_path.model_dir)

    @classmethod
    def from_finetuned(cls, config: ModelConfig) -> "BloomLoRa":
        return BloomLoRa(config, training=False)

    def fine_tuning(self) -> None:
        generate_prompt = self.prompt_generator.get_generate_method()
        if self.config.training.val_set_size > 0:
            train_val = self.data["train"].train_test_split(
                test_size=self.config.training.val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle().map(generate_prompt)
            val_data = train_val["test"].shuffle().map(generate_prompt)
        else:
            train_data = self.data["train"].shuffle().map(generate_prompt)
            val_data = None

        if torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.config.llm.micro_batch_size,
                gradient_accumulation_steps=self.config.llm.gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=self.config.llm.epochs,
                learning_rate=self.config.llm.learning_rate,
                logging_steps=20,
                optim="adamw_torch",
                evaluation_strategy="steps"
                if self.config.training.val_set_size > 0
                else "no",
                save_strategy="steps",
                eval_steps=200 if self.config.training.val_set_size > 0 else None,
                save_steps=200,
                output_dir=self.config.file_path.model_dir,
                save_total_limit=3,
                load_best_model_at_end=True
                if self.config.training.val_set_size > 0
                else False,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )
        self.model.config.use_cache = False

        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        trainer.train()
        self.model.save_pretrained(self.config.file_path.model_dir)
        print("\n If there's a warning about missing keys above, please disregard :)")

    def evaluate(self, instruction: str, input: Optional[str] = None) -> str:
        prompt_params: Dict[str, str] = {"instruction": instruction}
        if input is not None:
            prompt_params["input"] = input
        prompt = self.prompt_generator.generate(prompt_params)

        generation_config = GenerationConfig(**self.config.generation.dict())
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=prompt["input_ids"].to(self.device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.config.generation.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output: str = self.tokenizer.decode(s)
        return output.split("### Response:")[1].strip()
