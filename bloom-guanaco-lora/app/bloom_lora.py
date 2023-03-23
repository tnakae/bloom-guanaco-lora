import os
import sys
from typing import Dict, Union

import bitsandbytes as bnb  # noqa
import torch
import transformers
from datasets import (
    load_dataset,
    DatasetDict,
)
from transformers import (
    BloomTokenizerFast,
    BloomForCausalLM,
)
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from .config import ModelConfig
from .prompt import PromptGenerator


class BloomLoRa:
    def __init__(self, config: ModelConfig):
        device_map: Union[str, Dict[str, int]] = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.config = config
        self.config.llm.world_size = world_size
        self.ddp = world_size != 1

        if self.ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

        self.model = BloomForCausalLM.from_pretrained(
            config.training.model_name,
            load_in_8bit=config.training.load_in_8bit,
            device_map=device_map,
        )
        self.tokenizer = BloomTokenizerFast.from_pretrained(
            config.training.model_name,
            add_eos_token=True
        )

        self.model = prepare_model_for_int8_training(self.model)
        self.model = get_peft_model(self.model,
                                    LoraConfig(**config.lora.dict()))

        # unk. we want this to be different from the eos token
        self.tokenizer.pad_token_id = 0

        self.data: DatasetDict = load_dataset("json",
                                              data_files=config.file_path.data_path)

        self.prompt_generator = PromptGenerator(self.tokenizer,
                                                config.llm.cutoff_len)

    def fine_tuning(self):
        generate_prompt = self.prompt_generator.get_generate_method()
        if self.config.training.val_set_size > 0:
            train_val = self.data["train"].train_test_split(
                test_size=self.config.training.val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"].shuffle().map(generate_prompt)
            val_data = train_val["test"].shuffle().map(generate_prompt)
        else:
            train_data = self.data['train'].shuffle().map(generate_prompt)
            val_data = None

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
                fp16=True,
                logging_steps=20,
                evaluation_strategy="steps" if self.config.training.val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if self.config.training.val_set_size > 0 else None,
                save_steps=200,
                output_dir=self.config.file_path.output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if self.config.training.val_set_size > 0 else False,
                ddp_find_unused_parameters=False if self.ddp else None,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        self.model.config.use_cache = False

        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))

        if torch.__version__ >= "2" and sys.platform != 'win32':
            model = torch.compile(self.model)

        trainer.train()
        model.save_pretrained(self.config.file_path.output_dir)
        print("\n If there's a warning about missing keys above, please disregard :)")
