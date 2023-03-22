from typing import List

from pydantic import BaseModel


class LoRaConfig(BaseModel):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05

class LLMConfig(BaseModel):
    micro_batch_size: int = 4
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 3e-4
    cutoff_len = 256

class TrainingConfig(BaseModel):
    val_set_size: int = 2000
    target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]

class PathConfig(BaseModel):
    


class ModelConfig(BaseModel)
    VAL_SET_SIZE = 2000
    TARGET_MODULES = [
    ]
    DATA_PATH = "alpaca_data_cleaned.json"
    OUTPUT_DIR = "lora-alpaca"

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size
