from pathlib import Path
from typing import List

from pydantic import BaseModel


class LoRaParamConfig(BaseModel):
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class LLMConfig(BaseModel):
    micro_batch_size: int = 4
    batch_size: int = 128
    epochs: int = 3
    world_size: int = 1
    learning_rate: float = 3e-4
    cutoff_len = 256

    @property
    def gradient_accumulation_steps(self) -> int:
        res = (self.batch_size // self.micro_batch_size)
        res //= self.world_size
        return res


class TrainingConfig(BaseModel):
    model_name: str = "bigscience/bloom-7b1"
    load_in_8bit: bool = True
    val_set_size: int = 2000


class FilePathConfig(BaseModel):
    data_path: Path
    output_dir: Path


class ModelConfig(BaseModel):
    lora: LoRaParamConfig
    llm: LLMConfig
    training: TrainingConfig
    file_path: FilePathConfig
