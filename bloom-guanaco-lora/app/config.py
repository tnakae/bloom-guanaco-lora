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
    learning_rate: float = 3e-4
    cutoff_len = 256
    load_in_8bit: bool = True

    @property
    def gradient_accumulation_steps(self) -> int:
        res = self.batch_size // self.micro_batch_size
        return res


class TrainingConfig(BaseModel):
    model_name: str = "bigscience/bloom-7b1"
    val_set_size: int = 2000


class FilePathConfig(BaseModel):
    data_path: str
    model_dir: str


class GenerationParamConfig(BaseModel):
    temperature: float = 0.1
    top_p: float = 0.75
    top_k: int = 40
    num_beams: int = 4
    max_new_tokens: int = 128


class ModelConfig(BaseModel):
    lora: LoRaParamConfig
    llm: LLMConfig
    training: TrainingConfig
    generation: GenerationParamConfig
    file_path: FilePathConfig
