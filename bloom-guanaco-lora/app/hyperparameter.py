import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .config import ModelConfig


class Hyperparameters(BaseModel):
    base_model_name: Optional[str] = None
    disable_8bit: bool = False
    max_epoch: int = 3

    @classmethod
    def load(cls, fpath: Path) -> "Hyperparameters":
        with fpath.open(mode="r") as fp:
            return cls(**json.load(fp))

    def rewrite_config(self, config: ModelConfig) -> ModelConfig:
        res: ModelConfig = config.copy()
        if self.base_model_name is not None:
            res.training.model_name = self.base_model_name
        res.llm.load_in_8bit = not self.disable_8bit
        res.llm.epochs = self.max_epoch
        return res
