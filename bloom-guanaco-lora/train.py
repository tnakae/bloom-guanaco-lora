import logging
from pathlib import Path

import typer
import yaml
from app import BloomLoRa, Hyperparameters, ModelConfig


def init_logger() -> None:
    logging.root.setLevel(level="INFO")
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")


def main(
    hyperparameter_path: Path = Path("/opt/ml/input/config/hyperparameters.json"),
) -> int:
    init_logger()
    logger = logging.getLogger(__name__)

    with open("./config.yml", "r") as fp:
        model_config = ModelConfig(**yaml.safe_load(fp))

    if hyperparameter_path.exists():
        hp = Hyperparameters.load(hyperparameter_path)
        logger.info("hyperparameters.json is detected:")
        logger.info(str(hp))
        model_config = hp.rewrite_config(model_config)

    model = BloomLoRa(model_config)
    model.fine_tuning()

    return 1


if __name__ == "__main__":
    typer.run(main)
