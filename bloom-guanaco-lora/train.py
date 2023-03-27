import sys

import typer
import yaml
from app import BloomLoRa, ModelConfig


def main(model_output_path: str = "/opt/ml/model/") -> int:
    with open("./config.yml", "r") as fp:
        model_config = ModelConfig(**yaml.safe_load(fp))

    model_config.file_path.output_dir = "/opt/ml/"

    model = BloomLoRa(model_config)
    model.fine_tuning()

    return 1


if __name__ == "__main__":
    sys.exit(typer.run(main))
