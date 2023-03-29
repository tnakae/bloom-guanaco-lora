import sys

import typer
import yaml
from app import BloomLoRa, ModelConfig


def main(
    model_output_dir: str = "/opt/ml/model/",
    disable_8bit: bool = typer.Option(False, "--disable-8bit"),
) -> int:
    with open("./config.yml", "r") as fp:
        model_config = ModelConfig(**yaml.safe_load(fp))

    model_config.file_path.model_dir = model_output_dir
    if disable_8bit:
        model_config.llm.load_in_8bit = False

    model = BloomLoRa(model_config)
    model.fine_tuning()

    return 1


if __name__ == "__main__":
    sys.exit(typer.run(main))
