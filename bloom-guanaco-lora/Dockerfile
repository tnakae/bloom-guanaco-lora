FROM nvidia/cuda:12.0.1-base-ubuntu22.04

ARG PROGRAM_DIR="/opt/ml/code"

RUN apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3-pip

WORKDIR ${PROGRAM_DIR}/

COPY pyproject.toml poetry.lock train train.py config.yml ${PROGRAM_DIR}/
RUN pip install poetry==1.4.1 \
    && poetry config virtualenvs.in-project true \
    && poetry install

COPY app ${PROGRAM_DIR}/

ENV PATH=${PROGRAM_DIR}:${PYTHON_HOME}/bin:$PATH

RUN chmod +x train
