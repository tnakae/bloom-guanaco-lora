FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG PROGRAM_DIR="/opt/ml/code"

RUN apt-get update \
    && apt-get install -y python3.10 python3.10-venv python3-pip \
    && apt-get clean

WORKDIR ${PROGRAM_DIR}/

COPY pyproject.toml poetry.lock ${PROGRAM_DIR}/
RUN pip install poetry==1.4.1 \
    && poetry config virtualenvs.in-project true \
    && poetry install

COPY train train.py config.yml ${PROGRAM_DIR}/
COPY app/ ${PROGRAM_DIR}/app/
COPY data/ ${PRGORAM_DIR}/data/

ENV PATH=${PROGRAM_DIR}:$PATH

RUN chmod +x train
