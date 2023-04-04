FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG PROGRAM_DIR="/opt/ml/code"
ARG MODEL_DIR="/opt/ml/model"

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
COPY data/ ${PROGRAM_DIR}/data/

ENV PATH=${PROGRAM_DIR}:$PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

RUN mkdir -p ${MODEL_DIR}
RUN chmod +x train
