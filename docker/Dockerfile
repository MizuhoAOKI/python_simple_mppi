FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]

WORKDIR /dev_ws/python_simple_mppi
COPY . /dev_ws/python_simple_mppi

RUN apt update \
    && apt install -y ffmpeg nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync