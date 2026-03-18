FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]

WORKDIR /dev_ws/python_simple_mppi
COPY . /dev_ws/python_simple_mppi

RUN apt-get update
RUN apt install -y ffmpeg nano

RUN uv sync