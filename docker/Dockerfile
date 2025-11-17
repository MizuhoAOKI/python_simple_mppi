FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]

WORKDIR /dev_ws/

RUN apt-get update
RUN apt install -y git ffmpeg nano

RUN git clone https://github.com/MizuhoAOKI/python_simple_mppi.git
WORKDIR /dev_ws/python_simple_mppi
RUN uv sync

WORKDIR /dev_ws/python_simple_mppi