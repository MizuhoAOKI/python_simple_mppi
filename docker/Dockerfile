FROM ubuntu:22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-c"]

# Set uv environment variables for production
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=hardlink

WORKDIR /dev_ws/python_simple_mppi
COPY . /dev_ws/python_simple_mppi

RUN apt update \
    && apt install -y ffmpeg nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --no-cache