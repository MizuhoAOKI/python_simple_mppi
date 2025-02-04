FROM ubuntu:22.04


SHELL ["/bin/bash", "-c"]

WORKDIR /app/

RUN apt-get update
RUN apt install -y git python3 curl ffmpeg nano

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

RUN git clone https://github.com/MizuhoAOKI/python_simple_mppi.git
RUN cd python_simple_mppi && poetry install --no-root

WORKDIR /app/python_simple_mppi

CMD ["/bin/bash"]