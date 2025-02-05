# Simple MPPI Implementation with Python with Docker

Model Predictive Path-Integral (MPPI) Control [[G. Williams et al., 2018]](#references) is a promising sampling-based optimal control algorithm.  
This repository is for understanding the basic idea of the algorithm.

## Building docker image

```sh
git clone https://github.com/MizuhoAOKI/python_simple_mppi.git
cd python_simple_mppi
docker build -t dev_mppi:v0.0 -f docker/Dockerfile .
```
Please note, building the image might take a few minutes

## Starting docker container

```sh
docker rm dev_mppi_container || true && docker run -it -v .:/dev_ws/python_simple_mppi --name dev_mppi_container dev_mppi:v0.0 bash
```
Once the container starts, any changes made in the local repo will be reflected inside the container.