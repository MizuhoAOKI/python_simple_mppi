[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Docker](https://img.shields.io/badge/-Docker-EEE.svg?logo=docker&style=flat)](https://www.docker.com/)

# Simple MPPI Implementation with Python
Model Predictive Path-Integral (MPPI) Control [[G. Williams et al., 2018]](#references) is a promising sampling-based optimal control algorithm.  
This repository is for understanding the basic idea of the algorithm.

<img src="./media/pathtracking_obav_demo.gif" width="500px" alt="pathtracking and obstacle avoidance demonstraion">
<img src="./media/pathtracking_demo.gif" width="500px" alt="pathtracking demonstraion">
<img src="./media/pendulum_swingup_demo.gif" width="500px" alt="swinging up pendulum demonstraion">
<img src="./media/cartpole_demo.gif" width="500px" alt="swinging up pendulum demonstraion">
<!-- https://github.com/MizuhoAOKI/python_simple_mppi/assets/63337525/bda8cdbc-5cfd-4885-ac8d-3240867f027c -->

## Dependency

- [python](https://www.python.org/)
  - version 3.10 or higher is recommended.

- [poetry](https://python-poetry.org/)
  - setting up python environment easily and safely.
  - only `numpy`, `matplotlib`, `notebook` are needed to run all scripts in this repository.

- [ffmpeg](https://ffmpeg.org/)
  - mp4 movie writer
  - <details>
    <summary>installation details</summary>

    - For Ubuntu Users
        - `sudo apt-get update`
        - `sudo apt-get -y install ffmpeg`
    - For Windows Users
        - Install [scoop](https://scoop.sh/)
        - `scoop install ffmpeg`
    - For macOS Users
        - Install [homebrew](https://brew.sh/)
        - `brew install ffmpeg`
    - Check the official website if necessary
        - https://ffmpeg.org/

    </details>

## Setup
### [Option 1] Native environment
```sh
git clone https://github.com/MizuhoAOKI/python_simple_mppi.git
cd python_simple_mppi
poetry install
```

### [Option 2] Docker environment
<details>
<summary>CLICK HERE TO EXPAND</summary>

1. Install [docker](https://docs.docker.com/engine/install/).

1. Clone the project repository.
    ```
    cd <path to your workspace>
    git clone https://github.com/MizuhoAOKI/python_simple_mppi.git
    ```

1. Run for the first time setup to build the docker image. Building the image might take a few minutes.
    ```
    cd <path to your workspace>/python_simple_mppi
    docker build -t dev_mppi:v0.0 -f docker/Dockerfile .
    ```

1. Launch the docker container and get into the bash inside.
    ```
    cd <path to your workspace>/python_simple_mppi
    docker run -it -v .:/dev_ws/python_simple_mppi --name dev_mppi_container dev_mppi:v0.0 bash
    ```
    Once the container starts, any changes made in the local repository on the host will be reflected inside the container, and vice versa.

</details>


## Usage

### Path Tracking
<img src="./media/pathtracking.png" width="300px" alt="pendulum">

#### Simple Path Tracking
- Run simulation
    ```sh
    cd python_simple_mppi
    poetry run python scripts/mppi_pathtracking.py
    ```

- Run jupyter notebook if you would like to check mathematical explanations on the algorithm.
    ```sh
    cd python_simple_mppi
    poetry run jupyter notebook notebooks/mppi_pathtracking.ipynb
    ```

#### Path Tracking with Obstacle Avoidance
- Run simulation
    ```sh
    cd python_simple_mppi
    poetry run python scripts/mppi_pathtracking_obav.py
    ```

- Run jupyter notebook if you would like to check mathematical explanations on the algorithm.
    ```sh
    cd python_simple_mppi
    poetry run jupyter notebook notebooks/mppi_pathtracking_obav.ipynb
    ```

### Pendulum
<img src="./media/pendulum.png" width="300px" alt="pendulum">

- Run simulation to swing up a pendulum.
    ```sh
    cd python_simple_mppi
    poetry run python scripts/mppi_pendulum.py
    ```

- Run jupyter notebook if you would like to check mathematical explanations on the algorithm.
    ```sh
    cd python_simple_mppi
    poetry run jupyter notebook notebooks/mppi_pendulum.ipynb
    ```

### CartPole
<img src="./media/cartpole.png" width="300px" alt="cartpole">

- Run simulation of cartpole
    ```sh
    cd python_simple_mppi
    poetry run python scripts/mppi_cartpole.py
    ```

- Run jupyter notebook if you would like to check mathematical explanations on the algorithm.
    ```sh
    cd python_simple_mppi
    poetry run jupyter notebook notebooks/mppi_cartpole.ipynb
    ```

## References
1. G. Williams et al. "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving" 
    - URL : https://ieeexplore.ieee.org/document/8558663
    - PDF : https://arxiv.org/pdf/1707.02342.pdf
