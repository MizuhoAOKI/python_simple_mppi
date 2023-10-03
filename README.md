[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Simple MPPI Implementation with Python
Model Predictive Path-Integral (MPPI) Control is a promising sampling-based optimal control algorithm.  
This repository is for understanding the basic idea of the algorithm.

<img src="./media/pathtracking_demo.gif" width="500px" alt="pathtracking demonstraion">
<img src="./media/pendulum_swingup_demo.gif" width="500px" alt="swinging up pendulum demonstraion">
<img src="./media/cartpole_demo.gif" width="500px" alt="swinging up pendulum demonstraion">
<!-- https://github.com/MizuhoAOKI/python_simple_mppi/assets/63337525/bda8cdbc-5cfd-4885-ac8d-3240867f027c -->

## Dependency
- [poetry](https://python-poetry.org/) : seting up python environment easily.
- [ffmpeg](https://ffmpeg.org/) : mp4 movie writer

## Setup
```sh
git clone https://github.com/MizuhoAOKI/python_simple_mppi.git
cd python_simple_mppi
poetry install
```

## Usage

### Path-Tracking
<!-- [TODO] add image here -->

<!-- [TODO] add scripts/mppi_pathtracking.py -->

- Run path-tracking simulation
    ```sh
    cd python_simple_mppi
    poetry run jupyter notebook notebooks/mppi_pathtracking.ipynb
    ```

### Pendulum
<img src="./media/pendulum.png" width="300px" alt="pendulum">

- Run simulation to swing up a pendulum.
    ```sh
    cd python_simple_mppi
    poetry run python scripts/mppi_pendulum.py
    ```

- Run notebook if you would like to check mathematical explanations on the algorithm. 
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

- Run notebook if you would like to check mathematical explanations on the algorithm. 
    ```sh
    cd python_simple_mppi
    poetry run jupyter notebook notebooks/mppi_cartpole.ipynb
    ```

## References
1. G. Williams et al. "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving" 
    - URL : https://ieeexplore.ieee.org/document/8558663
    - PDF : https://arxiv.org/pdf/1707.02342.pdf
