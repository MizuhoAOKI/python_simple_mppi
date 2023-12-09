import math
import numpy as np
from cartpole import CartPole
from typing import Tuple

class MPPIControllerForCartPole():
    def __init__(
            self,
            delta_t: float = 0.02,
            mass_of_cart: float = 1.0,
            mass_of_pole: float = 0.01,
            length_of_pole: float = 2.0,
            max_force_abs: float = 100.0,
            horizon_step_T: int = 100,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: float = 10.0,
            stage_cost_weight: np.ndarray = np.array([5.0, 10.0, 0.1, 0.1]), # weight for [x, theta, x_dot, theta_dot]
            terminal_cost_weight: np.ndarray = np.array([5.0, 10.0, 0.1, 0.1]) # weight for [x, theta, x_dot, theta_dot]
    ) -> None:
        """initialize mppi controller for cartpole"""
        # mppi parameters
        self.dim_u = 1 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        # cartpole parameters
        self.g = 9.81
        self.delta_t = delta_t
        self.mass_of_cart = mass_of_cart
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_force_abs = max_force_abs

        # mppi variables
        self.u_prev = np.zeros((self.T))

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input"""
        # load privious control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # prepare buffer
        S = np.zeros((self.K)) # state cost list

        # sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T) # size is self.K x self.T

        # loop for 0 ~ K-1 samples
        for k in range(self.K):         
            # prepare buffer
            v = np.zeros((self.T)) # control input sequence with noise

            # set initial(t=0) state x i.e. observed state of the cartpole
            x = x0

            # loop for time step t = 1 ~ T
            for t in range(1, self.T+1):

                # get control input with noise
                if k < (1.0-self.param_exploration)*self.K:
                    v[t-1] = u[t-1] + epsilon[k, t-1] # sampling for exploitation
                else:
                    v[t-1] = epsilon[k, t-1] # sampling for exploration

                # update x
                x = self._F(x, self._g(v[t-1]))

                # add stage cost
                S[k] += self._c(x) + self.param_gamma * u[t-1] * (1.0/self.Sigma) * v[t-1]

            # add terminal cost
            S[k] += self._phi(x)

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)

        # calculate w_k * epsilon_k
        w_epsilon = np.zeros((self.T))
        for t in range(self.T): # loop for time step t = 0 ~ T-1
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # update privious control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u 

    def _calc_epsilon(self, sigma: float, size_sample: int, size_time_step: int) -> np.ndarray:
        """sample epsilon"""
        epsilon = np.random.normal(0.0, sigma, (self.K, self.T)) # size is self.K x self.T
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        v = np.clip(v, -self.max_force_abs, self.max_force_abs)
        return v

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        x, x_dot = x_t[0], x_t[2]
        theta, theta_dot = x_t[1], x_t[3]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate stage cost # (np.cos(theta)+1.0)
        stage_cost = self.stage_cost_weight[0]*x**2 + self.stage_cost_weight[1]*theta**2 + self.stage_cost_weight[2]*x_dot**2 + self.stage_cost_weight[3]*theta_dot**2
        return stage_cost

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, x_dot = x_T[0], x_T[2]
        theta, theta_dot = x_T[1], x_T[3]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate terminal cost # (np.cos(theta)+1.0)
        terminal_cost = self.terminal_cost_weight[0]*x**2 + self.terminal_cost_weight[1]*theta**2 + self.terminal_cost_weight[2]*x_dot**2 + self.terminal_cost_weight[3]*theta_dot**2
        return terminal_cost

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the cartpole"""
        # get previous state variables
        x, theta, x_dot, theta_dot = x_t[0], x_t[1], x_t[2], x_t[3]

        # prepare params
        g = self.g
        M = self.mass_of_cart
        m = self.mass_of_pole
        l = self.length_of_pole
        f = v_t
        dt = self.delta_t

        # get acc. values
        temp = (
            f + (m*l) * theta_dot**2 * np.sin(theta)
        ) / (M + m)
        new_theta_ddot = (g * np.sin(theta) - np.cos(theta) * temp) / (
            l * (4.0 / 3.0 - m * np.cos(theta)**2 / (M + m))
        )
        new_x_ddot = temp - (m*l)  * new_theta_ddot * np.cos(theta) / (M+m)

        # update pos. values
        theta = theta + theta_dot * dt
        x = x + x_dot * dt

        # update vel. values
        theta_dot = theta_dot + new_theta_ddot * dt
        x_dot = x_dot + new_x_ddot * dt

        # return updated state
        x_t_plus_1 = np.array([x, theta, x_dot, theta_dot])
        return x_t_plus_1

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )

        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(window_size/2)
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


def run_simulation_mppi_cartpole() -> None:
    """run simulation of cartpole with MPPI controller"""
    print("[INFO] Start simulation of cartpole with MPPI controller")

    # simulation settings
    delta_t = 0.02 # [sec]
    sim_steps = 200 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # initialize a cartpole as a control target
    cartpole = CartPole(
        mass_of_cart = 1.0,
        mass_of_pole = 0.01,
        length_of_pole = 2.0, 
        max_force_abs = 100.0,
    )
    cartpole.reset(
        init_state = np.array([0.0, np.pi, 0.0, 0.0]), # [x[m], theta[rad], x_dot[m/s], theta_dot[rad/s]]
    )

    # initialize a mppi controller for the cartpole
    mppi = MPPIControllerForCartPole(
        delta_t = delta_t,
        mass_of_cart = 1.0,
        mass_of_pole = 0.01,
        length_of_pole = 2.0,
        max_force_abs = 100.0,
        horizon_step_T = 100,
        number_of_samples_K = 1000,
        param_exploration = 0.0,
        param_lambda = 50.0,
        param_alpha = 1.0,
        sigma = 10.0,
        stage_cost_weight    = np.array([5.0, 10.0, 0.1, 0.1]), # weight for [x, theta, x_dot, theta_dot]
        terminal_cost_weight = np.array([5.0, 10.0, 0.1, 0.1]), # weight for [x, theta, x_dot, theta_dot]
    )

    # simulation loop
    for i in range(sim_steps):

        # get current state of cartpole
        current_state = cartpole.get_state()

        # calculate input force with MPPI
        input_force, input_force_sequence = mppi.calc_control_input(
            observed_x = current_state
        )

        # print current state and input force
        print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], theta={current_state[1]:>+3.3f}[rad], input force={input_force:>+6.2f}[N]")

        # update states of cartpole
        cartpole.update(u=[input_force], delta_t=delta_t)

    # save animation
    cartpole.save_animation("mppi_cartpole.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file

if __name__ == "__main__":
    run_simulation_mppi_cartpole()
