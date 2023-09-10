import math
import numpy as np
from pendulum import Pendulum

class MPPIControllerForPendulum():
    def __init__(
            self,
            delta_t: float = 0.05,
            mass_of_pole: float = 1.0,
            length_of_pole: float = 1.0,
            max_torque_abs: float = 2.0,
            max_speed_abs: float = 8.0,
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_alpha: float = 0.01,
            param_lambda: float = 1.0,
            param_a: float = 0.1,
            sigma: float = 1.0,
            stage_cost_weight: np.ndarray = np.array([1.0, 0.1]),
            terminal_cost_weight: np.ndarray = np.array([1.0, 0.1]),
    ) -> None:
        """initialize mppi controller for pendulum"""
        # mppi parameters
        self.dim_x = 2 # dimension of state vector
        self.dim_u = 1 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_alpha = param_alpha  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_a = param_a # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_a))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        # pendulum parameters
        self.g = 9.81
        self.delta_t = delta_t
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_torque = max_torque_abs
        self.max_speed = max_speed_abs

        # mppi variables
        self.u_prev = np.zeros((self.T))

    def calc_control_input(self, observed_x):
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

            # set initial(t=0) state x i.e. observed state of the pendulum
            x = x0

            # loop for time step t = 1 ~ T
            for t in range(1, self.T+1):

                # get control input with noise
                if k < (1.0-self.param_alpha)*self.K:
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
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=5)

        # update control input sequence
        u += w_epsilon

        # update privious control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u 

    def _calc_epsilon(self, sigma, size_sample, size_time_step):
        """sample epsilon"""
        epsilon = np.random.normal(0.0, sigma, (self.K, self.T)) # size is self.K x self.T
        return epsilon

    def _g(self, v):
        """clamp input"""
        v = np.clip(v, -self.max_torque, self.max_torque)
        return v

    def _c(self, x_t):
        """calculate stage cost"""
        # parse x_t
        theta, theta_dot = x_t[0], x_t[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate stage cost
        stage_cost = self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2
        return stage_cost

    def _phi(self, x_T):
        """calculate terminal cost"""
        # parse x_T
        theta, theta_dot = x_T[0], x_T[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate terminal cost
        terminal_cost = self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2
        return terminal_cost

    def _F(self, x_t, v_t):
        """calculate next state of the pendulum"""
        # get previous state variables
        theta, theta_dot = x_t[0], x_t[1]

        # prepare params
        g = self.g
        m = self.mass_of_pole
        l = self.length_of_pole
        dt = self.delta_t

        # calculate next state
        torque = v_t
        new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * dt

        # return updated state
        x_t_plus_1 = np.array([new_theta, new_theta_dot])
        return x_t_plus_1

    def _compute_weights(self, S):
        """compute weights for each sample"""
        # prepare buffer
        w = np.ones((self.K)) # for debug

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

    def _moving_average_filter(self, xx, window_size):
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

def run_simulation_mppi_pendulum():
    """run simulation of swinging up pendulum with MPPI controller"""
    print("[INFO] Start simulation of swinging up a pendulum with MPPI controller")

    # simulation settings
    delta_t = 0.05 # [sec]
    sim_steps = 150 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # initialize a pendulum as a control target
    pendulum = Pendulum(
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        delta_t = delta_t,
        visualize = True,
    )
    pendulum.reset(
        init_state = np.array([np.pi, 0.0]), # [theta(rad), theta_dot(rad/s)]
    )

    # initialize a mppi controller for the pendulum
    mppi = MPPIControllerForPendulum(
        delta_t = delta_t,
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        horizon_step_T = 20,
        number_of_samples_K = 2000,
        param_alpha = 0.05,
        param_lambda = 0.5,
        param_a = 0.8,
        sigma = 1.0,
        stage_cost_weight    = np.array([1.0, 0.1]), # weight for [theta, theta_dot]
        terminal_cost_weight = 5.0 * np.array([1.0, 0.1]), # weight for [theta, theta_dot]
    )

    # simulation loop
    for i in range(sim_steps):

        # get current state of pendulum
        current_state = pendulum.get_state()

        # calculate input force with MPPI
        input_torque, input_torque_sequence = mppi.calc_control_input(
            observed_x = current_state
        )

        # print current state and input torque
        print(f"Time: {i*delta_t:>2.2f}[s], theta={current_state[0]:>+3.3f}[rad], theta_dot={current_state[1]:>+3.3f}[rad/s], input torque={input_torque:>+3.2f}[Nm]", end="")
        print(", # currently staying upright #" if abs(current_state[0]) < 0.1 and abs(current_state[1] < 0.1) else "")

        # update states of pendulum
        pendulum.update(u=[input_torque], delta_t=delta_t)

    # save animation
    pendulum.save_animation("mppi_pendulum.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file

if __name__ == "__main__":
    run_simulation_mppi_pendulum()
