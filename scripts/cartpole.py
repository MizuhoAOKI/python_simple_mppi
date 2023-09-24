import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import ArtistAnimation
from typing import Tuple

class CartPole():
    def __init__(
            self,
            mass_of_cart: float = 1.0,
            mass_of_pole: float = 0.01,
            length_of_pole: float = 2.0,
            max_force_abs: float = 100.0,
            delta_t: float = 0.02,
            visualize: bool = True,
        ) -> None:
        """initialize cartpole environment
        state variables:
            x: horizontal position of the cart
            theta: angle of the pole (positive in the counter-clockwise direction)
            x_dot: horizontal velocity of the cart
            theta_dot: angular velocity of the pole
        control input:
            force: force applied to the cart (positive in the right direction)
        Note: dynamics of the cartpole is given by OpenAI gym implementation; https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        """
        # physical parameters
        self.g = 9.81
        self.mass_of_cart = mass_of_cart
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_force_abs = max_force_abs
        self.delta_t = delta_t

        # visualization settings
        self.cart_w = 1.8
        self.cart_h = 1.0
        self.max_length_of_force_arrow = 4.0
        self.view_x_lim_min, self.view_x_lim_max = -6.0, 6.0
        self.view_y_lim_min, self.view_y_lim_max = -6.0, 6.0

        # reset environment
        self.visualize_flag = visualize
        self.reset()

    def reset(
            self, 
            init_state: np.ndarray = np.array([0.0, np.pi, 0.0, 0.0]), # [x, theta, x_dot, theta_dot]
        ) -> None:
        """reset environment to initial state"""

        # reset state variables
        self.state = init_state

        # clear animation frames
        self.frames = []

        if self.visualize_flag:
            # prepare figure
            self.fig, self.ax = plt.subplots(1, 1, figsize=(9,9))

            # graph layout settings
            self.ax.set_xlim(self.view_x_lim_min, self.view_x_lim_max)
            self.ax.set_ylim(self.view_y_lim_min, self.view_y_lim_max)
            self.ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            self.ax.tick_params(bottom=False, left=False, right=False, top=False)
            self.ax.set_aspect('equal')

    def update(self, u: np.ndarray, delta_t: float = 0.0, append_frame=True) -> None:
        """update state variables"""
        # keep previous states
        x, theta, x_dot, theta_dot= self.state

        # prepare params
        g = self.g
        M = self.mass_of_cart
        m = self.mass_of_pole
        l = self.length_of_pole
        dt = self.delta_t if delta_t == 0.0 else delta_t

        # limit input force
        force = np.clip(u, -self.max_force_abs, self.max_force_abs)[0]

        # get acc. values
        temp = (
            force + (m*l) * theta_dot**2 * np.sin(theta)
        ) / (M + m)
        new_theta_ddot = (g * np.sin(theta) - np.cos(theta) * temp) / (
            l * (4.0 / 3.0 - m * np.cos(theta)**2 / (M + m))
        )
        new_x_ddot = temp - (m*l)  * new_theta_ddot * np.cos(theta) / (M+m)

        # update pos. values
        new_x = x + x_dot * dt
        new_theta = theta + theta_dot * dt
        new_theta = ((new_theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # update vel. values
        new_theta_dot = theta_dot + new_theta_ddot * dt
        new_x_dot = x_dot + new_x_ddot * dt

        # update state variables
        self.state = np.array([new_x, new_theta, new_x_dot, new_theta_dot])

        # record frame
        if append_frame:
            self.append_frame(force)

    def get_state(self) -> np.ndarray:
        """return state variables"""
        return self.state.copy()

    def append_frame(self, force) -> list:
        """draw a frame of the animation."""
        # draw the cartpole
        x, theta, x_dot, theta_dot = self.state
        origin_x, origin_y = x, 0.0
        w = 0.35 # width of the pole
        l = self.length_of_pole # length of the pole
        e = -0.2 # param of the pole shape
        d = 0.05 # param of the pole shape
        pole_shape_x = [e, e, e+d, l-d, l, l, l-d, e+d, e, e]
        pole_shape_y = [0.0, 0.5*w-d, 0.5*w, 0.5*w, 0.5*w-d, -0.5*w+d, -0.5*w, -0.5*w, -0.5*w+d, 0.0]
        rotated_pole_shape_x, rotated_pole_shape_y = self._affine_transform(pole_shape_x, pole_shape_y, theta+0.5*np.pi, [origin_x, origin_y])
        frame = self.ax.plot(rotated_pole_shape_x, rotated_pole_shape_y, color='black', linewidth=2.0, zorder=3)
        frame += self.ax.fill(rotated_pole_shape_x, rotated_pole_shape_y, color='white', zorder=2)

        # draw the cart and the horizontal line
        cart_x, cart_y = x, 0.0
        frame += [self.ax.add_artist(patches.Rectangle(xy=(cart_x-self.cart_w/2.0, cart_y-self.cart_h/2.0), width=self.cart_w, height=self.cart_h, ec="black", linewidth=2.0, fc="white", fill=True, zorder=1))]
        frame += [self.ax.hlines(0,self.view_x_lim_min,self.view_x_lim_max,colors="gray", zorder=0)]

        # draw the joint circle
        joint = patches.Circle([origin_x, origin_y], radius=abs(e)/2.8, fc='white', ec='black', linewidth=2.0, zorder=4)
        frame += [self.ax.add_artist(joint)]

        # draw the information text
        text = "x = {x:>+4.1f} [m], theta = {theta:>+6.1f} [deg], input force = {force:>+6.2f} [N]".format(x=x, theta=np.rad2deg(theta), force=force)
        frame += [self.ax.text(0.5, 1.02, text, ha='center', transform=self.ax.transAxes, fontsize=14, fontfamily='monospace')]

        # draw the arrow if the force is not too small
        if abs(force) > 1.0e-3:
            frame += [self.ax.arrow(cart_x + np.sign(force)*self.cart_w*0.55, cart_y, self.max_length_of_force_arrow*force/self.max_force_abs, 0.0, zorder=3, width=0.1, head_width=0.3, head_length=0.3, fc='black', ec='black')]
        self.frames.append(frame)

    # rotate shape and return location on the x-y plane.
    def _affine_transform(self, xlist: list, ylist: list, angle: float, translation: list=[0.0, 0.0]) -> Tuple[list, list]:
        transformed_x = []
        transformed_y = []
        if len(xlist) != len(ylist):
            print("[ERROR] xlist and ylist must have the same size.")
            raise AttributeError

        for i, xval in enumerate(xlist):
            transformed_x.append((xlist[i])*np.cos(angle)-(ylist[i])*np.sin(angle)+translation[0])
            transformed_y.append((xlist[i])*np.sin(angle)+(ylist[i])*np.cos(angle)+translation[1])
        transformed_x.append(transformed_x[0])
        transformed_y.append(transformed_y[0])
        return transformed_x, transformed_y

    def save_animation(self, filename, interval, movie_writer="ffmpeg") -> None:
        """save animation of the recorded frames (ffmpeg required)"""
        ani = ArtistAnimation(self.fig, self.frames, interval=interval, blit=True)
        ani.save(filename, writer=movie_writer)


if __name__ == "__main__":
    # test simulation with example control input
    sim_step = 100
    delta_t = 0.02
    cartpole = CartPole()
    for i in range(sim_step):
        cartpole.update(u=[10*np.sin(i/10)], delta_t=delta_t) # u is the control input to the cartpole, [ force[N] ]
    # save animation
    cartpole.save_animation("test_simulation_of_cartpole.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file