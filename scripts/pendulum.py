import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import ArtistAnimation
from typing import Tuple

class Pendulum():
    def __init__(
            self,
            mass_of_pole: float = 1.0,
            length_of_pole: float = 1.0,
            max_torque_abs: float = 2.0,
            max_speed_abs: float = 8.0,
            delta_t: float = 0.05,
            visualize: bool = True,
        ) -> None:
        """initialize pendulum environment
        state variables:
            theta: angle of the pole (positive in the counter-clockwise direction)
            theta_dot: angular velocity of the pole
        control input:
            torque: torque applied to the joint (positive in the counter-clockwise direction)
        Note: dynamics of the pendulum is given by OpenAI gym implementation; https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        """
        # physical parameters
        self.g = 9.81
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_speed = max_speed_abs
        self.max_torque = max_torque_abs
        self.delta_t = delta_t

        # visualization settings
        self.max_length_of_torque_arrow = 1.0
        self.view_x_lim_min, self.view_x_lim_max = -2.0, 2.0
        self.view_y_lim_min, self.view_y_lim_max = -2.0, 2.0

        # reset environment
        self.visualize_flag = visualize
        self.reset()

    def reset(
            self, 
            init_state: np.ndarray = np.array([np.pi, 0.0]), # [theta, theta_dot]
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
        # get previous state variables
        theta, theta_dot = self.state

        # prepare params
        g = self.g
        m = self.mass_of_pole
        l = self.length_of_pole
        dt = self.delta_t if delta_t == 0.0 else delta_t

        torque = np.clip(u, -self.max_torque, self.max_torque)[0]
        new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * dt
        new_theta = ((new_theta + np.pi) % (2 * np.pi)) - np.pi # normalize new_theta to [-pi, pi]

        # update state variables
        self.state = np.array([new_theta, new_theta_dot])

        # record frame if necessary
        if append_frame:
            self.append_frame(torque)

    def get_state(self) -> np.ndarray:
        """return state variables"""
        return self.state.copy()

    def append_frame(self, torque: float) -> None:
        """draw a frame of the animation."""
        # draw the pendulum
        theta, theta_dot = self.state
        origin_x, origin_y = 0.0, 0.0
        w = 0.2 # width of the pendulum
        l = self.length_of_pole # length of the pendulum
        e = -0.2 # param of the pendulum shape
        d = 0.05 # param of the pendulum shape
        pendulum_shape_x = [e, e, e+d, l-d, l, l, l-d, e+d, e, e]
        pendulum_shape_y = [0.0, 0.5*w-d, 0.5*w, 0.5*w, 0.5*w-d, -0.5*w+d, -0.5*w, -0.5*w, -0.5*w+d, 0.0]
        rotated_pendulum_shape_x, rotated_pendulum_shape_y = self._affine_transform(pendulum_shape_x, pendulum_shape_y, theta+0.5*np.pi, [origin_x, origin_y])
        frame = self.ax.plot(rotated_pendulum_shape_x, rotated_pendulum_shape_y, color='black', linewidth=2.0)

        # draw the joint circle
        joint = patches.Circle([origin_x, origin_y], radius=abs(e)/3.0, fc='white', ec='black', linewidth=2.0)
        frame += [self.ax.add_artist(joint)]

        # draw the information text
        text = "theta = {theta:>+7.2f} [deg], input torque = {torque:>+6.2f} [Nm]".format(theta=np.rad2deg(self.state[0]), torque=torque)
        frame += [self.ax.text(0.5, 1.02, text, ha='center', transform=self.ax.transAxes, fontsize=16, fontfamily='monospace')]

        # draw the arrow if the torque is not too small
        if abs(torque) > 1.0e-3:
            max_arrow_angle= 120 #[deg]
            angle_width = max_arrow_angle * abs(torque)/self.max_torque
            cent_angle = np.rad2deg(theta+0.5*np.pi)
            angle_s = cent_angle - angle_width/2
            angle_g = cent_angle + angle_width/2
            arrow_loc = 'e' if torque > 0 else 's'
            arrow_obj = self._draw_arc_arrow(self.ax,radius=self.length_of_pole*1.2,cent_x=origin_x,cent_y=origin_y,angle_s=angle_s,angle_g=angle_g,arrow_loc=arrow_loc)
            frame += arrow_obj

        # append frame
        self.frames.append(frame)

    def _draw_arc_arrow(self, ax: plt.axes, radius: float, cent_x: float, cent_y: float, \
                        angle_s: float, angle_g: float, arrow_loc: str='e', color: str='black') -> list:
        # create the arc
        theta2_ = angle_g - angle_s if angle_g > angle_s else angle_g - angle_s + 360
        diameter=2*radius
        arc = patches.Arc([cent_x,cent_y],diameter,diameter,angle=angle_s,
            theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=5,color=color)

        # create the arrow head
        arrow_head_scale_factor = 0.03
        if arrow_loc=='e':
            # locate arrow head at the end of the arc
            goalX=cent_x+(diameter/2)*np.cos(np.radians(theta2_+angle_s)) #Do trig to determine end position
            goalY=cent_y+(diameter/2)*np.sin(np.radians(theta2_+angle_s))
            head = patches.RegularPolygon( #Create triangle as arrow head
                        xy=(goalX, goalY),   # (x,y)
                        numVertices=3,     # number of vertices
                        radius=arrow_head_scale_factor * diameter, # radius of polygon
                        orientation=np.radians(angle_s+theta2_), # orientation
                        color=color
                    )
        elif arrow_loc == 's':
            # locate arrow head at the start of the arc
            startX=cent_x+(diameter/2)*np.cos(np.radians(angle_s)) #Do trig to determine end position
            startY=cent_y+(diameter/2)*np.sin(np.radians(angle_s))
            head = patches.RegularPolygon( #Create triangle as arrow head
                        xy=(startX, startY),   # (x,y)
                        numVertices=3,         # number of vertices
                        radius=arrow_head_scale_factor * diameter, # radius of polygon corners
                        orientation=np.radians(angle_s+180.0), # orientation
                        color=color
                    )
        return [self.ax.add_artist(arc)]+[self.ax.add_artist(head)] # return arrow object

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
    
    def save_animation(self, filename: str, interval: int, movie_writer: str="ffmpeg") -> None:
        """save animation of the recorded frames (ffmpeg required)"""
        ani = ArtistAnimation(self.fig, self.frames, interval=interval)
        ani.save(filename, writer=movie_writer)


if __name__ == "__main__":
    # test simulation with example control input
    sim_step = 100
    delta_t = 0.05
    pendulum = Pendulum()
    for i in range(sim_step):
        pendulum.update(u=[2.0 * np.sin(i/5.0)], delta_t=delta_t) # u is the control input to the pendulum, [ torque[Nm] ]
    # save animation
    pendulum.save_animation("test_simulation_of_pendulum.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file