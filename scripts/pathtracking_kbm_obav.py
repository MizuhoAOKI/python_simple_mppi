import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import patches
from matplotlib.animation import ArtistAnimation

class Vehicle():
    def __init__(
            self,
            wheel_base: float = 2.5, # [m]
            vehicle_width = 3.0, # [m]
            vehicle_length = 4.0, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[-100.0, 0.0], [100.0, 0.0]]),
            obstacle_circles: np.ndarray = np.array([[-2.0, 1.0, 1.0], [2.0, -1.0, 1.0]]), # [obs_x, obs_y, obs_radius]
            delta_t: float = 0.05, # [s]
            visualize: bool = True,
        ) -> None:
        """initialize vehicle environment
        state variables:
            x: x-axis position in the global frame [m]
            y: y-axis position in the global frame [m]
            yaw: orientation in the global frame [rad]
            v: longitudinal velocity [m/s]
        control input:
            steer: front tire angle of the vehicle [rad] (positive in the counterclockwize direction)
            accel: longitudinal acceleration of the vehicle [m/s^2] (positive in the forward direction)
        Note: dynamics of the vehicle is the Kinematic Bicycle Model. 
        """
        # vehicle parameters
        self.wheel_base = wheel_base#[m]
        self.vehicle_w = vehicle_width
        self.vehicle_l = vehicle_length
        self.max_steer_abs = max_steer_abs # [rad]
        self.max_accel_abs = max_accel_abs # [m/s^2]
        self.delta_t = delta_t #[s]
        self.ref_path = ref_path

        # obstacle parameters
        self.obstacle_circles = obstacle_circles

        # visualization settings
        self.view_x_lim_min, self.view_x_lim_max = -20.0, 20.0
        self.view_y_lim_min, self.view_y_lim_max = -25.0, 25.0
        self.minimap_view_x_lim_min, self.minimap_view_x_lim_max = -40.0, 40.0
        self.minimap_view_y_lim_min, self.minimap_view_y_lim_max = -10.0, 40.0

        # reset environment
        self.visualize_flag = visualize
        self.reset()

    def reset(
            self, 
            init_state: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0]), # [x, y, yaw, v]
        ) -> None:
        """reset environment to initial state"""

        # reset state variables
        self.state = init_state

        # clear animation frames
        self.frames = []

        if self.visualize_flag:
            # prepare figure
            self.fig = plt.figure(figsize=(9,9))
            self.main_ax = plt.subplot2grid((3,4), (0,0), rowspan=3, colspan=3)
            self.minimap_ax = plt.subplot2grid((3,4), (0,3))
            self.steer_ax = plt.subplot2grid((3,4), (1,3))
            self.accel_ax = plt.subplot2grid((3,4), (2,3))

            # graph layout settings
            ## main view
            self.main_ax.set_aspect('equal')
            self.main_ax.set_xlim(self.view_x_lim_min, self.view_x_lim_max)
            self.main_ax.set_ylim(self.view_y_lim_min, self.view_y_lim_max)
            self.main_ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            self.main_ax.tick_params(bottom=False, left=False, right=False, top=False)
            ## mini map
            self.minimap_ax.set_aspect('equal')
            self.minimap_ax.axis('off')
            self.minimap_ax.set_xlim(self.minimap_view_x_lim_min, self.minimap_view_x_lim_max)
            self.minimap_ax.set_ylim(self.minimap_view_y_lim_min, self.minimap_view_y_lim_max)
            ## steering angle view
            self.steer_ax.set_title("Steering Angle", fontsize="12")
            self.steer_ax.axis('off')
            ## acceleration view
            self.accel_ax.set_title("Acceleration", fontsize="12")
            self.accel_ax.axis('off')
            
            # apply tight layout
            self.fig.tight_layout()

    def update(
            self, 
            u: np.ndarray, 
            delta_t: float = 0.0, 
            append_frame: bool = True, 
            optimal_traj: np.ndarray = np.empty(0), # predicted optimal trajectory from mppi
            sampled_traj_list: np.ndarray = np.empty(0), # sampled trajectories from mppi
        ) -> None:
        """update state variables"""
        # keep previous states
        x, y, yaw, v = self.state

        # prepare params
        l = self.wheel_base
        dt = self.delta_t if delta_t == 0.0 else delta_t

        # limit control inputs
        steer = np.clip(u[0], -self.max_steer_abs, self.max_steer_abs)
        accel = np.clip(u[1], -self.max_accel_abs, self.max_accel_abs)

        # update state variables
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt
        self.state = np.array([new_x, new_y, new_yaw, new_v])

        # record frame
        if append_frame:
            self.append_frame(steer, accel, optimal_traj, sampled_traj_list)

    def get_state(self) -> np.ndarray:
        """return state variables"""
        return self.state.copy()

    def append_frame(self, steer: float, accel: float, optimal_traj: np.ndarray, sampled_traj_list: np.ndarray) -> list:
        """draw a frame of the animation."""
        # get current states
        x, y, yaw, v = self.state

        ### main view ###
        # draw the vehicle shape
        vw, vl = self.vehicle_w, self.vehicle_l
        vehicle_shape_x = [-0.5*vl, -0.5*vl, +0.5*vl, +0.5*vl, -0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, -0.5*vw, -0.5*vw, 0.0]
        rotated_vehicle_shape_x, rotated_vehicle_shape_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [0, 0]) # make the vehicle be at the center of the figure
        frame = self.main_ax.plot(rotated_vehicle_shape_x, rotated_vehicle_shape_y, color='black', linewidth=2.0, zorder=3)

        # draw wheels
        ww, wl = 0.4, 0.7 #[m]
        wheel_shape_x = np.array([-0.5*wl, -0.5*wl, +0.5*wl, +0.5*wl, -0.5*wl, -0.5*wl])
        wheel_shape_y = np.array([0.0, +0.5*ww, +0.5*ww, -0.5*ww, -0.5*ww, 0.0])

        ## rear-left wheel
        wheel_shape_rl_x, wheel_shape_rl_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, 0.3*vw])
        wheel_rl_x, wheel_rl_y = \
            self._affine_transform(wheel_shape_rl_x, wheel_shape_rl_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_rl_x, wheel_rl_y, color='black', zorder=3)

        ## rear-right wheel
        wheel_shape_rr_x, wheel_shape_rr_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, -0.3*vw])
        wheel_rr_x, wheel_rr_y = \
            self._affine_transform(wheel_shape_rr_x, wheel_shape_rr_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_rr_x, wheel_rr_y, color='black', zorder=3)

        ## front-left wheel
        wheel_shape_fl_x, wheel_shape_fl_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, steer, [0.3*vl, 0.3*vw])
        wheel_fl_x, wheel_fl_y = \
            self._affine_transform(wheel_shape_fl_x, wheel_shape_fl_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_fl_x, wheel_fl_y, color='black', zorder=3)

        ## front-right wheel
        wheel_shape_fr_x, wheel_shape_fr_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, steer, [0.3*vl, -0.3*vw])
        wheel_fr_x, wheel_fr_y = \
            self._affine_transform(wheel_shape_fr_x, wheel_shape_fr_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_fr_x, wheel_fr_y, color='black', zorder=3)

        # draw the vehicle center circle
        vehicle_center = patches.Circle([0, 0], radius=vw/20.0, fc='white', ec='black', linewidth=2.0, zorder=6)
        frame += [self.main_ax.add_artist(vehicle_center)]

        # draw the reference path
        ref_path_x = self.ref_path[:, 0] - np.full(self.ref_path.shape[0], x)
        ref_path_y = self.ref_path[:, 1] - np.full(self.ref_path.shape[0], y)
        frame += self.main_ax.plot(ref_path_x, ref_path_y, color='black', linestyle="dashed", linewidth=1.5)

        # draw the information text
        text = "vehicle velocity = {v:>+6.1f} [m/s]".format(pos_e=x, head_e=np.rad2deg(yaw), v=v)
        frame += [self.main_ax.text(0.5, 0.02, text, ha='center', transform=self.main_ax.transAxes, fontsize=14, fontfamily='monospace')]

        # draw the predicted optimal trajectory from mppi
        if optimal_traj.any():
            optimal_traj_x_offset = np.ravel(optimal_traj[:, 0]) - np.full(optimal_traj.shape[0], x)
            optimal_traj_y_offset = np.ravel(optimal_traj[:, 1]) - np.full(optimal_traj.shape[0], y)
            frame += self.main_ax.plot(optimal_traj_x_offset, optimal_traj_y_offset, color='#990099', linestyle="solid", linewidth=1.5, zorder=5)

        # draw the sampled trajectories from mppi
        if sampled_traj_list.any():
            min_alpha_value = 0.25
            max_alpha_value = 0.35
            for idx, sampled_traj in enumerate(sampled_traj_list):
                # draw darker for better samples
                alpha_value = (1.0 - (idx+1)/len(sampled_traj_list)) * (max_alpha_value - min_alpha_value) + min_alpha_value
                sampled_traj_x_offset = np.ravel(sampled_traj[:, 0]) - np.full(sampled_traj.shape[0], x)
                sampled_traj_y_offset = np.ravel(sampled_traj[:, 1]) - np.full(sampled_traj.shape[0], y)
                frame += self.main_ax.plot(sampled_traj_x_offset, sampled_traj_y_offset, color='gray', linestyle="solid", linewidth=0.2, zorder=4, alpha=alpha_value)

        # draw the circular obstacles in the main view
        for obs in self.obstacle_circles:
            obs_x, obs_y, obs_r = obs
            obs_circle = patches.Circle([obs_x-x, obs_y-y], radius=obs_r, fc='white', ec='black', linewidth=2.0, zorder=0)
            frame += [self.main_ax.add_artist(obs_circle)]

        ### mini map view ###
        frame += self.minimap_ax.plot(self.ref_path[:, 0], self.ref_path[:,1], color='black', linestyle='dashed')
        rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [x, y]) # make the vehicle be at the center of the figure
        frame += self.minimap_ax.plot(rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap, color='black', linewidth=2.0, zorder=3)
        frame += self.minimap_ax.fill(rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap, color='white', zorder=2)

        # draw the circular obstacles in the mini map view
        for obs in self.obstacle_circles:
            obs_x, obs_y, obs_r = obs
            obs_circle = patches.Circle([obs_x, obs_y], radius=obs_r, fc='white', ec='black', linewidth=2.0, zorder=0)
            frame += [self.minimap_ax.add_artist(obs_circle)]

        ### control input view ###
        # steering angle
        MAX_STEER = self.max_steer_abs
        PIE_RATE = 3.0/4.0
        PIE_STARTANGLE = 225 # [deg]
        s_abs = np.abs(steer)
        if steer < 0.0: # when turning right
            steer_pie_obj, _ = self.steer_ax.pie([MAX_STEER*PIE_RATE, s_abs*PIE_RATE, (MAX_STEER-s_abs)*PIE_RATE, 2*MAX_STEER*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        else: # when turning left
            steer_pie_obj, _ = self.steer_ax.pie([(MAX_STEER-s_abs)*PIE_RATE, s_abs*PIE_RATE, MAX_STEER*PIE_RATE, 2*MAX_STEER*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        frame += steer_pie_obj
        frame += [self.steer_ax.text(0, -1, f"{np.rad2deg(steer):+.2f} " + r"$ \rm{[deg]}$", size = 14, horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # acceleration
        MAX_ACCEL = self.max_accel_abs
        PIE_RATE = 3.0/4.0
        PIE_STARTANGLE = 225 # [deg]
        a_abs = np.abs(accel)
        if accel > 0.0:
            accel_pie_obj, _ = self.accel_ax.pie([MAX_ACCEL*PIE_RATE, a_abs*PIE_RATE, (MAX_ACCEL-a_abs)*PIE_RATE, 2*MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        else:
            accel_pie_obj, _ = self.accel_ax.pie([(MAX_ACCEL-a_abs)*PIE_RATE, a_abs*PIE_RATE, MAX_ACCEL*PIE_RATE, 2*MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        frame += accel_pie_obj
        frame += [self.accel_ax.text(0, -1, f"{accel:+.2f} " + r"$ \rm{[m/s^2]}$", size = 14, horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # append frame
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

    def save_animation(self, filename: str, interval: int, movie_writer: str="ffmpeg") -> None:
        """save animation of the recorded frames (ffmpeg required)"""
        ani = ArtistAnimation(self.fig, self.frames, interval=interval)
        ani.save(filename, writer=movie_writer)

if __name__ == "__main__":
    # test simulation with example control input
    sim_step = 100
    delta_t = 0.05
    ref_path = np.genfromtxt('./data/ovalpath.csv', delimiter=',', skip_header=1)
    vehicle = Vehicle(ref_path=ref_path[:, 0:2])
    for i in range(sim_step):
        vehicle.update(u=[0.6 * np.sin(i/3.0), 2.2 * np.sin(i/10.0)], delta_t=delta_t) # u is the control input to the vehicle, [ steer[rad], accel[m/s^2] ]
    # save animation
    vehicle.save_animation("vehicle.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file