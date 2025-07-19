from enum import Enum
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
import gymnasium as gym
from gymnasium import spaces


class Actions(Enum):
    FORWARD_LEFT = 0
    FORWARD_RIGHT = 1
    BACKWARD_RIGHT = 2
    BACKWARD_LEFT = 3


# TODO: steps limit and truncated condition
class Breaker(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str = "rgb_array", continuous: bool = False):
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(3,), dtype=np.float32
        )  # agent pose: x, y, theta
        if continuous:
            self.action_space = spaces.Box(
                low=np.array([-0.2, -np.pi / 6]),
                high=np.array([0.2, np.pi / 6]),
                shape=(2,),
                dtype=np.float32,
            )  # agent cmd_vel: lin, ang
        else:
            self.action_space = spaces.Discrete(4)
            self._action_codebook = {
                Actions.FORWARD_LEFT.value: np.array(
                    [0.1, np.pi / 12], dtype=np.float32
                ),
                Actions.FORWARD_RIGHT.value: np.array(
                    [0.1, -np.pi / 12], dtype=np.float32
                ),
                Actions.BACKWARD_RIGHT.value: np.array(
                    [-0.1, np.pi / 12], dtype=np.float32
                ),
                Actions.BACKWARD_LEFT.value: np.array(
                    [-0.1, -np.pi / 12], dtype=np.float32
                ),
            }  # body frame: x pointing head, y pointing left
        """
        If human-rendering is used, `self.canvas` will be a reference
        to the window that we render to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # Init fixed patches
        outer_wall = Circle(xy=(0, 0), radius=8.0, fc="grey", ec=None)
        inner_wall = Circle(xy=(0, 0), radius=7.0, fc="white", ec=None)
        doorway = Wedge(
            center=(0, 0), r=8.0, theta1=85.0, theta2=95, fc="white", ec=None
        )
        self._fixed_patches = [outer_wall, inner_wall, doorway]
        # Env vars
        self._max_episode_steps = 500
        self._continuous = continuous
        self._agent_pose = None
        self._agent_status = None
        self._agent_traj = []
        self._fig = None
        self._ax = None

    def _get_obs(self):
        return self._agent_pose

    def _get_info(self):
        return {
            "distance to exit": np.linalg.norm(
                self._agent_pose[:2] - np.array([0.0, 8.0]), ord=2
            ),
            "status": self._agent_status,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[str] = None):
        super().reset(seed=seed)
        # reset escaper to origin or randomly
        self._step_counts = 0
        self._agent_traj = []
        self._agent_pose = np.zeros(3, dtype=np.float32)  # x, y, th
        if options == "random":
            rho = np.random.uniform(6.5)
            alpha = np.random.uniform(low=-np.pi, high=np.pi)
            self._agent_pose = np.array(
                (
                    rho * np.cos(alpha),
                    rho * np.sin(alpha),
                    np.random.uniform(-np.pi, np.pi),
                ),
                dtype=np.float32,
            )
        self._agent_traj.append(self._agent_pose[:2].copy())
        # get obs and info
        self._agent_status = "trapped"
        observation = self._get_obs()
        info = self._get_info()
        # prepare renderer
        if self.render_mode == "human":
            # render
            self._render_frame()

        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        # Init returns
        observation = None
        terminated = False
        truncated = False
        reward = None
        info = None
        self._step_counts += 1
        # Update agent pose
        if self._continuous:
            action = np.clip(
                action,
                a_min=self.action_space.low,
                a_max=self.action_space.high,
            )
            vx = action[0]
            vth = action[1]
        else:
            vx = self._action_codebook[action][0]
            vth = self._action_codebook[action][1]  # rotate along z
        last_pose = self._agent_pose.copy()
        dx = vx * np.cos(last_pose[-1])  # dx = vx cos(theta) * dt
        dy = vx * np.sin(last_pose[-1])  # dy = vx sin(theta) * dt
        dth = vth  # dtheta = vtheta * dt, dt ignored
        self._agent_pose += np.array((dx, dy, dth), dtype=np.float32)
        if self._agent_pose[-1] > np.pi:  # orientation within (-pi, pi)
            self._agent_pose[-1] -= 2 * np.pi
        elif self._agent_pose[-1] < -np.pi:
            self._agent_pose[-1] += 2 * np.pi
        observation = self._get_obs()
        # Reward: higher rewards if agent could lower |x| and increase y
        reward = (
            np.abs(last_pose[0])
            - np.abs(self._agent_pose[0])
            + self._agent_pose[1]
            - last_pose[1]
        )  # |x0| - |x1| + (y1 - y0)
        self._agent_traj.append(self._agent_pose[:2].copy())  # track x, y
        # Check termination
        if self._fixed_patches[0].contains_point(self._agent_pose[:2], radius=0.1):
            if not self._fixed_patches[1].contains_point(
                self._agent_pose[:2], radius=0.1
            ):
                if not self._fixed_patches[2].contains_point(
                    self._agent_pose[:2], radius=0.1
                ):
                    terminated = True
                    self._agent_status = "crashed"
        else:
            reward = 100.0  # escape reward
            terminated = True
            self._agent_status = "escaped"
        if self._step_counts >= self._max_episode_steps:
            truncated = True
        info = self._get_info()
        # render
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):  # TODO: rgb_array
        if self._fig is None:
            self._fig = plt.figure(figsize=(8, 8))
            self._ax = self._fig.add_subplot(111)
        self._ax = self._fig.get_axes()[0]
        self._ax.cla()
        patch_list = []
        patch_list += self._fixed_patches
        # add patches
        escaper_pat = Circle(
            xy=(self._agent_pose[0], self._agent_pose[1]),
            radius=0.2,
            ec="blueviolet",
            fc="white",
        )
        patch_list.append(escaper_pat)
        pats = PatchCollection(
            patch_list, match_original=True
        )  # match_origin prevent PatchCollection mess up original color
        # plot patches
        self._ax.add_collection(pats)
        # plot heading line
        head_tip = self._agent_pose[0:2] + np.array(
            [
                0.25 * np.cos(self._agent_pose[-1]),
                0.25 * np.sin(self._agent_pose[-1]),
            ]
        )
        head_line = np.vstack((self._agent_pose[:2], head_tip))
        self._ax.plot(head_line[:, 0], head_line[:, 1], "k", linewidth=1)
        # plot trajectory
        if self._agent_traj:
            traj_arr = np.array(self._agent_traj)
            self._ax.plot(
                traj_arr[-100:, 0],
                traj_arr[-100:, 1],
                linestyle=":",
                linewidth=0.5,
                color="blueviolet",
            )
        # Set ax
        self._ax.axis(np.array([-10, 10, -10, 10]))
        self._ax.set_xticks(np.arange(-10, 10))
        self._ax.set_yticks(np.arange(-10, 10))
        self._ax.grid(color="grey", linestyle=":", linewidth=0.5)
        # update figure
        plt.pause(1 / self.metadata["render_fps"])
        self._fig.show()

    def close(self):
        self._fig = None
        self._ax = None
        plt.close("all")


# Uncomment following to test env
if __name__ == "__main__":
    env = Breaker(render_mode="rgb_array", continuous=True)
    obs, info = env.reset()
    for i in range(500):
        obs, rew, term, trun, info = env.step(env.action_space.sample())
        print(obs, rew, term, trun, info)
        if term:
            env.reset()
