from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
import gymnasium as gym
from gymnasium import spaces


# TODO:
class EscaperEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None, continuous: bool = False):
        self.prev_reward = None
        self.max_episode_steps = 1000
        self.continuous = continuous
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(3,), dtype=np.float32
        )
        if self.continuous:
            self.action_space = spaces.Box(
                low=np.array([-0.2, -np.pi / 4]),
                high=np.array([0.2, np.pi / 4]),
                shape=(2,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(4)
            self.action_codebook = {
                0: np.array([0.1, np.pi / 8]),  # linear, angular
                1: np.array([0.1, -np.pi / 8]),
                2: np.array([-0.1, np.pi / 8]),
                3: np.array([-0.1, -np.pi / 8]),
            }
        # vars
        self._agent_pose = np.zeros(3, dtype=np.float32)  # x, y, th
        self._agent_traj = []
        self._agent_status = None
        # render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

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
        self.step_counter = 0
        # reset escaper to origin or randomly
        self._agent_pose = np.zeros(3, dtype=np.float32)
        if options == "random":
            x = np.random.uniform(low=-6.5, high=6.5)
            y = np.random.uniform(low=-6.5, high=6.5)
            th = np.random.uniform(low=-np.pi, high=np.pi)
            self._agent_pose = np.array([x, y, th], dtype=np.float32)
        self._agent_traj.append(self._agent_pose[:2].copy())
        # get obs and info
        observation = self._get_obs()
        info = self._get_info()
        # prepare renderer
        if self.render_mode == "human":
            self._fig = plt.figure(figsize=(8, 8))
            self._ax = self._fig.add_subplot(111)
            outer_wall = Circle(xy=(0, 0), radius=8.0, fc="grey", ec=None)
            inner_wall = Circle(xy=(0, 0), radius=7.0, fc="white", ec=None)
            doorway = Wedge(
                center=(0, 0), r=8.0, theta1=85.0, theta2=95, fc="white", ec=None
            )
            self._fixed_patches = [outer_wall, inner_wall, doorway]
            self._render_frame()

        return observation, info

    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return self._render_frame()

    def _render_frame(self):
        self._ax = self._fig.get_axes()[0]
        self._ax.cla()
        patch_list = []
        patch_list += self._fixed_patches
        # add wall patches
        escaper_pat = Circle(
            # xy=(self.position[0], self.position[-1]),
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
        plt.pause(0.02)
        self._fig.show()

    # def step(self, action):
    #     # compute displacement
    #     if self.continuous:
    #         np.clip(
    #             action,
    #             a_min=np.array([-0.2, -np.pi/4]),
    #             a_max=np.array([0.2, np.pi/4]),
    #             out=action,
    #         )
    #         vx = action[0]
    #         vth = action[1]
    #     else:
    #         assert self.action_space.contains(
    #             action
    #         ), f"{action!r} ({type(action)}) invalid "
    #         vx = self.action_codebook[action][0]
    #         vth = self.action_codebook[action][1]
    #     done = False
    #     info = {"status": "trapped"}
    #     reward = 0
    #     # compute new position
    #     prev_pose = self.pose.copy()
    #     dx = vx * np.cos(prev_pose[-1])
    #     dy = vx * np.sin(prev_pose[-1])
    #     dth = vth
    #     self.pose += np.array([dx, dy, dth])
    #     if self.pose[-1] > np.pi:  # orientation within [-pi, pi]
    #         self.pose[-1] -= 2*np.pi
    #     elif self.pose[-1] < -np.pi:
    #         self.pose[-1] += 2*np.pi
    #     # compute reward
    #     reward = (
    #         np.abs(prev_pose[0])
    #         - np.abs(self.pose[0])
    #         + self.pose[1]
    #         - prev_pose[1]
    #     )  # |x0 - x1| + (y1 - y0)
    #     self.trajectory.append(self.pose[:2].copy())
    #     # check crash
    #     if self.fixed_patches[0].contains_point(self.pose[:2], radius=0.1):
    #         if not self.fixed_patches[1].contains_point(self.pose[:2], radius=0.1):
    #             if not self.fixed_patches[2].contains_point(self.pose[:2], radius=0.1):
    #                 done = True
    #                 info = {"status": "crash"}
    #     else:
    #         reward = 100.0
    #         done = True
    #         info = {"status": "escaped"}
    #
    #     return self.pose, reward, done, info
    #
    #
    # def close(self):
    #     plt.close("all")


# Uncomment following to test env
env = EscaperEnv(render_mode="human", continuous=False)
obs, info = env.reset()
# for _ in range(1000):
#     env.render()
#     o, r, d, i = env.step(env.action_space.sample())
#     # o,r,d,i = env.step([1,2])
#     print(o, r, d, i)
#     if d:
#         break
