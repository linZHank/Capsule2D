from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
import gym
from gym import spaces


# TODO:
class EscaperEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, continuous: bool = False):
        self.prev_reward = None
        self.max_episode_steps = 1000
        self.continuous = continuous
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(3,), dtype=np.float32
        )
        if self.continuous:
            # self.action_space = spaces.Box(
            #     low=-0.4, high=0.4, shape=(2,), dtype=np.float32
            # )
            self.action_space = spaces.Box(
                low=np.array([-0.4, -np.pi/3]),
                high=np.array([0.4, np.pi/3]),
                shape=(2,), 
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(4)
            # self.action_codebook = np.array(
            #     [
            #         [0.0, 0.2],  # up
            #         [0.0, -0.2],  # down
            #         [-0.2, 0.0],  # left
            #         [0.2, 0.0],  # right
            #     ]
            # )
            self.action_codebook = {
                0: np.array([0.2, np.pi / 6]),  # forward left
                1: np.array([0.2, -np.pi / 6]),  # forward right
                2: np.array([-0.2, np.pi / 6]),  # backward right
                3: np.array([-0.2, -np.pi / 6]),  # backward left
            }
        # vars
        self.pose = np.zeros(3, dtype=np.float32)  # x, y, th
        # self.position = np.zeros(2)  # x, y
        self.trajectory = []
        # prepare renderer
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        outer_wall = Circle(xy=(0, 0), radius=8.0, fc="grey", ec=None)
        inner_wall = Circle(xy=(0, 0), radius=7.0, fc="white", ec=None)
        doorway = Wedge(
            center=(0, 0), r=8.0, theta1=85.0, theta2=95, fc="white", ec=None
        )
        self.fixed_patches = [outer_wall, inner_wall, doorway]

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
    #
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[str] = None,
        return_info: bool = False,
    ):
        super().reset(seed=seed)
        self.step_counter = 0
        # reset escaper to origin or randomly
        # r = 0.0
        # th = 0.0
        self.pose = np.zeros(3, dtype=np.float32)
        if options == "random":
            # r = np.random.uniform(low=-6.5, high=6.5)
            # th = np.random.uniform(low=-np.pi, high=np.pi)
            x = np.random.uniform(low=-6.5, high=6.5)
            y = np.random.uniform(low=-6.5, high=6.5)
            th = np.random.uniform(low=-np.pi, high=np.pi)
            self.pose = np.array([x, y, th], dtype=np.float32)
        # x = r * np.cos(th)
        # y = r * np.sin(th)
        # self.position = np.array([x, y], dtype=np.float32)  # init position
        # self.trajectory = [self.position.copy()]
        self.trajectory = [self.pose[:2].copy()]
        info = {"status": "reset"}

        # return (self.position, info) if return_info else self.position
        return (self.pose, info) if return_info else self.pose

    def step(self, action):
        # compute displacement
        if self.continuous:
            # ds = np.clip(action, -0.4, 0.4).astype(np.float32)
            vx = action[0]
            vth = action[1]
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "
            # ds = self.action_codebook[action]
            vx = self.action_codebook[action][0]
            vth = self.action_codebook[action][1]
        done = False
        info = {"status": "trapped"}
        reward = 0
        # compute new position
        prev_pose = self.pose.copy()
        dx = vx * np.cos(prev_pose[-1])
        dy = vx * np.sin(prev_pose[-1])
        dth = vth
        # self.position += ds
        self.pose += np.array([dx, dy, dth])
        # compute reward
        # reward = (
        #     np.abs(prev_pos[0])
        #     - np.abs(self.position[0])
        #     + self.position[1]
        #     - prev_pos[1]
        # )
        reward = (
            np.abs(prev_pose[0])
            - np.abs(self.position[0])
            + self.position[1]
            - prev_pose[1]
        )  # |x0 - x1| + (y1 - y0)
        # self.trajectory.append(self.position.copy())
        self.trajectory.append(self.pose[:2].copy())
        # check crash
        if self.fixed_patches[0].contains_point(self.pose[:2], radius=0.1):
            if not self.fixed_patches[1].contains_point(self.pose[:2], radius=0.1):
                if not self.fixed_patches[2].contains_point(self.pose[:2], radius=0.1):
                    done = True
                    info = {"status": "crash"}
        else:
            reward = 100.0
            done = True
            info = {"status": "escaped"}
        # for pat in self.fixed_patches:
        #     if np.sum(pat.contains_point(self.position, radius=0.001)):
        #         done = True
        #         info = {"status": "crash wall"}
        #         break
        # check escape
        # if self.position[1] > 5.5:
        #     reward = 100.0
        #     done = True
        #     info = {"status": "escaped"}

        return self.position, reward, done, info

    def render(self, mode="human"):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        patch_list = []
        patch_list += self.fixed_patches
        # add wall patches
        escaper_pat = Circle(
            # xy=(self.position[0], self.position[-1]),
            xy=(self.pose[0], self.pose[1]),
            radius=0.2,
            ec="blueviolet",
            fc="white",
        )
        patch_list.append(escaper_pat)
        pats = PatchCollection(
            patch_list, match_original=True
        )  # match_origin prevent PatchCollection mess up original color
        # plot patches
        self.ax.add_collection(pats)
        # plot trajectory
        if self.trajectory:
            traj_arr = np.array(self.trajectory)
            self.ax.plot(
                traj_arr[-100:, 0],
                traj_arr[-100:, 1],
                linestyle=":",
                linewidth=0.5,
                color="blueviolet",
            )
        # Set ax
        self.ax.axis(np.array([-10, 10, -10, 10]))
        self.ax.set_xticks(np.arange(-10, 10))
        self.ax.set_yticks(np.arange(-10, 10))
        self.ax.grid(color="grey", linestyle=":", linewidth=0.5)
        plt.pause(0.02)
        self.fig.show()

    def close(self):
        plt.close("all")


# Uncomment following to test env
# env = EscaperEnv(continuous=False)
# env.reset()
# for _ in range(1000):
#     env.render()
#     o, r, d, i = env.step(env.action_space.sample())
#     # o,r,d,i = env.step([1,2])
#     print(o, r, d, i)
#     if d:
#         break
