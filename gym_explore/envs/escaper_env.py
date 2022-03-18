import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
import gym
from gym import error, spaces, utils
from gym.utils import seeding


# TODO: create openning
class EscaperEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.seed()
        self.viewer = None
        self.prev_reward = None
        self.max_episode_steps = 1000
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-0.5 , high=0.5 , shape=(2,) , dtype=np.float32)
        # vars
        self.position = np.zeros(2)
        self.trajectory = []
        # prepare renderer
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        outer_wall = Circle(xy=(0, 0), radius=8., fc='black', ec='black')
        inner_wall = Circle(xy=(0, 0), radius=7., fc='white', ec='black')
        self.fixed_patches = [outer_wall, inner_wall]

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]
    #
    # def reset(self):
    #     self.step_counter = 0
    #     # init rod coordinations
    #     x = np.random.uniform(-4.5, 4.5)
    #     y = np.random.uniform(.5, 4.5)
    #     self.position = np.array([x, y])
    #     self.trajectory = [self.position.copy()]
    #
    #     return self.position
    #
    # # TODO: upgrade to continuous action
    # def step(self, action):
    #     done = False
    #     info = {}
    #     reward = 0
    #     # compute displacement
    #     prev_pos = self.position.copy()
    #     self.position += self.action_codebook[action]
    #     # compute reward
    #     reward = (
    #         np.abs(prev_pos[0])
    #         - np.abs(self.position[0])
    #         + self.position[1]
    #         - prev_pos[1]
    #     )
    #     self.trajectory.append(self.position.copy())
    #     # check crash
    #     for pat in self.fixed_patches:
    #         if np.sum(pat.contains_point(self.position, radius=0.001)):
    #             done = True
    #             info = {"status": "crash wall"}
    #             break
    #     # check escape
    #     if self.position[1] > 5.5:
    #         reward = 100.0
    #         done = True
    #         info = {"status": "escaped"}
    #
    #     return self.position, reward, done, info

    def render(self, mode="human"):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        patch_list = []
        patch_list += self.fixed_patches
        # add wall patches
        escaper_pat = Circle(
            xy=(self.position[0], self.position[-1]),
            radius=0.1,
            ec="black",
            fc="white",
        )
        patch_list.append(escaper_pat)
        pats = PatchCollection(
            patch_list, match_original=True
        )  # match_origin prevent PatchCollection mess up original color
        # plot patches
        self.ax.add_collection(pats)
        # plot trajectory
        # if self.trajectory:
        #     traj_arr = np.array(self.trajectory)
        #     self.ax.plot(
        #         traj_arr[-100:, 0],
        #         traj_arr[-100:, 1],
        #         linestyle=":",
        #         linewidth=0.5,
        #         color="black",
        #     )
        # Set ax
        self.ax.axis(np.array([-10, 10, -10, 10]))
        self.ax.set_xticks(np.arange(-10, 10))
        self.ax.set_yticks(np.arange(-10, 10))
        self.ax.grid(color="grey", linestyle=":", linewidth=0.5)
        plt.pause(0.02)
        self.fig.show()


# Uncomment following to test env
env = EscaperEnv()
#  for _ in range(20):
   #  env.reset()
   #  for _ in range(100):
       #  env.render()
       #  o,r,d,i = env.step(env.action_space.sample())
       #  # o,r,d,i = env.step([1,2])
       #  print(o, r, d, i)
       #  if d:
           #  break
