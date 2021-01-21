import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TriPullerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.viewer = None
        self.prev_reward = None

        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(2),
                                          spaces.Discrete(2),
                                          spaces.Discrete(2)))

        self.bot_patches = []
        for i in range(3):
            bot_patch = Rectangle(xy=(np.cos(np.pi/2+2*np.pi/3*i)-.05, np.sin(np.pi/2+2*np.pi/3*i)-.05),
                                  width=.1,
                                  height=.1)
            self.bot_patches.append(bot_patch)
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        pass

    def render(self, mode='human'):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        pc = PatchCollection(self.bot_patches, match_original=True) # match_origin prevent PatchCollection mess up original color
        self.ax.add_collection(pc)
        # Set ax
        self.ax.axis(np.array([-1.1, 1.1, -.75, 1.1]))
        plt.pause(0.02)
        self.fig.show()


# def step(self, action):
# def reset(self):
# def render(self, mode='human'):
# def close(self):
