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

        self.fixed_patches = []
        for i in range(3):
            bot_patch = Rectangle(xy=(np.cos(np.pi/2+2*np.pi/3*i)-.05, np.sin(np.pi/2+2*np.pi/3*i)-.05),
                                  width=.1,
                                  height=.1)
            self.fixed_patches.append(bot_patch)
        bound_patch = RegularPolygon(xy=(0,0),
                                     numVertices=3,
                                     radius=1,
                                     fill=False)
        self.fixed_patches.append(bound_patch)
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0
        target_rho = np.random.uniform(0,1)
        while target_rho<=.05:
            target_rho = np.random.uniform(0, 1)
        target_theta = np.random.uniform(-np.pi, np.pi)
        state = np.array([target_rho, target_theta])
        self.target_patch = RegularPolygon(xy=(target_rho*np.cos(target_theta), target_rho*np.sin(target_theta)),
                                     numVertices=6,
                                     radius=.03,
                                     fc='salmon')
        self.catcher_patch = Circle(xy=(0,0),
                                     radius=.03,
                                     fc='black')

        return state
 
        

    def render(self, mode='human'):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        pc = PatchCollection(self.fixed_patches+[self.target_patch]+[self.catcher_patch], match_original=True) # match_origin prevent PatchCollection mess up original color
        self.ax.add_collection(pc)
        # Set ax
        self.ax.axis(np.array([-1.2, 1.2, -1., 1.4]))
        plt.pause(0.02)
        self.fig.show()


# def step(self, action):
# def reset(self):
# def render(self, mode='human'):
# def close(self):
