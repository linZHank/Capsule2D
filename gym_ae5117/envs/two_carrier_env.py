import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, RegularPolygon, Circle
from matplotlib.collections import PatchCollection
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TwoCarrierEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.viewer = None
        self.prev_reward = None

        self.observation_space = spaces.Box(low=-10., high=10., shape=(3,), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4))) # ^v<>
        self.action_codebook = np.array([
            [0., .02],
            [0., -.02],
            [-.02, 0.],
            [.02, 0.]
        ])
        # vars
        self.rod_pose = np.zeros(3)
        self.c0_position = np.array([
            self.rod_pose[0]+.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]+.5*np.sin(self.rod_pose[-1])
        ])
        self.c1_position = np.array([
            self.rod_pose[0]-.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]-.5*np.sin(self.rod_pose[-1])
        ])
        self.max_episode_steps = 200
        # prepare render
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0
        # init rod coordinations
        x = np.random.uniform(-3.9, 3.9)
        y = .2
        theta = 0.
        self.rod_pose = np.array([x, y, theta])
        self.c0_position = np.array([
            self.rod_pose[0]+.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]+.5*np.sin(self.rod_pose[-1])
        ])
        self.c1_position = np.array([
            self.rod_pose[0]-.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]-.5*np.sin(self.rod_pose[-1])
        ])

        return self.rod_pose 
        
    def step(self, action):
        done = False
        info = ''
        reward = 0
        # compute rod's displacement and rotation
        disp = self.action_codebook[action[0]] + self.action_codebook[action[1]]
        rot = 0.
        if disp[0]==0:
            rot += -np.arctan2(self.action_codebook[action[0]][0]*np.sin(self.rod_pose[-1]), .5)
        if disp[1]==0:
            rot += np.arctan2(self.action_codebook[action[0]][1]*np.cos(self.rod_pose[-1]), .5)
        deltas = np.append(disp, rot)
        self.rod_pose += deltas
        self.c0_position = np.array([
            self.rod_pose[0]+.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]+.5*np.sin(self.rod_pose[-1])
        ])
        self.c1_position = np.array([
            self.rod_pose[0]-.5*np.cos(self.rod_pose[-1]), 
            self.rod_pose[1]-.5*np.sin(self.rod_pose[-1])
        ])

        return self.rod_pose, reward, done, info

    def render(self, mode='human'):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        patch_list = []
        # patches
        bbpat = Rectangle(xy=(-5,0), width=10, height=6.5, fill=False, ec='black') # bounding box
        patch_list.append(bbpat)
        lwpat = Rectangle(xy=(-5,5), width=4.7, height=.5, fc='gray')
        patch_list.append(lwpat)
        rwpat = Rectangle(xy=(.3,5), width=4.7, height=.5, fc='gray')
        patch_list.append(rwpat)
        c0pat = Circle(
            xy=(self.c0_position[0], self.c0_position[-1]), 
            radius=.1, 
            ec='black',
            fc='white'
        )
        patch_list.append(c0pat)
        c1pat = Circle(
            xy=(self.c1_position[0], self.c1_position[-1]), 
            radius=.1, 
            fc='black'
        )
        patch_list.append(c1pat)
        pc = PatchCollection(patch_list, match_original=True) # match_origin prevent PatchCollection mess up original color
        # plot patches
        self.ax.add_collection(pc)
        # plot rod
        self.ax.plot(
            [self.c0_position[0], self.c1_position[0]],
            [self.c0_position[1], self.c1_position[1]],
            color='darkorange'
        )
        # plot trajectory
        # Set ax
        self.ax.axis(np.array([-5.1, 5.1, -.1, 6.6]))
        plt.pause(0.02)
        self.fig.show()


# Uncomment following to test env
env = TwoCarrierEnv()
env.reset()
for _ in range(2000):
    env.render()
    o,r,d,i = env.step(np.random.randint(0,4,(2)))
    print(o)
