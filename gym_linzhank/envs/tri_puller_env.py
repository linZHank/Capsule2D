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
        self.max_episode_steps = 200
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(2),
                                          spaces.Discrete(2),
                                          spaces.Discrete(2)))
        self.puller_locations = np.zeros((3,2))
        for i in range(3):
            self.puller_locations[i] = np.array([np.cos(np.pi/2+2*np.pi/3*i), np.sin(np.pi/2+2*np.pi/3*i)])
        self.traj_catcher = []
        # vars
        self.target_coord_pole = np.zeros(2)
        self.target_coord_cartesian = np.zeros(2)
        self.catcher_coord_pole = np.zeros(2)
        self.catcher_coord_cartesian = np.zeros(2)
        # prepare renderer
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.fixed_patches = []
        for i in range(3):
            ppat = Rectangle(xy=self.puller_locations[i]-.05, width=.1, height=.1, fc='gray') # pullers
            self.fixed_patches.append(ppat)
        bbpat = RegularPolygon(xy=(0,0), numVertices=3, radius=1, fill=False) # bounding box
        self.fixed_patches.append(bbpat)
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0
        target_rho = np.random.uniform(0,1)
        while target_rho<=.05:
            target_rho = np.random.uniform(0, 1)
        target_theta = np.random.uniform(-np.pi, np.pi)
        self.target_coord_pole = np.array([target_rho, target_theta])
        self.target_coord_cartesian = np.array([target_rho*np.cos(target_theta), target_rho*np.sin(target_theta)])
        self.catcher_coord_pole = np.zeros(2)
        self.catcher_coord_cartesian = np.zeros(2)
        self.traj_catcher = [self.catcher_coord_cartesian.copy()]
        self.prev_dist = target_rho

        return self.target_coord_pole 
        
    def step(self, action):
        done = False
        info = ''
        # compute catcher's location change
        vec_c2p = self.puller_locations - self.catcher_coord_cartesian
        unit = vec_c2p/np.expand_dims(np.linalg.norm(vec_c2p, axis=1), axis=1)
        deltas = unit*np.expand_dims(np.array(action), axis=1)*0.02
        resolved_delta  = np.sum(deltas, axis=0) # resolved delta
        self.catcher_coord_cartesian += resolved_delta
        if self.catcher_coord_cartesian[0] < np.cos(-5*np.pi/6):
            self.catcher_coord_cartesian[0] = np.cos(-5*np.pi/6)
        elif self.catcher_coord_cartesian[0] > np.cos(-np.pi/6):
            self.catcher_coord_cartesian[0] = np.cos(-np.pi/6)
        if self.catcher_coord_cartesian[1] > 1.:
            self.catcher_coord_cartesian[1] = 1.
        self.catcher_coord_pole = np.array([
            np.linalg.norm(self.catcher_coord_cartesian), 
            np.arctan2(self.catcher_coord_cartesian[1], self.catcher_coord_cartesian[0])
        ])
        vec_c2t = self.target_coord_cartesian - self.catcher_coord_cartesian
        dist = np.linalg.norm(vec_c2t) # catcher-target distance
        ang = np.arctan2(vec_c2t[1], vec_c2t[0]) # catcher-target angle
        self.traj_catcher.append(self.catcher_coord_cartesian.copy())
        # compute reward
        reward = self.prev_dist - dist
        self.prev_dist = dist
        # Check episode done condition
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            done = True
        if dist<=0.02:
            reward = 100
            done = True
            info = '\n!!!!\nTarget Caught\n!!!!\n'

        return np.array([dist, ang]), reward, done, info

    def render(self, mode='human'):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        patch_list = []
        patch_list += self.fixed_patches
        tpat = RegularPolygon(
            xy=(self.target_coord_cartesian[0], self.target_coord_cartesian[1]),
            numVertices=6,
            radius=.04,
            fc='darkorange'
        ) # target
        patch_list.append(tpat)
        cpat = Circle(
            xy=(self.catcher_coord_cartesian[0], self.catcher_coord_cartesian[1]), 
            radius=.03, 
            fc='black'
        ) # catcher
        patch_list.append(cpat)
        pc = PatchCollection(patch_list, match_original=True) # match_origin prevent PatchCollection mess up original color
        # plot patches
        self.ax.add_collection(pc)
        # plot cables
        self.ax.plot(
            [self.catcher_coord_cartesian[0],self.puller_locations[0,0]], 
            [self.catcher_coord_cartesian[1],self.puller_locations[0,1]],
            linewidth=.1,
            color='k'
        )
        self.ax.plot(
            [self.catcher_coord_cartesian[0],self.puller_locations[1,0]], 
            [self.catcher_coord_cartesian[1],self.puller_locations[1,1]],
            linewidth=.1,
            color='k'
        )
        self.ax.plot(
            [self.catcher_coord_cartesian[0],self.puller_locations[2,0]], 
            [self.catcher_coord_cartesian[1],self.puller_locations[2,1]],
            linewidth=.1,
            color='k'
        )
        # plot catcher's trajectory
        if self.traj_catcher:
            traj_c = np.array(self.traj_catcher)
            self.ax.plot(traj_c[-100:,0], traj_c[-100:,1], linestyle=':', linewidth=0.5, color='black')
        # Set ax
        self.ax.axis(np.array([-1.2, 1.2, -1., 1.4]))
        self.ax.grid(color='grey', linestyle=':', linewidth=0.5)
        plt.pause(0.02)
        self.fig.show()


# Uncomment following to test env
# env = TriPullerEnv()
# env.reset()
# for _ in range(env.max_episode_steps):
#     env.render()
#     # o,r,d,i = env.step([1,0,0])
#     o,r,d,i = env.step(np.random.randint(0,2,(3)))
#     print(o)
