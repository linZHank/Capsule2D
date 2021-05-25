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
        self.max_episode_steps = 500
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
        self.c0_traj = []
        self.c1_traj = []
        # prepare renderer
        self.fig = plt.figure(figsize=(12,8))
        self.ax = self.fig.add_subplot(111)
        nwwpat = Rectangle(xy=(-5.5,5), width=5.1, height=.5, fc='gray')
        newpat = Rectangle(xy=(.4,5), width=5.1, height=.5, fc='gray')
        wwpat = Rectangle(xy=(-5.5,-.5), width=.5, height=6, fc='gray')
        ewpat = Rectangle(xy=(5,-.5), width=.5, height=6, fc='gray')
        swpat = Rectangle(xy=(-5.5,-.5), width=11, height=.5, fc='gray')
        self.fixed_patches = [nwwpat, newpat, wwpat, ewpat, swpat]
            
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
        self.c0_traj = [self.c0_position.copy()]
        self.c1_traj = [self.c1_position.copy()]

        return self.rod_pose 
        
    def step(self, action):
        done = False
        info = ''
        reward = 0
        prev_rod = self.rod_pose.copy()
        prev_c0 = self.c0_position.copy()
        prev_c1 = self.c1_position.copy()
        # compute rod's displacement and rotation
        disp = self.action_codebook[action[0]] + self.action_codebook[action[1]]
        rot = 0.
        rot += -np.arctan2(self.action_codebook[action[0]][0]*np.sin(self.rod_pose[-1]), .5) + \
            np.arctan2(self.action_codebook[action[0]][1]*np.cos(self.rod_pose[-1]), .5) + \
            np.arctan2(self.action_codebook[action[1]][0]*np.sin(self.rod_pose[-1]), .5) - \
            np.arctan2(self.action_codebook[action[1]][1]*np.cos(self.rod_pose[-1]), .5)
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
        self.c0_traj.append(self.c0_position.copy())
        self.c1_traj.append(self.c1_position.copy())
        # restrict angle in (-pi,pi)
        if np.pi<self.rod_pose[-1]<=2*np.pi:
            self.rod_pose[-1] -= 2*np.pi 
        elif -np.pi>self.rod_pose[-1]>=-2*np.pi:
            self.rod_pose[-1] += 2*np.pi
        # compute reward
        # uvec_vert = np.array([0., 1.]) # unit vertical vector
        # uvec_prod = (prev_c0-prev_c1)/np.linalg.norm(prev_c0-prev_c1) # unit vector of previous rod
        # uvec_rod = (self.c0_position-self.c1_position)/np.linalg.norm(self.c0_position-self.c1_position) # unit vector of current rod
        # prev_vertang = np.arccos(np.dot(uvec_vert, uvec_prod)) # angle between previous rod and vertical vector
        # if prev_vertang>np.pi/2:
        #     prev_vertang = np.pi-prev_vertang # restrict angle to (0, pi/2)
        # vertang = np.arccos(np.dot(uvec_vert, uvec_rod)) # angle between current rod and vertical vector
        # if vertang>np.pi/2:
        #     vertang = np.pi-vertang
        # reward = np.abs(prev_rod[0])-np.abs(self.rod_pose[0]) + \
        #     self.rod_pose[1]-prev_rod[1] + \
        #     prev_vertang-vertang 
        reward = np.abs(prev_c0[0])-np.abs(self.c0_position[0]) + np.abs(prev_c1[0])-np.abs(self.c1_position[0]) + \
            (self.c0_position[1]-prev_c0[1] + self.c1_position[1]-prev_c1[1])
        
        # check crash
        rod_points = np.linspace(self.c0_position, self.c1_position, 50)
        for p in self.fixed_patches:
            if np.sum(p.contains_points(rod_points, radius=.001)):
                done = True
                info = 'crash wall'
                break
        # check escape
        if self.c0_position[1]>5.5 and self.c1_position[1]>5.5:
            reward = 100.
            done = True
            info = 'escaped'

        return self.rod_pose, reward, done, info

    def render(self, mode='human'):
        self.ax = self.fig.get_axes()[0]
        self.ax.cla()
        patch_list = []
        patch_list += self.fixed_patches
        # add wall patches
        c0pat = Circle(
            xy=(self.c0_position[0], self.c0_position[-1]), 
            radius=.05, 
            ec='black',
            fc='white'
        )
        patch_list.append(c0pat)
        c1pat = Circle(
            xy=(self.c1_position[0], self.c1_position[-1]), 
            radius=.05, 
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
        if self.c0_traj and self.c0_traj:
            traj_c0 = np.array(self.c0_traj)
            traj_c1 = np.array(self.c1_traj)
            self.ax.plot(traj_c0[-100:,0], traj_c0[-100:,1], linestyle=':', linewidth=0.5, color='black')
            self.ax.plot(traj_c1[-100:,0], traj_c1[-100:,1], linestyle=':', linewidth=0.5, color='black')
        # Set ax
        self.ax.axis(np.array([-6, 6, -1, 7]))
        self.ax.set_xticks(np.arange(-6, 7))
        self.ax.set_yticks(np.arange(-1, 8))
        self.ax.grid(color='grey', linestyle=':', linewidth=0.5)
        plt.pause(0.02)
        self.fig.show()


# Uncomment following to test env
# env = TwoCarrierEnv()
# env.reset()
# for _ in range(20):
#     o,r,d,i = env.step([0,0])
# for _ in range(env.max_episode_steps):
#     env.render()
#     # o,r,d,i = env.step(np.random.randint(0,4,(2)))
#     o,r,d,i = env.step([1,2])
#     print(o, r, d, i)
#     if d:
#         break
