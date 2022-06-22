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
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        if self.continuous:
            self.action_space = spaces.Box(
                low=-0.4, high=0.4, shape=(2,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(4)
            self.action_codebook = {
                0: np.array([0.0, 0.2]),  # up
                1: np.array([0.0, -0.2]),  # down
                2: np.array([-0.2, 0.0]),  # left
                3: np.array([0.2, 0]),  # right
            }
            # self.action_codebook = np.array(
            #     [
            #         [0.0, 0.2],  # up
            #         [0.0, -0.2],  # down
            #         [-0.2, 0.0],  # left
            #         [0.2, 0.0],  # right
            #     ]
            # )
        # vars
        self.position = np.zeros(2)
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
        r = 0.
        th = 0.
        if options == "random":
            r = np.random.uniform(low=-6.5, high=6.5)
            th = np.random.uniform(low=-np.pi, high=np.pi)
        x = r * np.cos(th)
        y = r * np.sin(th)
        self.position = np.array([x, y], dtype=np.float32)  # escaper init position
        print(f'obs: {self.position}')
        self.trajectory = [self.position.copy()]
        info = {"status": "reset"}

        return (self.position, info) if return_info else self.position

    def step(self, action):
        # compute displacement
        if self.continuous:
            ds = np.clip(action, -0.4, 0.4).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "
            ds = self.action_codebook[action]
        done = False
        info = {"status": "trapped"}
        reward = 0
        # compute new position
        prev_pos = self.position.copy()
        self.position += ds
        # compute reward
        reward = (
            np.abs(prev_pos[0])
            - np.abs(self.position[0])
            + self.position[1]
            - prev_pos[1]
        )
        self.trajectory.append(self.position.copy())
        # check crash
        if self.fixed_patches[0].contains_point(self.position, radius=0.1):
            if not self.fixed_patches[1].contains_point(self.position, radius=0.1):
                if not self.fixed_patches[2].contains_point(self.position, radius=0.1):
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
            xy=(self.position[0], self.position[-1]),
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
