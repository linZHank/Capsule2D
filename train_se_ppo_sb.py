"""
Train SoloEscaper using PPO
"""
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("gym_linzhank:SoloEscaper-v0", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e5))

env = gym.make("gym_linzhank:SoloEscaper-v0")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
