"""
Train SoloEscape using DQN
"""
import gym

from stable_baselines3 import DQN

env = gym.make("gym_linzhank:SoloEscape-v0")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(2e5), log_interval=4)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
