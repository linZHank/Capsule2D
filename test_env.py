import gymnasium as gym
import gym_explore


env = gym.make('escaper-v0', render_mode="human", continuous=False)
obs, info = env.reset()
for i in range(1000):
    obs, rew, term, trun, info = env.step(env.action_space.sample())
    print(obs, rew, term, trun, info)
    if term or trun:
        env.reset()
env.close()
