import gymnasium as gym
import gym_explore


env = gym.make('escaper-v0', render_mode="human", continuous=False)
# env.reset()
# for _ in range(1000):
#     env.render()
#     o, r, d, i = env.step(env.action_space.sample())
#     # o,r,d,i = env.step([1,2])
#     print(o, r, d, i)
#     if d:
#         break
obs, info = env.reset()
for i in range(1000):
    obs, rew, term, trun, info = env.step(env.action_space.sample())
    print(obs, rew, term, trun, info)
    if term:
        env.reset()
env.close()
