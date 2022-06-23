import gym
import gym_explore

env = gym.make('escaper-v0', continuous=False)
env.reset()
for _ in range(1000):
    env.render()
    o, r, d, i = env.step(env.action_space.sample())
    # o,r,d,i = env.step([1,2])
    print(o, r, d, i)
    if d:
        break
