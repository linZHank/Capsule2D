# gym-explore

This project is originated from a course project which was introduced in
University of Cincinnati's
[AEEM6117](https://www.coursicle.com/uc/courses/AEEM/6117/) in 2021 Spring.

Objective of this project is to investigate RL agents' exploration strategies.
This repo contains a personal collection of 
[OpenAI Gym](https://github.com/openai/gym) like environment rendered with 
[Matplotlib](https://matplotlib.org/) only.

## Installation (Linux & MacOS)
```shell
git clone https://github.com/linzhank/gym-explore.git
pip install -e gym-explore
```

# Usage

```python
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
```
