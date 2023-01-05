# gym-explore

This project is originated from a course project which was introduced in
University of Cincinnati's
[AEEM6117](https://www.coursicle.com/uc/courses/AEEM/6117/) in 2021 Spring.

Objective of this project is to investigate RL agents' exploration strategies.
This repo contains a personal collection of 
[OpenAI Gym](https://github.com/openai/gym) like environemnt rendered with 
[Matplotlib](https://matplotlib.org/) only.

- Escaper-v0

# Installation

## Pre-requisites

- [Python3](https://www.python.org/)

> The environments were tested in Python 3.10.8 Other versions of Python may work as well.

- [pip](https://pypi.org/project/pip/)
- [Git](https://git-scm.com/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

> The environment were tested with gymnasium 0.27

## Installation (Linux & MacOS)

```shell
git clone https://github.com/linzhank/gym-explore.git
pip install -e gym-explore
```

# Usage

```python
import gym
from gym.envs.registration import register
register(
    id='escaper-v0',
    entry_point='gym_explore.envs:EscaperEnv',
)

env = gym.make('escaper-v0', continuous=True)
observation, info = env.reset(return_info=True)
for _ in range(100):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        observation, info = env.reset(return_info=True)

env.close()
```

## TODO:
- [] Fix displacement clip (under `continuous` setting) 
- [] Add time limit and truncated condition
