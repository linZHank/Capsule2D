# Capsule2D

An implementation of [gymnasium](https://github.com/Farama-Foundation/Gymnasium/) environments for control and reinforcement learning.

__Feature__: Rendered with [Matplotlib](https://matplotlib.org/).

## Installation (Linux & MacOS)

```shell
git clone https://github.com/linzhank/Capsule2D.git
cd gym-explore
uv pip install -e .
```

## Usage

```python
import gymnasium as gym
import capsule2d

env = gym.make("CapsuleBreaker-v0", render_mode="human", continuous=True)
obs, info = env.reset()
for i in range(4 * env.spec.max_episode_steps):
    obs, rew, term, trun, info = env.step(env.action_space.sample())
    print(obs, rew, term, trun, info)
    if term or trun:
        env.reset(options="random")
env.close()
```

## About

> - This project is originated from a course project which was introduced in
University of Cincinnati's [AEEM6117](https://www.coursicle.com/uc/courses/AEEM/6117/) in 2021 Spring.
> - So far, only `CapsuleBreaker-v0` environment is available.
