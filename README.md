# gym-linzhank
This project is originated from a course project which was introduced in University of Cincinnati's [AEEM6117](https://www.coursicle.com/uc/courses/AEEM/6117/) in 2021 Spring.

This repo contains a personal collection of simple simulated environments for the study of reinforcement learning.
As opposed to those large-scale environments depending on a complicated multibody dynamics simulation engine, the ones in this repo are simple. 
Only relying on kinematics to move and [Matplotlib](https://matplotlib.org/) to visualized. 


- TriPuller-v0
- TwoCarrier-v0

# Installation
## Pre-requisites
- [Python3](https://www.python.org/)
> The environments were tested under Python 3.6.9, but Python 2 should be OK.
- [pip](https://pypi.org/project/pip/)
- [Git](https://git-scm.com/)

## Installation (Ubuntu)
1. Bring up a terminal console  
2. `cd` *`desired_directory/`* 
3. `git clone https://github.com/IRASatUC/gym-linzhank.git`
4. `cd gym-plot`
5. `pip install -e .`

## Quick Start
### TriPuller-v0
![TriPuller](https://github.com/linZHank/gym-linzhank/blob/main/images/TriPuller.png)
```python
import gym
env = gym.make('gym_plot:TriPuller-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
```

### TwoCarrier-v0
![TwoCarrier](https://github.com/linZHank/gym-linzhank/blob/main/images/TwoCarrier.png)
```python
import gym
env = gym.make('gym_plot:TwoCarrier-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
```
