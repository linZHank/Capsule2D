# gym-coop
This repo provides two training environment: 
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
3. `git clone https://github.com/IRASatUC/gym-coop.git`
4. `cd gym-coop`
5. `pip install -e .`

## Quick Start
### TriPuller-v0
![TriPuller](https://github.com/IRASatUC/gym-coop/blob/main/images/TriPuller.png)
```python
import gym
env = gym.make('gym_coop:TriPuller-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
```

### TwoCarrier-v0
![TwoCarrier](https://github.com/IRASatUC/gym-coop/blob/main/images/TwoCarrier.png)
```python
import gym
env = gym.make('gym_coop:TwoCarrier-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
```
