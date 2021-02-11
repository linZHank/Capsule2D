# gym-ae5117
This repo provides two training environment for **AE5117: Intelligent Robotics**'s course project. 
- TriPuller-v0
- TwoCarrier-v0

# Installation
## Pre-requisites
- [Python3](https://www.python.org/)
> The environments were tested under Python 3.6.9, but Python 2 should be OK.
- [pip](https://pypi.org/project/pip/)
- [Git](https://git-scm.com/)
- For **Ubuntu** or **RaspberryPi** users, bring up a terminal and run
```console
sudo apt install python3-dev python3-pip git
```
## Installation
1. Bring up a terminal console  
2. `cd` *`desired_directory/`* 
3. `git clone https://github.com/IRASatUC/gym-ae5117.git`
4. `cd gym-ae5117`
5. `pip install -e .`
> May update installation guide for Windows and Mac. 

## Quick Start
### TriPuller-v0
![TriPuller](https://github.com/IRASatUC/gym-ae5117/blob/main/images/TriPuller.png)
```python
import gym
env = gym.make('gym_ae5117:TriPuller-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
```

### TwoCarrier-v0
![TwoCarrier](https://github.com/IRASatUC/gym-ae5117/blob/main/images/TwoCarrier.png)
```python
import gym
env = gym.make('gym_ae5117:TwoCarrier-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
```
