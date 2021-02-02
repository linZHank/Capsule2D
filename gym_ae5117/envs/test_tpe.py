import numpy as np
from tri_puller_env import TriPullerEnv

env = TriPullerEnv()
env.reset()
for _ in range(200):
    env.render()
    o,r,d,i = env.step([0,0,1])
    # o,r,d,i = env.step(np.random.randint(0,2,(3)))
    print(o)
    
