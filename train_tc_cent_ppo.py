import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

from agents.ppo import PPOAgent, PPOBuffer

# instantiate env
env = gym.make('gym_coop:TwoCarrier-v0')
action_codebook = np.array([
    [0,0],
    [0,1],
    [0,2],
    [0,3],
    [1,0],
    [1,1],
    [1,2],
    [1,3],
    [2,0],
    [2,1],
    [2,2],
    [2,3],
    [3,0],
    [3,1],
    [3,2],
    [3,3],
])
# instantiate actor-critic and replay buffer
agent = PPOAgent(dim_obs=(3,), num_act=16)
replay_buffer = PPOBuffer(dim_obs=3, max_size=6000) # max_size is the upper-bound
# paramas
min_steps_per_train = replay_buffer.max_size - env.max_episode_steps
assert min_steps_per_train>0
num_trains = 25
train_epochs = 80
save_freq = 20
# variables
ep_cntr, st_cntr = 0, 0
stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
episodic_steps = []
start_time = time.time()
# main loop
obs, ep_ret, ep_len = env.reset(), 0, 0
for t in range(num_trains):
    for s in range(replay_buffer.max_size):
        act, val, lpa = agent.make_decision(np.expand_dims(obs,0)) 
        nobs, rew, done, _ = env.step(action_codebook[act.numpy()])
        stepwise_rewards.append(rew)
        ep_ret += rew
        ep_len += 1
        st_cntr += 1
        replay_buffer.store(obs, act, rew, val, lpa)
        obs = nobs # SUPER CRITICAL!!!
        if done or ep_len>=env.max_episode_steps:
            val = 0.
            if ep_len>=env.max_episode_steps:
                _, val, _ = agent.make_decision(np.expand_dims(obs,0))
            replay_buffer.finish_path(val)
            # summarize episode
            ep_cntr += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/ep_cntr)
            episodic_steps.append(st_cntr)
            logging.debug("\n----\nEpisode: {}, EpisodeLength: {}, TotalSteps: {}, StepsInLoop: {}, \nEpReturn: {}\n----\n".format(ep_cntr, ep_len, st_cntr, s+1, ep_ret))
            obs, ep_ret, ep_len = env.reset(), 0, 0
            if s+1>=min_steps_per_train:
                break
    # update actor-critic
    data = replay_buffer.get()
    loss_pi, loss_v, loss_info = agent.train(data, train_epochs)
    logging.info("\n====\nTraining: {} \nTotalSteps: {} \nDataSize: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n====\n".format(t+1, st_cntr, data['ret'].shape[0], sedimentary_returns[-1], loss_pi, loss_v, loss_info['kld'], loss_info['entropy'], time.time()-start_time))


# Test trained model
input("press any key to continue")
num_episodes = 10
num_steps = env.max_episode_steps
for ep in range(num_episodes):
    obs, done = env.reset(), False
    for st in range(num_steps):
        env.render()
        act, _, _ = agent.make_decision(np.expand_dims(obs, 0))
        next_obs, rew, done, info = env.step(action_codebook[act.numpy()])
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs
        if done:
            break
