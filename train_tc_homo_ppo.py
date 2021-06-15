import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

from agents.ppo import PPOAgent, PPOBuffer

env = gym.make('gym_coop:TwoCarrier-v0')
agent0 = PPOAgent(dim_obs=3, num_act=4, target_kld=0.2)
agent1 = PPOAgent(dim_obs=3, num_act=4, target_kld=0.2)
rb0 = PPOBuffer(dim_obs=3, max_size=6000) # max_size is the upper-bound
rb1 = PPOBuffer(dim_obs=3, max_size=6000) # max_size is the upper-bound

# paramas
min_steps_per_train = rb0.max_size - env.max_episode_steps
assert min_steps_per_train>0
num_trains = 200
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
    for s in range(rb0.max_size):
        act0, val0, lpa0 = agent0.make_decision(np.expand_dims(obs,0)) 
        act1, val1, lpa1 = agent1.make_decision(np.expand_dims(obs,0)) 
        nobs, rew, done, _ = env.step([act0.numpy(), act1.numpy()])
        stepwise_rewards.append(rew)
        ep_ret += rew
        ep_len += 1
        st_cntr += 1
        rb0.store(obs, act0, rew, val0, lpa0)
        rb1.store(obs, act0, rew, val1, lpa1)
        obs = nobs # SUPER CRITICAL!!!
        if done or ep_len>=env.max_episode_steps:
            val0, val1 = 0., 0.
            if ep_len>=env.max_episode_steps:
                _, val0, _ = agent0.make_decision(np.expand_dims(obs,0))
                _, val1, _ = agent1.make_decision(np.expand_dims(obs,0))
            rb0.finish_path(val0)
            rb1.finish_path(val1)
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
    data0 = rb0.get()
    data1 = rb1.get()
    loss_pi0, loss_v0, loss_info0 = agent0.train(data0, train_epochs)
    loss_pi1, loss_v1, loss_info1 = agent1.train(data1, train_epochs)
    logging.info("\n====\nTraining: {} \nTotalSteps: {} \nTotalEpisodes: {} \nDataSize: {} \nAveReturn: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n====\n".format(t+1, st_cntr, ep_cntr, (data0['ret'].shape[0], data1['ret'].shape[0]), sedimentary_returns[-1], (loss_pi0, loss_pi1), (loss_v0, loss_v1), (loss_info0['kld'], loss_info1['kld']), (loss_info0['entropy'], loss_info1['entropy']), time.time()-start_time))


# Test trained model
input("press any key to continue")
num_episodes = 10
num_steps = env.max_episode_steps
for ep in range(num_episodes):
    obs, done = env.reset(), False
    for st in range(num_steps):
        env.render()
        act0, _, _ = agent0.make_decision(np.expand_dims(obs, 0))
        act1, _, _ = agent1.make_decision(np.expand_dims(obs, 0))
        next_obs, rew, done, info = env.step([act0.numpy(), act1.numpy()])
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs
        if done:
            break
