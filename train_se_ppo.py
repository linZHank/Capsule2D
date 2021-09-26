import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

from agents.ppo import PPOAgent, PPOBuffer

env = gym.make("gym_linzhank:SoloEscape-v0")
agent = PPOAgent(dim_obs=2, num_act=4, target_kld=0.02)
rb = PPOBuffer(dim_obs=2, max_size=6000)  # max_size is the upper-bound

# paramas
min_steps_per_train = rb.max_size - env.max_episode_steps
assert min_steps_per_train > 0
num_trains = 100
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
    for s in range(rb.max_size):
        act, val, lpa = agent.make_decision(np.expand_dims(obs, 0))
        nobs, rew, done, _ = env.step(act.numpy())
        stepwise_rewards.append(rew)
        ep_ret += rew
        ep_len += 1
        st_cntr += 1
        rb.store(obs, act, rew, val, lpa)
        obs = nobs  # SUPER CRITICAL!!!
        if done or ep_len >= env.max_episode_steps:
            val = 0.0
            if ep_len >= env.max_episode_steps:
                _, val, _ = agent.make_decision(np.expand_dims(obs, 0))
            rb.finish_path(val)
            # summarize episode
            ep_cntr += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns) / ep_cntr)
            episodic_steps.append(st_cntr)
            logging.debug(
                "\n----\nEpisode: {}, \
                EpisodeLength: {}, \
                TotalSteps: {}, \
                StepsInLoop: {}, \
                \nEpReturn: {}\n----\n".format(
                    ep_cntr, ep_len, st_cntr, s + 1, ep_ret
                )
            )
            obs, ep_ret, ep_len = env.reset(), 0, 0
            if s + 1 >= min_steps_per_train:
                break
    # update actor-critic
    data = rb.get()
    loss_pi, loss_v, loss_info = agent.train(data, train_epochs)
    logging.info(
        "\n====\nTraining: {} \
        \nTotalSteps: {} \
        \nTotalEpisodes: {} \
        \nDataSize: {} \
        \nAveReturn: {} \
        \nLossPi: {} \
        \nLossV: {} \
        \nKLDivergence: {} \
        \nEntropy: {} \
        \nTimeElapsed: {}\n====\n".format(
            t + 1,
            st_cntr,
            ep_cntr,
            data["ret"].shape[0],
            sedimentary_returns[-1],
            loss_pi,
            loss_v,
            loss_info["kld"],
            loss_info["entropy"],
            time.time() - start_time,
        )
    )


# Test trained model
input("press any key to continue")
num_episodes = 10
num_steps = env.max_episode_steps
for ep in range(num_episodes):
    obs, done = env.reset(), False
    for st in range(num_steps):
        env.render()
        act, _, _ = agent.make_decision(np.expand_dims(obs, 0))
        next_obs, rew, done, info = env.step(act.numpy())
        # print("\n-\nepisode: {}, step: {} \naction: {} \nobs: {}, \nreward: {}".format(ep+1, st+1, act, obs, rew))
        obs = next_obs
        if done:
            break
