import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

from agents.dqn import DQNAgent, DQNBuffer

env = gym.make('gym_coop:TwoCarrier-v0')
agent0 = DQNAgent(dim_obs=3, num_act=4)
agent1 = DQNAgent(dim_obs=3, num_act=4)
rb0 = DQNBuffer(dim_obs=3, size=int(2e5)) # max_size is the upper-bound
rb1 = DQNBuffer(dim_obs=3, size=int(2e5)) # max_size is the upper-bound

# paramas
batch_size = 500
update_freq = 100
update_after = 1000
decay_period = 1500
warmup = 50
total_steps = int(1e6)
episodic_returns = []
sedimentary_returns = []
episodic_steps = []
save_freq = 100
episode_counter = 0
model_dir = './models/dqn/'+env.spec.id
start_time = time.time()
obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
for t in range(total_steps):
    # env.render()
    act0 = np.squeeze(agent0.make_decision(obs.reshape(1,-1)))
    act1 = np.squeeze(agent0.make_decision(obs.reshape(1,-1)))
    nobs, rew, done, _ = env.step([act0, act1])
    ep_ret += rew
    ep_len += 1
    done = False if ep_len == env.max_episode_steps else done
    rb0.store(obs, act0, rew, done, nobs)
    rb1.store(obs, act1, rew, done, nobs)
    obs = nobs
    if done or (ep_len==env.max_episode_steps):
        episode_counter += 1
        episodic_returns.append(ep_ret)
        sedimentary_returns.append(sum(episodic_returns)/episode_counter)
        episodic_steps.append(t+1)
        print("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpsilon: {} \nEpisodeReturn: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(episode_counter, ep_len, t+1, agent0.epsilon, ep_ret, sedimentary_returns[-1], time.time()-start_time))
        # save model
        # if not episode_counter%save_freq:
        #     model_path = os.path.join(model_dir, str(episode_counter))
        #     if not os.path.exists(os.path.dirname(model_path)):
        #         os.makedirs(os.path.dirname(model_path))
        #     dqn.q.q_net.save(model_path)
        # reset env
        obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
        agent0.linear_epsilon_decay(episode=episode_counter, decay_period=decay_period, warmup_episodes=warmup)
        agent1.linear_epsilon_decay(episode=episode_counter, decay_period=decay_period, warmup_episodes=warmup)
    if not t%update_freq and t>=update_after:
        for _ in range(update_freq):
            minibatch0 = rb0.sample_batch(batch_size=batch_size)
            minibatch1 = rb1.sample_batch(batch_size=batch_size)
            loss_q0 = agent0.train_one_batch(data=minibatch0)
            loss_q1 = agent1.train_one_batch(data=minibatch1)
            logging.debug("\nloss_q: {}".format(loss_q0, loss_q1))

# Save returns 
# np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
# np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
# np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
# with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
#     f.write("{}".format(time.time()-start_time))
# Save final model
# model_path = os.path.join(model_dir, str(episode_counter))
# dqn.q.q_net.save(model_path)
# plot returns
# fig, ax = plt.subplots(figsize=(8, 6))
# fig.suptitle('Averaged Returns')
# ax.plot(sedimentary_returns)
# plt.show()

# Test
input("Press ENTER to test lander...")
agent0.epsilon = 0.
agent1.epsilon = 0.
for ep in range(10):
    o, d, ep_ret = env.reset(), False, 0
    for st in range(env.max_episode_steps):
        env.render()
        a0 = np.squeeze(agent0.make_decision(o.reshape(1,-1)))
        a1 = np.squeeze(agent1.make_decision(o.reshape(1,-1)))
        o2,r,d,_ = env.step([a0, a1])
        ep_ret += r
        o = o2
        if d:
            print("EpReturn: {}".format(ep_ret))
            break 


