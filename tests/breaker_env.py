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
