import gymnasium as gym
import capsule2d

env = gym.make("CapsuleBreaker-v0", render_mode="human", continuous=True)
last_obs, info = env.reset()
for i in range(4 * env.spec.max_episode_steps):  # test 4 espisodes
    act = env.action_space.sample()
    next_obs, rew, term, trunc, info = env.step(act)
    # Step statistics
    print("\n")
    print(f"last observation: {last_obs}")
    print(f"action: {act}")
    print(f"next observation: {next_obs}")
    print(f"reward: {rew}")
    print(f"episode terminated: {term}")
    print(f"episode truncated: {trunc}")
    print(f"info: {info}")
    print("\n")
    last_obs = next_obs.copy()
    if term or trunc:
        env.reset(options="random")
env.close()
