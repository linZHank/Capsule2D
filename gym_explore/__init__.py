from gymnasium.envs.registration import register

register(
    id='escaper-v0',
    entry_point='gym_explore.envs:EscaperEnv',
    max_episode_steps=1000,
)

