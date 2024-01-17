from gymnasium.envs.registration import register

register(
    id='Escaper-v0',
    entry_point='gym_explore.envs:EscaperEnv',
    max_episode_steps=1000,
)

