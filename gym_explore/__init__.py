from gym.envs.registration import register

register(
    id='escaper-v0',
    entry_point='gym_explore.envs:EscaperEnv',
)

