from gym.envs.registration import register

register(
    id='TriPuller-v0',
    entry_point='gym_ae5117.envs:TriPullerEnv',
    timestep_limit=100,
    nondeterministic=True
)
