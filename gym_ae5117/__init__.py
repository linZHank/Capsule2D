from gym.envs.registration import register

register(
    id='ThreePuller-v0',
    entry_point='gym_ae5117.envs:ThreePullerEnv',
    timestep_limit=100,
    nondeterministic=True
)

register(
    id='TwoCarrier-v0',
    entry_point='gym_ae5117.envs:TwoCarrierEnv',
    timestep_limit=1000,
    nondeterministic=True
)

