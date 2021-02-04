from gym.envs.registration import register

register(
    id='TriPuller-v0',
    entry_point='gym_ae5117.envs:TriPullerEnv',
    nondeterministic=True
)

register(
    id='TwoCarrier-v0',
    entry_point='gym_ae5117.envs:TwoCarrierEnv',
    nondeterministic=True
)

