from gym.envs.registration import register

register(
    id='SoloEscape-v0',
    entry_point='gym_linzhank.envs:SoloEscapeEnv',
    nondeterministic=True
)

register(
    id='TriPuller-v0',
    entry_point='gym_linzhank.envs:TriPullerEnv',
    nondeterministic=True
)

register(
    id='TwoCarrier-v0',
    entry_point='gym_linzhank.envs:TwoCarrierEnv',
    nondeterministic=True
)

