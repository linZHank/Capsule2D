from gym.envs.registration import register

register(
    id='SoloEscaper-v0',
    entry_point='gym_linzhank.envs:SoloEscaperEnv',
    nondeterministic=True
)

register(
    id='TriPuller-v0',
    entry_point='gym_linzhank.envs:TriPullerEnv',
    nondeterministic=True
)

register(
    id='DuoCarrier-v0',
    entry_point='gym_linzhank.envs:DuoCarrierEnv',
    nondeterministic=True
)

