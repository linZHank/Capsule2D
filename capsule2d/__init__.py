from gymnasium.envs.registration import register

register(
    id="CapsuleBreaker-v0",
    entry_point="capsule2d.envs:Breaker",
    max_episode_steps=500,
)
