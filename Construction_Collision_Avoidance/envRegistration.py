import gymnasium as gym


def registerEnv():
    gym.register(
        id='ConstructionCollisions',
        entry_point='ConstructionCollisions:ConstructionCollisions',
        max_episode_steps=5000
    )


