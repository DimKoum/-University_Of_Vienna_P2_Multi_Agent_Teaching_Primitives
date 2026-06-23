import gymnasium as gym


def registerEnv():
    gym.register(
        id='ConstructionCollisions',
        entry_point='ConstructionCollisions:ConstructionCollisions',
        max_episode_steps=5000
    )

    gym.register(
        id='ConstructionCollisionsConstrained',
        entry_point='ConstructionCollisions_Constrained:ConstructionCollisionsConstrained',
        max_episode_steps=5000
    )

    gym.register(
        id='ConstructionCollisionsConstrained_Encoded',
        entry_point='ConstructionCollisions_Constrained_Encoded:ConstructionCollisionsConstrainedEncoded',
        max_episode_steps=5000
    )


