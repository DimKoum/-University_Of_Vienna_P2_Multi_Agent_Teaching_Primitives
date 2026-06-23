import gymnasium as gym


def registerEnv():
    gym.register(
        id='ConstructionSupport',
        entry_point='ConstructionSupport:ConstructionSupport',
        max_episode_steps=5000
    )

    gym.register(
        id='ConstructionSupport_Encoded',
        entry_point='ConstructionSupport_Encoded:ConstructionSupportEncoded',
        max_episode_steps=5000
    )



