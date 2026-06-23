import gymnasium as gym


def registerEnv():
    gym.register(
        id='ConstructionPrototypeRandom',
        entry_point='ConstructionPrototypeRandom:ConstructionPrototypeRandom',
        max_episode_steps=5000
    )

    gym.register(
        id='ConstructionPrototypeRandom_Encoded',
        entry_point='ConstructionPrototypeRandom_Encoded:ConstructionPrototypeRandomEncoded',
        max_episode_steps=5000
    )

    gym.register(
        id='ConstructionPrototypeRandom_Original',
        entry_point='ConstructionPrototypeRandom_Original:ConstructionPrototypeRandomOriginal',
        max_episode_steps=5000
    )






