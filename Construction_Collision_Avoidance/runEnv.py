import gymnasium as gym
from stable_baselines3 import PPO
import envRegistration

envRegistration.registerEnv()

map_size_x = 10
map_size_y = 10

predefined_schematic = None  # Set to None for random schematics


env = gym.make("ConstructionCollisions", render_mode="human", map_size_x=map_size_x, map_size_y=map_size_y
               , predefined_schematic=predefined_schematic)

model = PPO.load("Models/ppo_Construction_Collision_Avoidance_10x10_AdjMap_Only__3000000.zip")
# model = None
obs, info = env.reset()
while True:
    if model is not None:
        action, state = model.predict(obs)
        action = action.item()
    else:
        action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, info = env.reset()

    print(obs["adjacency_map"])
    print(info)
    print(f"REWARD = {reward}")
