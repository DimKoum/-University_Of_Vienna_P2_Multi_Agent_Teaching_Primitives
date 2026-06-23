import gymnasium as gym
from stable_baselines3 import PPO
import envRegistration
from matplotlib import pyplot as plt

envRegistration.registerEnv()

map_size_x = 10
map_size_y = 10

predefined_schematic = None  # Set to None for random schematics

env = gym.make("ConstructionSupport", render_mode="human", map_size_x=map_size_x, map_size_y=map_size_y
               , predefined_schematic=predefined_schematic, n_constructors=5)

model = PPO.load("Models/ppo_Construction_Support_10x10__2000000.zip")
obs, info = env.reset()

for i in range(10):
    for j in range(1000):
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

n_services_in_each_timestep = info["serviced_consturctors_in_each_timestep"]
unique_values = list(dict.fromkeys(n_services_in_each_timestep))

instances = []
for value in unique_values:
    count = n_services_in_each_timestep.count(value)
    instances.append(count)

x = unique_values
y = instances

plt.ylabel("Instances")
plt.xlabel("Number of adjacent constructors")
plt.bar(x, y)
plt.show()
