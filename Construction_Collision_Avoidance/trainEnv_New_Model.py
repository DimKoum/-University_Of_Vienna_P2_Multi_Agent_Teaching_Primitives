import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3.common import monitor
from stable_baselines3 import PPO
import dataUtility

import envRegistration
envRegistration.registerEnv()

# Map size parameters
map_size_x = 10
map_size_y = 10

extra_name_notes = "Constrained"
model_Name = f"Construction_Collision_Avoidance_{map_size_x}x{map_size_y}_AdjMap_Only_{extra_name_notes}"
total_training_timesteps = 2000000

env = monitor.Monitor(
    gym.make("ConstructionCollisionsConstrained", render_mode="none", map_size_x=map_size_x, map_size_y=map_size_y),
    filename="training_logs")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=total_training_timesteps)

#Save the model
# model.save(f"Models/ppo_{model_Name}_{total_training_timesteps}")

dataUtility.plot_training_rewards(f"training_logs/monitor.csv",
                     f"training_logs/ConstructionPrototype_Single_Agent_Random_Schematic-{total_training_timesteps}-timesteps_{extra_name_notes}")
plt.show()

# Plot the elapsed timesteps and correct placement ratio of each episode together
obs, info = env.reset()
plt.subplot(1, 3, 1)
dataUtility.plot_elapsed_timesteps(info["timesteps"])
plt.subplot(1, 3, 2)
dataUtility.plot_construction_ratios(info["correct_build_action_ratio"])
plt.subplot(1, 3, 3)
dataUtility.plot_construction_ratios(info["number_of_collisions"])
plt.suptitle("Collision Avoidance Performance Metrics")
plt.show()
