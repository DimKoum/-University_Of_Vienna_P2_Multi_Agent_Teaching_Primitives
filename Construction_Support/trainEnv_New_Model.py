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

extra_name_notes = ""
model_Name = f"Construction_Support_{map_size_x}x{map_size_y}_{extra_name_notes}"
total_training_timesteps = 100000

env = monitor.Monitor(
    gym.make("ConstructionSupport", render_mode="none", map_size_x=map_size_x, map_size_y=map_size_y,n_constructors=5),
    filename="training_logs")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=total_training_timesteps)

#Save the model
model.save(f"Models/ppo_{model_Name}_{total_training_timesteps}")

dataUtility.plot_training_rewards(f"training_logs/monitor.csv",
                     f"training_logs/ConstructionSupport-{total_training_timesteps}-timesteps_{extra_name_notes}")
plt.show()

#Plot the elapsed timesteps and correct placement ratio of each episode together
obs, info = env.reset()
dataUtility.plot_elapsed_timesteps(info["timesteps"])
plt.show()
