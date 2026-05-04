import gymnasium as gym
from stable_baselines3.common import monitor
from stable_baselines3 import PPO
import pandas as pd
from matplotlib import pyplot as plt

import envRegistration
envRegistration.registerEnv()

def plot_training_rewards(log_file_path, save_fig_name=None):
    df = pd.read_csv(log_file_path, skiprows=1)

    rewards = df["r"]
    timesteps = df["l"].cumsum()

    plt.plot(timesteps, rewards)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Training Rewards")
    plt.grid(True)

    if save_fig_name is not None:
        plt.savefig(f"{save_fig_name}.png")
    plt.show()



existing_model_path = "Models/AdjMap_Only/Original_Obs_Space/ppo_Construction_Prototype_10x10_AdjMap_Only_3000000.zip"

map_size_x = 20
map_size_y = 20

model_Name = f"Construction_Prototype_{map_size_x}x{map_size_y}_AdjMap_Only"


total_training_timesteps = 3000000
env = monitor.Monitor(
gym.make("ConstructionPrototypeRandom", render_mode="none", map_size_x=map_size_x, map_size_y=map_size_y),
filename="training_logs")

model = PPO.load(path=existing_model_path, env=env)
model.learn(total_timesteps=total_training_timesteps, reset_num_timesteps=False)

# Save the model
model.save(f"Models/ppo_{model_Name}_{total_training_timesteps}")
plot_training_rewards(f"training_logs/monitor.csv",
                        f"training_logs/ConstructionPrototype_Single_Agent_Random_Schematic-{total_training_timesteps}-timesteps")