import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import dataUtility
from Autoencoder import Autoencoder

#Environment registers
import envRegistration
envRegistration.registerEnv()


def build_autoencoder(autoenc_path=None):
    # Set up model autoencoder

    autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=1024,
                          latent_space_size=25,
                          input_dim=49,
                          activation=torch.nn.ReLU)

    state_dict = torch.load(autoenc_path)
    autoenc.load_state_dict(state_dict=state_dict)
    return autoenc


env = gym.make("ConstructionCollisions", render_mode="none", map_size_x=10, map_size_y=10)

# # Build autoencoder(If env uses one)
#
# autoencoder_path = f"Autoencoder_Model_Pairs/4/autoenc_numSamples100000_epochs500_activationReLU_numLayers1_numHidden1024_latSpace25_simplified_directSampling_2.pth"
# autoenc = build_autoencoder(autoenc_path=autoencoder_path)
# env = gym.make("ConstructionPrototypeRandom_Encoded", render_mode="none", map_size_x=10, map_size_y=10,
#                autoencoder=autoenc)

experiment_repetitions = 5
training_run_timesteps = 2000000

# Create dynamic lists in a dictionary for holding the info data of each training run

experiment_data = {}

_, info = env.reset()

# Set up the data dictionary keys
for key in info.keys():  # env info is always a dictionary
    new_key = key + "_list"
    experiment_data[new_key] = []

# Append data to the corresponding key
for i in range(experiment_repetitions):
    # Recreate environment
    env = gym.make("ConstructionCollisions", render_mode="none", map_size_x=10, map_size_y=10)

    #Train environment
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=training_run_timesteps)

    #Get info data and save it
    _, info = env.reset()

    for key in info.keys():
        corresponding_key = key + "_list"
        experiment_data[corresponding_key].append(info[key])

# Pad the data for computing the averages next
for key in experiment_data.keys():  # For each metric
    data = experiment_data[key]

    # Find the largest entry list for that metric
    largest_list_size = 0
    for list in data:
        if len(list) > largest_list_size:
            largest_list_size = len(list)
    # Go through each list and add placeholder values if necessary
    for list in data:
        if len(list) < largest_list_size:
            size_difference = np.abs(len(list) - largest_list_size)
            for i in range(size_difference):
                list.append(np.nan)

# For each info key, take the average of each index
experiment_data_averages = {}

plot_number = 1  # For plotting in a subplot

for key in info.keys():
    corresponding_key = key + "_list"

    np_array = np.array(experiment_data[corresponding_key])
    averages = np.nanmean(np_array, axis=0)

    experiment_data_averages[key] = averages

    # Plot the averages for this key
    plt.subplot(len(info.keys()), 1, plot_number)
    plot_number += 1

    dataUtility.line_Plot(x=np.arange(0, len(averages)), y=averages, xLabel="Episode", yLabel=key, title=" ")

plt.suptitle(f"Construction(Collision Avoidance Reduced Penalty) Training Performance ({experiment_repetitions} runs)")
plt.show()

# Save the averages as a csv file
df = pd.DataFrame(experiment_data_averages)
df.to_csv(f"Construction_Collision_Avoidance_Reduced_Penalty{experiment_repetitions}repetitions_Averages.csv",
          index=False)
print("Results saved")
