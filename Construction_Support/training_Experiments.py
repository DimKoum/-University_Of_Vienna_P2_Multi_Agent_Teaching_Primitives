import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
import pandas as pd
import dataUtility
from Autoencoder import Autoencoder

import envRegistration

envRegistration.registerEnv()


def build_autoencoder(autoenc_path=None):
    # Set up model autoencoder
    autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=2048,
                          latent_space_size=25,
                          input_dim=49,
                          activation=torch.nn.ReLU)

    state_dict = torch.load(autoenc_path)
    autoenc.load_state_dict(state_dict=state_dict)

    return autoenc


env_id = "ConstructionSupport"
env_id_encoded = "ConstructionSupport_Encoded"

# Build autoencoder(If env uses one)
autoencoder_path = "Saved_Autoencoders/autoenc2.pth"

autoenc = None

if autoencoder_path is not None:
    autoenc = build_autoencoder(autoenc_path=autoencoder_path)
    env = gym.make(env_id_encoded, render_mode="none", map_size_x=10, map_size_y=10,
                   autoencoder=autoenc, n_constructors=5)
else:
    env = gym.make(env_id, render_mode="none", map_size_x=10, map_size_y=10, n_constructors=5)

experiment_repetitions = 1
training_run_timesteps = 1000000

# Create dynamic lists in a dictionary for holding the info data of each training run

experiment_data = {}

_, info = env.reset()

# Set up the data dictionary keys
for key in info.keys():  # env info is always a dictionary
    new_key = key + "_list"
    experiment_data[new_key] = []

# Append data to the corresponding key
for i in range(experiment_repetitions):

    policy = "MultiInputPolicy"

    # Recreate environment
    if autoencoder_path is not None:
        env = gym.make(env_id_encoded, render_mode="none", map_size_x=10, map_size_y=10,
                       autoencoder=autoenc, n_constructors=5)
        policy = "MlpPolicy"
    else:
        env = gym.make(env_id, render_mode="none", map_size_x=10, map_size_y=10,n_constructors=5)

    #Train environment
    model = PPO(policy, env, verbose=1)
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
experiment_data_standard_deviations = {}
experiment_data_variances = {}

plot_number = 1  # For plotting in a subplot

for key in info.keys():
    corresponding_key = key + "_list"

    np_array = np.array(experiment_data[corresponding_key])

    # Save the averages
    averages = np.nanmean(np_array, axis=0)
    experiment_data_averages[key] = averages

    #Save variances
    variances = np.nanvar(np_array, axis=0)
    experiment_data_variances[key] = variances

    #Save standard deviations
    standard_deviations = np.nanstd(np_array, axis=0)
    experiment_data_standard_deviations[key] = standard_deviations

    # Plot the averages for this key
    plt.subplot(len(info.keys()), 1, plot_number)
    plot_number += 1

    dataUtility.line_Plot(x=np.arange(0, len(averages)), y=averages, xLabel="Episode", yLabel=key, title=" ")

plt.suptitle(f"Construction_Support(Average Training Performance ({experiment_repetitions} runs)")
plt.show()

# Save the averages as a csv file
df = pd.DataFrame(experiment_data_averages)
df.to_csv(f"ConstructionSupport_{experiment_repetitions}repetitions_Averages_5constructors.csv",
          index=False)

df = pd.DataFrame(experiment_data_standard_deviations)
df.to_csv(f"ConstructionSupport_{experiment_repetitions}repetitions_Standard_Deviations_5constructors.csv",
          index=False)

df = pd.DataFrame(experiment_data_variances)
df.to_csv(f"ConstructionSupport_{experiment_repetitions}repetitions_Variances_5constructors.csv",
          index=False)

print("Results saved")
