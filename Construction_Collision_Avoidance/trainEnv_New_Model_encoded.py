import gymnasium as gym
import torch
from stable_baselines3.common import monitor
from stable_baselines3 import PPO
import dataUtility
from matplotlib import pyplot as plt
from Autoencoder import Autoencoder


def build_autoencoder(autoenc_path = None):

    # Set up model autoencoder
    autoencoder_path = f"Saved_Autoencoders/autoenc_numSamples100000_epochs100_activationReLU_numLayers1_numHidden256_latSpace25_simplified_directSampling.pth"
    autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=num_hidden_neurons,
                          latent_space_size=latent_observation_space_size,
                          input_dim=49,
                          activation=torch.nn.ReLU)

    state_dict = torch.load(autoencoder_path)
    autoenc.load_state_dict(state_dict=state_dict)
    return autoenc

gym.register(
    id='ConstructionPrototypeRandom_Encoded',
    entry_point='ConstructionPrototypeRandom_Encoded:ConstructionPrototypeRandomEncoded',
    max_episode_steps=5000
)

#Autoencoder_specifications
latent_observation_space_size = 25
num_hidden_neurons = 256

# Set up model autoencoder
autoencoder_path = f"Saved_Autoencoders/autoenc_numSamples100000_epochs100_activationReLU_numLayers1_numHidden256_latSpace25_simplified_directSampling.pth"
autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=num_hidden_neurons,
                      latent_space_size=latent_observation_space_size,
                      input_dim=49,
                      activation=torch.nn.ReLU)

state_dict = torch.load(autoencoder_path)
autoenc.load_state_dict(state_dict=state_dict)

# Map size parameters
map_size_x = 10
map_size_y = 10

extra_name_notes = "simplified_direct_labeling"  # For appending extra information at the end of the model and learning curve figure name

model_Name = f"Construction_Encoded_{map_size_x}x{map_size_y}_AdjMap_Only_{latent_observation_space_size}ls_{num_hidden_neurons}numHid_{extra_name_notes}"
total_training_timesteps = 1000000

env = monitor.Monitor(
    gym.make("ConstructionPrototypeRandom_Encoded", render_mode="none", map_size_x=map_size_x, map_size_y=map_size_y,
             autoencoder=autoenc), filename="training_logs")

# Start training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=total_training_timesteps)

# Save the model
model.save(f"Models/ppo_{model_Name}_{total_training_timesteps}")
dataUtility.plot_training_rewards(f"training_logs/monitor.csv",
                                  f"training_logs/ConstructionPrototype_Single_Agent_Random_Schematic-{total_training_timesteps}-timesteps_encoded_{latent_observation_space_size}ls_{extra_name_notes}")
plt.show()

# Plot the elapsed timesteps and correct placement ratio of each episode together
obs, info = env.reset()
plt.subplot(1, 2, 1)
dataUtility.plot_elapsed_timesteps(info["episode_timesteps_list"])
plt.subplot(1, 2, 2)
dataUtility.plot_construction_ratios(info["episode_correct_build_action_ratios"])
plt.show()
