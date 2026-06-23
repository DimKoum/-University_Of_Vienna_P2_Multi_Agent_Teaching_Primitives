import gymnasium as gym
import torch
from stable_baselines3.common import monitor
from stable_baselines3 import PPO
import dataUtility
from matplotlib import pyplot as plt
from Autoencoder import Autoencoder

from envRegistration import registerEnv
registerEnv()

#Autoencoder_specifications
latent_observation_space_size = 25
num_hidden_neurons = 2048

# Set up model autoencoder
autoencoder_path = f"Saved_Autoencoders/autoenc3.pth"
autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=num_hidden_neurons,
                      latent_space_size=latent_observation_space_size,
                      input_dim=49,
                      activation=torch.nn.ReLU)

state_dict = torch.load(autoencoder_path)
autoenc.load_state_dict(state_dict=state_dict)

# Map size parameters
map_size_x = 10
map_size_y = 10
number_of_constructors = 5
extra_name_notes = f"{number_of_constructors}_Constructors_Encoder"  # For appending extra information at the end of the model and learning curve figure name

model_Name = f"ConstructionSupport_Encoded_{map_size_x}x{map_size_y}_{latent_observation_space_size}ls_{num_hidden_neurons}numHid_{extra_name_notes}_nconst_{number_of_constructors}"
total_training_timesteps = 100000

env = monitor.Monitor(
    gym.make("ConstructionSupport_Encoded", render_mode="none", map_size_x=map_size_x, map_size_y=map_size_y,
             autoencoder=autoenc,n_constructors=number_of_constructors), filename="training_logs")

# Start training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=total_training_timesteps)

# Save the model
model.save(f"Models/ppo_{model_Name}_{total_training_timesteps}")
dataUtility.plot_training_rewards(f"training_logs/monitor.csv",
                                  f"training_logs/Construction_Support-{total_training_timesteps}-timesteps_encoded_{latent_observation_space_size}ls_{extra_name_notes}")
plt.show()

# Plot the elapsed timesteps and correct placement ratio of each episode together
obs, info = env.reset()
dataUtility.plot_elapsed_timesteps(info["timesteps"])
plt.show()
