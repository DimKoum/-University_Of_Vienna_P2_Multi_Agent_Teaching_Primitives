import gymnasium as gym
import torch
from stable_baselines3 import PPO
from Autoencoder import Autoencoder
from matplotlib import pyplot as plt

from envRegistration import registerEnv
registerEnv()

# Set up autoencoder
autoencoder_path = f"Saved_Autoencoders/autoenc2.pth"
autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=2048, latent_space_size=25,
                      input_dim=49,
                      activation=torch.nn.ReLU)

state_dict = torch.load(autoencoder_path)
autoenc.load_state_dict(state_dict=state_dict)

map_size_x = 10
map_size_y = 10

predefined_schematic = None  # Set to None for random schematics

env = gym.make("ConstructionSupport_Encoded", render_mode="human", map_size_x=map_size_x, map_size_y=map_size_y
               , predefined_schematic=predefined_schematic,autoencoder=autoenc,n_constructors=5)

model = PPO.load("Models/ppo_ConstructionSupport_10x10_25ls_2048numHid_5_Constructors_Encoder_100000.zip")

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
