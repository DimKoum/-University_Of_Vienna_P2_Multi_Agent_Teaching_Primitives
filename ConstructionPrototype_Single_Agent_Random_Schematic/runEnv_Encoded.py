import gymnasium as gym
import torch
from stable_baselines3 import PPO
from Autoencoder import Autoencoder


import envRegistration
envRegistration.registerEnv()

# Set up autoencoder
autoencoder_path = "Autoencoder_Model_Pairs/3/autoenc_numSamples100000_epochs500_activationReLU_numLayers1_numHidden1024latSpace25_simplified_directSampling.pth"
autoenc = Autoencoder(num_layers_dec=1, num_layers_enc=1, hidden_dim=1024, latent_space_size=25,
                      input_dim=49,
                      activation=torch.nn.ReLU)

state_dict = torch.load(autoencoder_path)
autoenc.load_state_dict(state_dict=state_dict)

map_size_x = 10
map_size_y = 10

predefined_schematic = None  # Set to None for random schematics

env = gym.make("ConstructionPrototypeRandom_Encoded", render_mode="human", map_size_x=map_size_x, map_size_y=map_size_y
               , predefined_schematic=predefined_schematic,autoencoder=autoenc)

model = PPO.load("Autoencoder_Model_Pairs/3/ppo_Construction_Encoded_10x10_AdjMap_Only_25ls_1024numHid_simplified_direct_sampling_1000000.zip")


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

    print(obs)
    print(f"REWARD = {reward}")
