import itertools

import pandas as pd

from Autoencoder import Autoencoder
import gymnasium as gym
import numpy as np
import torch
from torch import nn

import envRegistration
envRegistration.registerEnv()


#Sample observation space
def sample_obs(n=1000, sampling_method="random"):
    x = []
    for _ in range(n):
        sample = None
        env.reset()
        if sampling_method == "direct":
            sample, _, _, _, _ = env.step(action=env.action_space.sample())
        else:
            sample = env.observation_space.sample()
        flattened_sample = flatten_dict_sample(sample)
        x.append(flattened_sample)
    return x


def flatten_dict_sample(sample_dict):
    return np.concatenate([np.array(v).ravel() for v in sample_dict.values()])


def sum_abs_differences(x, y):
    x = np.array(x)
    y = np.array(y)
    if len(x) != len(y):
        raise Exception("Vectors x and y must be the same length")

    total_difference = np.absolute(np.array(x) - np.array(y))
    return sum(total_difference)


def get_avg_reconstruction_score(autoencoder, tensor_data):
    total_reconstruction_score = 0
    with torch.no_grad():
        for i in range(len(tensor_data)):
            original = tensor_data[i]
            encoded = autoencoder.encode(original)
            decoded = autoencoder.decode(encoded)
            total_reconstruction_score += sum_abs_differences(original, decoded)
        return total_reconstruction_score / len(tensor_data)


# Create and sample environment
env = gym.make("ConstructionPrototypeRandom", render_mode="none")

number_of_observation_space_samples = 100000
observation_space_sample_data = sample_obs(n=number_of_observation_space_samples,sampling_method="direct")

# Train/test split
train_test_data_split = 0.8  # 80:20 train/test data split
split_point = int(train_test_data_split * len(observation_space_sample_data))
X_train = observation_space_sample_data[:split_point]
X_test = observation_space_sample_data[split_point:]

# Transform training data into a tensor
x_tensor = torch.tensor(X_train, dtype=torch.float32)
x_tensor_test = torch.tensor(X_test, dtype=torch.float32)

# Define hyperparameters
activation_functions = [nn.ReLU]
number_of_layers_encoder = [1]
number_of_layers_decoder = [1]
hidden_layer_neurons = [256]
latent_space_size = [25]

hyperparameters = list(itertools.product(activation_functions, number_of_layers_encoder,
                                         number_of_layers_decoder, hidden_layer_neurons, latent_space_size))
total_hyperparameter_combinations = len(hyperparameters)

experiment_repetitions = 1
epochs = 100
learning_rate = 0.001

results = []

counter = 0  # Simply for showing how many iterations have been completed

# Run Experiments
for activation, num_layers_enc, num_layers_dec, hid_layer_neurons, lat_space_size in hyperparameters:
    total_average_reconstruction_score_training = 0
    total_average_reconstruction_score_test = 0

    for i in range(experiment_repetitions):
        autoenc = Autoencoder(activation=activation, input_dim=len(X_train[0]), num_layers_enc=num_layers_enc,
                              num_layers_dec=num_layers_dec,
                              hidden_dim=hid_layer_neurons, latent_space_size=lat_space_size)
        autoenc.fit(x_train=x_tensor, epochs=epochs, learning_rate=learning_rate)

        #Test Autoencoder
        total_average_reconstruction_score_training += get_avg_reconstruction_score(autoencoder=autoenc,
                                                                                    tensor_data=x_tensor)
        total_average_reconstruction_score_test += get_avg_reconstruction_score(autoencoder=autoenc,
                                                                                tensor_data=x_tensor_test)

        if experiment_repetitions == 1:  # Save model if we are not performing hyperparameter search
            # Save the autoencoder model
            print("Saving Model")
            model_save_path = f"Saved_Autoencoders/autoenc_numSamples{number_of_observation_space_samples}_epochs{epochs}_activation{activation.__name__}_numLayers{num_layers_enc}_numHidden{hid_layer_neurons}_latSpace{lat_space_size}_simplified_directSampling.pth"
            torch.save(autoenc.state_dict(), f=model_save_path)
            print("Model Saved")

    average_average_reconstruction_score_training = total_average_reconstruction_score_training / experiment_repetitions
    average_average_reconstruction_score_test = total_average_reconstruction_score_test / experiment_repetitions

    print(f"Average reconstruction score (Training set):{average_average_reconstruction_score_training}")
    print(f"Average reconstruction score (Test set):{average_average_reconstruction_score_test}")

    counter += 1
    print(f"Finished with hyperparameter combination {counter} out of {total_hyperparameter_combinations}")

    # Gather Data and save
    results.append({
        "activation": activation.__name__,
        "num_layers_enc": num_layers_enc,
        "num_layers_dec": num_layers_dec,
        "hid_layer_neurons": hid_layer_neurons,
        "lat_space_size": lat_space_size,
        "training_reconstruction_score": average_average_reconstruction_score_training,
        "test_reconstruction_score": average_average_reconstruction_score_test
    })

df = pd.DataFrame(results)
df.to_csv(f"Autoencoder_hyperparameter_search_results_epochs{epochs}_learningRate{learning_rate}_simplified_direct_sampling.csv",
          index=False)
print("Results saved")
