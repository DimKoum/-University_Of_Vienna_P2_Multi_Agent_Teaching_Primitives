import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, num_layers_enc, hidden_dim,
                 latent_space_size, num_layers_dec, activation):
        super().__init__()

        # Encoder
        self.encoder_layers = nn.ModuleList()
        dim = input_dim
        for _ in range(num_layers_enc):
            self.encoder_layers.append(nn.Linear(dim, hidden_dim))
            self.encoder_layers.append(activation())
            dim = hidden_dim

        # Latent space Layer
        self.encoder_layers.append(nn.Linear(dim, latent_space_size))

        # Decoder
        self.decoder_layers = nn.ModuleList()
        dim = latent_space_size
        for _ in range(num_layers_dec):
            self.decoder_layers.append(nn.Linear(dim, hidden_dim))
            self.decoder_layers.append(activation())
            dim = hidden_dim

        # Reconstruction layer
        self.decoder_layers.append(nn.Linear(dim, input_dim))

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, l):
        for layer in self.decoder_layers:
            l = layer(l)
        return l

    def forward(self, x):
        l = self.encode(x)
        d = self.decode(l)
        return d

    def fit(self, x_train, epochs=100, learning_rate=0.001, batch_size=64):

        dataset = TensorDataset(x_train, x_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_function = nn.MSELoss()
        optimiser = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, _ in loader:
                optimiser.zero_grad()

                reconstructed = self(batch_x)
                loss = loss_function(reconstructed, batch_x)
                loss.backward()

                epoch_loss += loss.item()
                optimiser.step()

            epoch_loss /= len(loader)  # Average patch loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
