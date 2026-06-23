from gymnasium import spaces

import torch
import numpy as np
import math

from ConstructionCollisions_Constrained import ConstructionCollisionsConstrained


def getEuclideanDistance(originCords, endCords):
    euclideanDistance = math.dist(originCords, endCords)
    return euclideanDistance


class ConstructionCollisionsConstrainedEncoded(ConstructionCollisionsConstrained):
    def __init__(self, render_mode, map_size_x=10, map_size_y=10, predefined_schematic=None, autoencoder=None):
        super().__init__(render_mode, map_size_x, map_size_y, predefined_schematic)

        self.observation_space = spaces.Box(
            low=-9999,
            high=9999,
            shape=(25,),
            dtype=np.float32
        )

        # Instantiate autoencoder
        self.autoenc = autoencoder

    def _get_obs(self):
        adj_map = self.get_Adjacency_Map(self.test_Repair_Unit, size=7)  # 7x7 adjacency map
        adj_map = adj_map.flatten()
        adj_map_tensor = torch.tensor(adj_map, dtype=torch.float32)
        encoded = self.autoenc.encode(adj_map_tensor)
        encoded_np = encoded.detach().cpu().numpy().flatten()

        return encoded_np
