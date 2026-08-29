from gymnasium import spaces
import numpy as np
import torch
from ConstructionPrototypeRandom import ConstructionPrototypeRandom


class ConstructionPrototypeRandomEncoded(ConstructionPrototypeRandom):
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

    def get_Adjacency_Map(self, repair_unit, size=3):
        """
            :param repair_unit: Repair unit worker to get the adjacency map for.
            :Size repair_unit: Size of the adjacency map (Must have a remainder of 1 when divided by 2.
            :return: The "size" x "size" adjacency map with the agent represented at the center.
        """
        if size % 2 != 1:
            raise Exception(
                "Agent adjacency map size must not be divisible by 2, try sizes such as 3,5 for 3x3 and 5x5 maps")
        adjacency_map = np.zeros([size, size])

        repair_unit_x = repair_unit.x
        repair_unit_y = repair_unit.y

        coordinate_search_range = np.arange(step=1, start=-(size // 2), stop=(size // 2) + 1)

        for i in range(len(coordinate_search_range)):
            for j in range(len(coordinate_search_range)):

                x_to_check = repair_unit_x + coordinate_search_range[i]
                y_to_check = repair_unit_y + coordinate_search_range[j]

                if x_to_check < 0 or x_to_check >= self.map_size_X:
                    adjacency_map[i][j] = -1
                    continue

                if y_to_check < 0 or y_to_check >= self.map_size_Y:
                    adjacency_map[i][j] = -1
                    continue

                adjacency_map[i][j] = self.ground_map[x_to_check][y_to_check]

                # A simplification of the adjacency map for the agent. Set all positions with a state id of 1 or above to 0
                if adjacency_map[i][j] > 1:
                    adjacency_map[i][j] = 0

        return adjacency_map.transpose()
