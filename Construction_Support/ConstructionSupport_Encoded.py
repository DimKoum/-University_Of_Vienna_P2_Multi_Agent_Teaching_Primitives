import torch
from gymnasium import spaces
import numpy as np
from ConstructionSupport import ConstructionSupport


class ConstructionSupportEncoded(ConstructionSupport):
    def __init__(self, render_mode, map_size_x=10, map_size_y=10, predefined_schematic=None, n_constructors=2,autoencoder= None):

        super().__init__(render_mode, map_size_x, map_size_y, predefined_schematic, n_constructors)

        self.observation_space = spaces.Box(
            low=-9999,
            high=9999,
            shape=(25,),
            dtype=np.float32
        )
        self.autoenc = autoencoder

    def _get_obs(self):

        # Apply a manual filter to the agent map to exclude agents other than the repair/construction units
        filtered_map = self.agent_map.copy()
        filter_out_values_list = [2, 3]
        for x in range(len(filtered_map)):
            for y in range(len(filtered_map[0])):
                if filtered_map[x][y] in filter_out_values_list:
                    filtered_map[x][y] = 0

        # Take note of any repair/construction units stacked together
        coordinates_record = []
        for repairUnit in self.repairUnits:
            if [repairUnit.x, repairUnit.y] in coordinates_record:
                filtered_map[repairUnit.x][repairUnit.y] += 1
            else:
                coordinates_record.append([repairUnit.x, repairUnit.y])

        adj_map = self.get_Adjacency_Map(full_map=filtered_map, size=7, entity=self.supportUnit)  # 7x7 adjacency map
        adj_map = adj_map.flatten()
        adj_map_tensor = torch.tensor(adj_map, dtype=torch.float32)
        encoded = self.autoenc.encode(adj_map_tensor)
        encoded_np = encoded.detach().cpu().numpy().flatten()

        return encoded_np
