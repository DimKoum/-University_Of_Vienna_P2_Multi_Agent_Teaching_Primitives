from gymnasium import spaces

from Renderer import Renderer
from Repairunit import *
import numpy as np
import gymnasium as gym
import random
import math


def getEuclideanDistance(originCords, endCords):
    euclideanDistance = math.dist(originCords, endCords)
    return euclideanDistance


class ConstructionPrototypeRandom(gym.Env):
    def __init__(self, render_mode, map_size_x=10, map_size_y=10, predefined_schematic=None):

        self.map_size_X = map_size_x
        self.map_size_Y = map_size_y

        # Set up renderer if render mode is human
        self.renderer = None
        self.renderer_mode = render_mode
        if self.renderer_mode == "human":
            self.renderer = Renderer()

        self.test_Repair_Unit = RepairUnit(random.randint(0, self.map_size_X - 1),
                                           random.randint(0, self.map_size_Y - 1))

        self.repairUnits = []
        self.repairUnits.append(self.test_Repair_Unit)

        self.agent_map = np.zeros((self.map_size_X, self.map_size_Y))
        self.ground_map = np.zeros((self.map_size_X, self.map_size_Y))

        self.schematic_blocks = []
        self.predefined_schematic = predefined_schematic

        if self.predefined_schematic is None:
            # Random number of schematics with the max being map_size_x * map_size_y divided by 2
            self.number_of_schematic_blocks = random.randint(0, np.abs(self.map_size_Y * self.map_size_Y) // 2)

            # Set up random sample coordinates for the blocks
            all_coords = [(x, y) for x in range(self.map_size_X) for y in range(self.map_size_Y)]
            self.schematic_blocks = random.sample(all_coords, self.number_of_schematic_blocks)
        else:
            self.schematic_blocks = predefined_schematic.copy()

        self.placed_blocks = []
        self.number_of_correctly_placed_blocks = 0

        self.previous_closest_distance_to_schematic = None

        # Action and observation space
        self.action_space = gym.spaces.Discrete(5)

        self.observation_space = spaces.Dict({
            "adjacency_map": spaces.Box(
                low=-1,
                high=1,
                shape=(7, 7),
                dtype=np.int32
            ),
        })

        # Info metrics
        self.episode_timesteps = 0
        self.total_build_actions = 0

        #Info lists
        self.episode_timesteps_list = []
        self.correct_build_actions_ratios = []

    def _get_obs(self):
        return {
            "adjacency_map": self.get_Adjacency_Map(self.test_Repair_Unit, size=7)
        }


    def getClosestSchematicTile(self, own_coordinates):
        """
            :param own_coordinates: Set of coordinates (Array)
            :return: Coordinates of the closest schematic tile to the input coordinates
        """
        minDistanceTile = None
        minDistance = math.inf

        # Find which tile has the minimum distance
        for tile in self.schematic_blocks:
            distance = getEuclideanDistance(own_coordinates, tile)
            if distance < minDistance:

                # If the tile is not a state ther than "to be built on" do not take it into account
                if self.ground_map[tile[0]][tile[1]] != 1:
                    continue
                minDistanceTile = tile
                minDistance = distance

        # If the min tile is still none, it means there are no more tiles to build on, so return a placeholder value
        if minDistanceTile is None:
            return [-1, -1]

        return minDistanceTile

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

    def _get_info(self):

        return {
            "episode_timesteps": self.episode_timesteps_list,
            "episode_correct_build_action_ratios": self.correct_build_actions_ratios
        }

    def reset(self, seed=None, options=None):

        self.test_Repair_Unit.x = random.randint(0, self.map_size_X - 1)
        self.test_Repair_Unit.y = random.randint(0, self.map_size_Y - 1)

        self.placed_blocks.clear()

        self.schematic_blocks.clear()
        if self.predefined_schematic is None:
            # Random number of schematics with the max being map_size_x * map_size_y divided by 2
            self.number_of_schematic_blocks = random.randint(0, np.abs(self.map_size_Y * self.map_size_Y) // 2)

            # Set up random sample coordinates for the blocks
            all_coords = [(x, y) for x in range(self.map_size_X) for y in range(self.map_size_Y)]
            self.schematic_blocks = random.sample(all_coords, self.number_of_schematic_blocks)
        else:
            self.schematic_blocks = self.predefined_schematic.copy()

        observation = self._get_obs()
        info = self._get_info()

        # Append episode metrics to lists
        if self.total_build_actions != 0:
            self.episode_timesteps_list.append(self.episode_timesteps)
            self.correct_build_actions_ratios.append(self.number_of_correctly_placed_blocks / self.total_build_actions)

        # Reset episode metrics

        self.number_of_correctly_placed_blocks = 0
        self.episode_timesteps = 0
        self.total_build_actions = 0

        return observation, info

    def movement_Intent(self, entity, movementType):

        if movementType == "right":
            if entity.x < self.map_size_X - 1:
                entity.moveRight()

        if movementType == "left":
            if entity.x > 0:
                entity.moveLeft()

        if movementType == "down":
            if entity.y < self.map_size_Y - 1:
                entity.moveDown()

        if movementType == "up":
            if entity.y > 0:
                entity.moveUp()

    def action_map(self, action, entity):
        if isinstance(entity, RepairUnit):
            if action == 0 and not entity.isConstructing:
                self.movement_Intent(entity=self.test_Repair_Unit, movementType="right")
            if action == 1 and not entity.isConstructing:
                self.movement_Intent(entity=self.test_Repair_Unit, movementType="left")
            if action == 2 and not entity.isConstructing:
                self.movement_Intent(entity=self.test_Repair_Unit, movementType="up")
            if action == 3 and not entity.isConstructing:
                self.movement_Intent(entity=self.test_Repair_Unit, movementType="down")
            if action == 4 and not entity.isConstructing and (
                    self.ground_map[entity.x][entity.y] == 0 or self.ground_map[entity.x][entity.y] == 1):
                self.test_Repair_Unit.begin_Construction()

    def step(self, action):
        terminated = False
        truncated = False
        reward = -0.01

        self.action_map(action=action, entity=self.test_Repair_Unit)

        self.agent_map = np.zeros((self.map_size_X, self.map_size_Y))
        self.ground_map = np.zeros((self.map_size_X, self.map_size_Y))

        # BUILD THE AGENT(S) MAP
        self.agent_map[self.test_Repair_Unit.x][self.test_Repair_Unit.y] = 1

        # BUILD THE GROUND MAP
        for schematic_block in self.schematic_blocks:
            self.ground_map[schematic_block[0]][schematic_block[1]] = 1

        for placed_block in self.placed_blocks:
            # Check if placed blocks align with the schematic
            if self.ground_map[placed_block[0]][placed_block[1]] == 1:
                self.ground_map[placed_block[0]][placed_block[1]] = 2
            else:
                self.ground_map[placed_block[0]][placed_block[1]] = 3

        # HANDLE REPAIR UNIT EVENTS
        for repairUnit in self.repairUnits:
            repairUnit.timestep()

            # Reward the unit for constructing while on a schematic tile
            if repairUnit.isConstructing and self.ground_map[self.test_Repair_Unit.x][self.test_Repair_Unit.y] == 1:
                reward += 5
            elif repairUnit.isConstructing and self.ground_map[self.test_Repair_Unit.x][self.test_Repair_Unit.y] != 1:
                reward -= 1

            if repairUnit.isConstruction_done():
                self.total_build_actions += 1

                self.placed_blocks.append((self.test_Repair_Unit.x, self.test_Repair_Unit.y))
                # Construction rewards and penalties

                # Check If the built tile is wrongly placed and perform more updates on the ground map
                if self.ground_map[self.test_Repair_Unit.x][self.test_Repair_Unit.y] != 1:
                    reward += -5
                    self.ground_map[self.test_Repair_Unit.x][self.test_Repair_Unit.y] = 3
                else:
                    reward += 10
                    self.ground_map[self.test_Repair_Unit.x][self.test_Repair_Unit.y] = 2
                    self.number_of_correctly_placed_blocks += 1

        if self.renderer is not None:
            self.renderer.render(self.map_size_X, self.map_size_Y, self.agent_map, self.ground_map)

        if self.number_of_correctly_placed_blocks == len(self.schematic_blocks):
            terminated = True
            reward += 100

        observation = self._get_obs()

        closest_schematic_tile = self.getClosestSchematicTile([self.test_Repair_Unit.x, self.test_Repair_Unit.y])

        # Give the repair unit a reward for getting closer to schematic tiles
        if self.previous_closest_distance_to_schematic is None:
            distance = getEuclideanDistance([self.test_Repair_Unit.x, self.test_Repair_Unit.y],
                                            closest_schematic_tile)
            self.previous_closest_distance_to_schematic = distance
        else:
            current_distance = getEuclideanDistance([self.test_Repair_Unit.x, self.test_Repair_Unit.y],
                                                    closest_schematic_tile)
            if current_distance < self.previous_closest_distance_to_schematic:
                self.previous_closest_distance_to_schematic = current_distance
                reward += 1

        info = self._get_info()

        self.episode_timesteps += 1

        return observation, reward, terminated, truncated, info
