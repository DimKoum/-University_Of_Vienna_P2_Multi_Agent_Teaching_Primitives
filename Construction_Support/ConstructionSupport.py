from gymnasium import spaces
from Renderer import Renderer
from Repairunit import *
from SupportUnit import *
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym
import random
import math


def getEuclideanDistance(originCords, endCords):
    euclideanDistance = math.dist(originCords, endCords)
    return euclideanDistance


class ConstructionSupport(gym.Env):
    def __init__(self, render_mode, map_size_x=10, map_size_y=10, predefined_schematic=None, n_constructors=2,
                 expensive_logging=False):

        self.map_size_X = map_size_x
        self.map_size_Y = map_size_y

        # Set up renderer if render mode is human
        self.renderer = None
        self.renderer_mode = render_mode
        if self.renderer_mode == "human":
            self.renderer = Renderer()

        self.construction_model_path = "Models/ppo_Construction_Prototype_10x10_AdjMap_Only_Simplified_3000000.zip"
        self.construction_model = PPO.load(self.construction_model_path)

        self.supportUnit = SupportUnit(random.randint(0, self.map_size_X - 1),
                                       random.randint(0, self.map_size_Y - 1))
        self.repairUnits = []

        for i in range(n_constructors):
            repairUnit = RepairUnit(random.randint(0, self.map_size_X - 1),
                                    random.randint(0, self.map_size_Y - 1))
            self.repairUnits.append(repairUnit)

        self.agent_map = np.zeros((self.map_size_X, self.map_size_Y))
        self.ground_map = np.zeros((self.map_size_X, self.map_size_Y))

        self.schematic_blocks = []
        self.obstacle_blocks = []
        self.predefined_schematic = predefined_schematic

        if self.predefined_schematic is None:
            # Random number of schematics with the max being map_size_x * map_size_y divided by 2
            self.number_of_schematic_blocks = random.randint(0, np.abs(self.map_size_Y * self.map_size_Y) / 2)

            # Number of obstacles is half of the number of schematic blocks
            self.number_of_obstacle_blocks = 0  # self.number_of_schematic_blocks // 2

            # Set up random sample coordinates for the blocks
            all_coords = [(x, y) for x in range(self.map_size_X) for y in range(self.map_size_Y)]
            self.schematic_blocks = random.sample(all_coords, self.number_of_schematic_blocks)

            remaining_coords = list(set(all_coords) - set(self.schematic_blocks))

            self.obstacle_blocks = random.sample(remaining_coords, self.number_of_obstacle_blocks)
        else:
            self.schematic_blocks = predefined_schematic.copy()

        self.placed_blocks = []
        self.number_of_correctly_placed_blocks = 0

        self.previous_closest_distance_to_schematic = None

        # Action and observation space
        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = spaces.Dict({
            "adjacency_map": spaces.Box(
                low=-1,
                high=n_constructors,
                shape=(7, 7),
                dtype=np.int32
            ),
        })

        self.expensive_logging = expensive_logging  # Boolean for recording high volume data. Ideally not used during training as it slows it down
        # Info metrics
        self.episode_timesteps = 0
        self.total_build_actions = 0
        self.timesteps_spent_in_obstacles = 0

        #Info lists
        self.episode_timesteps_list = []
        self.correct_build_actions_ratios = []
        self.timesteps_spend_in_obstacles_list = []
        self.number_of_constructors_serviced_list = []  # List containing the number of constructors the support unit is servicing in each timestep.

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

        return {
            "adjacency_map": self.get_Adjacency_Map(full_map=filtered_map, size=7, entity=self.supportUnit)
        }

    def _get_obs_construction_unit(self, construction_unit):
        return {
            "adjacency_map": self.get_Adjacency_Map(full_map=self.ground_map, size=7, entity=construction_unit,
                                                    map_values_to_exclude=[2, 3, 4])
        }

    def getClosestSchematicTile(self, own_coordinates):
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

    def get_Adjacency_Map(self, full_map, entity, size=3, map_values_to_exclude=None):

        if map_values_to_exclude is None:
            map_values_to_exclude = []

        if size % 2 != 1:
            raise Exception(
                "Agent adjacency map size must not be divisible by 2, try sizes such as 3,5 for 3x3 and 5x5 maps")
        adjacency_map = np.zeros([size, size])

        repair_unit_x = entity.x
        repair_unit_y = entity.y

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

                adjacency_map[i][j] = full_map[x_to_check][y_to_check]

                if adjacency_map[i][j] in map_values_to_exclude:
                    adjacency_map[i][j] = 0

        return adjacency_map.transpose()

    def _get_info(self):

        return {
            "timesteps": self.episode_timesteps_list,
            "serviced_consturctors_in_each_timestep": self.number_of_constructors_serviced_list
        }

    def reset(self, seed=None, options=None):

        # Randomly re-place repair units around the grid
        for repairUnit in self.repairUnits:
            repairUnit.x = random.randint(0, self.map_size_X - 1)
            repairUnit.y = random.randint(0, self.map_size_Y - 1)

            repairUnit.construction_period_timesteps = 10  # Reset construction speed

        self.placed_blocks.clear()

        self.schematic_blocks.clear()
        if self.predefined_schematic is None:
            # Random number of schematics with the max being map_size_x * map_size_y divided by 2
            self.number_of_schematic_blocks = random.randint(0, np.abs(self.map_size_Y * self.map_size_Y) / 2)
            self.number_of_obstacle_blocks = 0  #self.number_of_schematic_blocks // 2

            # Set up random sample coordinates for the blocks
            all_coords = [(x, y) for x in range(self.map_size_X) for y in range(self.map_size_Y)]
            self.schematic_blocks = random.sample(all_coords, self.number_of_schematic_blocks)

            remaining_coords = list(set(all_coords) - set(self.schematic_blocks))

            self.obstacle_blocks = random.sample(remaining_coords, self.number_of_obstacle_blocks)
        else:
            self.schematic_blocks = self.predefined_schematic.copy()

        observation = self._get_obs()
        info = self._get_info()

        # Append episode metrics to lists
        if self.total_build_actions != 0:
            self.episode_timesteps_list.append(self.episode_timesteps)
            self.correct_build_actions_ratios.append(self.number_of_correctly_placed_blocks / self.total_build_actions)
            self.timesteps_spend_in_obstacles_list.append(self.timesteps_spent_in_obstacles)

        # Reset episode metrics

        self.number_of_correctly_placed_blocks = 0
        self.episode_timesteps = 0
        self.total_build_actions = 0
        self.timesteps_spent_in_obstacles = 0

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

    def action_map_repair_unit(self, action, entity):
        if isinstance(entity, RepairUnit):
            if action == 0 and not entity.isConstructing:
                self.movement_Intent(entity=entity, movementType="right")
            if action == 1 and not entity.isConstructing:
                self.movement_Intent(entity=entity, movementType="left")
            if action == 2 and not entity.isConstructing:
                self.movement_Intent(entity=entity, movementType="up")
            if action == 3 and not entity.isConstructing:
                self.movement_Intent(entity=entity, movementType="down")
            if action == 4 and not entity.isConstructing and (
                    self.ground_map[entity.x][entity.y] == 0 or self.ground_map[entity.x][entity.y] == 1):
                if not self.is_current_tile_already_being_constructed_on(entity=entity):
                    entity.begin_Construction()
                    self.ground_map[entity.x][entity.y] = 4

    def action_map_support_unit(self, action, support_unit):
        if isinstance(support_unit, SupportUnit):
            if action == 0:
                self.movement_Intent(entity=support_unit, movementType="right")
            if action == 1:
                self.movement_Intent(entity=support_unit, movementType="left")
            if action == 2:
                self.movement_Intent(entity=support_unit, movementType="up")
            if action == 3:
                self.movement_Intent(entity=support_unit, movementType="down")

    def is_current_tile_already_being_constructed_on(self, entity):
        for repairUnit in self.repairUnits:
            if repairUnit == entity:
                continue
            if repairUnit.isConstructing and repairUnit.x == entity.x and repairUnit.y == entity.y:
                return True
        return False

    def step(self, action):
        terminated = False
        truncated = False
        reward = -0.01

        # Perform Unit Actions

        self.action_map_support_unit(action=action, support_unit=self.supportUnit)

        for repairUnit in self.repairUnits:
            constr_unit_obs = self._get_obs_construction_unit(construction_unit=repairUnit)
            constr_unit_action, state = self.construction_model.predict(constr_unit_obs)
            constru_unit_action = constr_unit_action.item()
            self.action_map_repair_unit(action=constru_unit_action, entity=repairUnit)

        self.agent_map = np.zeros((self.map_size_X, self.map_size_Y))
        self.ground_map = np.zeros((self.map_size_X, self.map_size_Y))

        # BUILD THE AGENT(S) MAP
        n_constuctors_serviced = 0
        for repairUnit in self.repairUnits:
            self.agent_map[repairUnit.x][repairUnit.y] = 1

            if getEuclideanDistance([repairUnit.x, repairUnit.y], [self.supportUnit.x, self.supportUnit.y]) < 2:
                self.agent_map[repairUnit.x][repairUnit.y] = 3
                if repairUnit.construction_period_timesteps >= 10:
                    repairUnit.construction_period_timesteps -= 10
                reward += 1
                n_constuctors_serviced += 1
        if self.expensive_logging:
            self.number_of_constructors_serviced_list.append(n_constuctors_serviced)

        self.agent_map[self.supportUnit.x][self.supportUnit.y] = 2

        # BUILD THE GROUND MAP
        for schematic_block in self.schematic_blocks:
            self.ground_map[schematic_block[0]][schematic_block[1]] = 1

        for obstacle_block in self.obstacle_blocks:
            self.ground_map[obstacle_block[0]][obstacle_block[1]] = -1

        for placed_block in self.placed_blocks:
            # Check if placed blocks align with the schematic
            if self.ground_map[placed_block[0]][placed_block[1]] == 1:
                self.ground_map[placed_block[0]][placed_block[1]] = 2
            else:
                self.ground_map[placed_block[0]][placed_block[1]] = 3

        # HANDLE REPAIR UNIT EVENTS
        for repairUnit in self.repairUnits:
            repairUnit.timestep()
            repairUnit.construction_period_timesteps += 1  # Increase construction time

            if repairUnit.isConstruction_done():
                self.total_build_actions += 1

                self.placed_blocks.append((repairUnit.x, repairUnit.y))

                # Check If the built tile is wrongly placed and perform more updates on the ground map
                if self.ground_map[repairUnit.x][repairUnit.y] != 1:
                    self.ground_map[repairUnit.x][repairUnit.y] = 3
                else:
                    self.ground_map[repairUnit.x][repairUnit.y] = 2
                    self.number_of_correctly_placed_blocks += 1

        if self.renderer is not None:
            self.renderer.render(self.map_size_X, self.map_size_Y, self.agent_map, self.ground_map)

        if self.number_of_correctly_placed_blocks == len(self.schematic_blocks):
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        self.episode_timesteps += 1

        return observation, reward, terminated, truncated, info
