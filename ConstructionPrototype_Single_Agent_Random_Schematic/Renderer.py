import pygame
import numpy as np


class Renderer:
    def __init__(self):

        self.window_size_X = 600
        self.window_size_Y = 600

        self.square_Offset_X = None
        self.square_Offset_Y = None

        pygame.init()
        self.screen = pygame.display.set_mode([self.window_size_X, self.window_size_Y], pygame.RESIZABLE)
        pygame.display.set_caption('Construction Prototype(Single Agent)')

        self.clock = pygame.time.Clock()

    def render(self, map_size_X, map_size_Y, agent_Data, ground_Data):

        self.clock.tick(60)

        self.square_Offset_X = self.window_size_X // map_size_X
        self.square_Offset_Y = self.window_size_Y // map_size_Y

        # Fill background
        background = pygame.Surface(self.screen.get_size())
        background = background.convert()
        background.fill((0, 0, 0))

        # Blit everything to the screen
        self.screen.blit(background, (0, 0))
        pygame.display.flip()

        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Render Background
        self.screen.blit(background, (0, 0))

        # Render grid on screen
        self.draw_Map(agent_Data=agent_Data,ground_Data=ground_Data)

        pygame.display.flip()

    def draw_Map(self, agent_Data,ground_Data):
        # Render grid on screen
        mapX = 0
        mapY = 0

        for x_position in list(range(0, self.window_size_X, self.square_Offset_X)):
            for y_position in list(range(0, self.window_size_Y, self.square_Offset_Y)):

                # First draw the empty tile
                pygame.draw.rect(self.screen, "white",
                                 [x_position, y_position, self.square_Offset_X, self.square_Offset_Y],
                                 1)

                # Draw terrain
                """ 
                If ground_data x,y value = 1 : Schematic (light brown tile)
                If ground_data x,y value = 2 : Correctly Placed block (dark green tile)
                If ground_data x,y value = 3 : Incorrectly Placed block (dark red tile)
                """

                if ground_Data[mapX][mapY] == 1:
                    pygame.draw.rect(self.screen, "antiquewhite2",
                                     [x_position, y_position, self.square_Offset_X, self.square_Offset_Y],
                                     0)
                if ground_Data[mapX][mapY] == 2:
                    pygame.draw.rect(self.screen, "aquamarine3",
                                     [x_position, y_position, self.square_Offset_X, self.square_Offset_Y],
                                     0)
                if ground_Data[mapX][mapY] == 3:
                    pygame.draw.rect(self.screen, "coral3",
                                     [x_position, y_position, self.square_Offset_X, self.square_Offset_Y],
                                     0)

                # Draw agents
                """ 
                If agent_data x,y value = 1 : Agent ( White tile) smaller than a tile
                """
                if agent_Data[mapX][mapY] == 1:
                    pygame.draw.rect(self.screen, "white",
                                     [x_position +(self.square_Offset_X/4), y_position+(self.square_Offset_Y/4), self.square_Offset_X/2, self.square_Offset_Y/2],
                                     0)
                mapY += 1

            mapY = 0
            mapX += 1


if __name__ == '__main__':
    map_size_X = 10
    map_size_Y = 10
    renderer = Renderer()

    player_Coordinates = (2, 2)
    target_Coordinates = (2,2)

    mock_agent_map = np.zeros((map_size_X, map_size_Y))
    mock_agent_map[player_Coordinates[0]][player_Coordinates[1]] = 1

    mock_ground_map = np.zeros((map_size_X, map_size_Y))
    mock_ground_map[target_Coordinates[0]][target_Coordinates[1]] = 1

    while True:
        renderer.render(map_size_X=map_size_X, map_size_Y=map_size_Y, agent_Data=mock_agent_map, ground_Data=mock_ground_map)
