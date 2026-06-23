from Movable2DDiscrete import *


class RepairUnit(Movable2DDiscrete):

    def __init__(self, x, y):
        super().__init__(x, y)

        self.current_construction_progress = 0
        self.construction_period_timesteps = 10
        self.isConstructing = False

        self.current_cooldown = 0
        self.construction_cooldown_timesteps = 10

    def begin_Construction(self):
        # If already constructing, cancel the action
        if self.isConstructing:
            return

        # Check if enough time has passed since the previous construction job
        if self.current_cooldown < self.construction_cooldown_timesteps:
            return

        self.isConstructing = True
        # Reset the cooldown
        self.current_cooldown = 0

    def isConstruction_done(self):
        if not self.isConstructing:
            return

        if self.current_construction_progress >= self.construction_period_timesteps:
            # Reset the construction progress
            self.current_construction_progress = 0

            self.isConstructing = False
            return True

        return False

    def timestep(self):
        # Fill the construction progress if constructing
        if self.isConstructing:
            self.current_construction_progress += 1

        # Charge the construction cooldown while not constructing
        if not self.isConstructing:
            self.current_cooldown += 1
