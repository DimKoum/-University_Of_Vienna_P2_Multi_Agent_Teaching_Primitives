
class Movable2DDiscrete:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def moveLeft(self):
        self.x -= 1

    def moveRight(self):
        self.x += 1

    def moveDown(self):
        self.y += 1

    def moveUp(self):
        self.y -= 1