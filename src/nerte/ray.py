from nerte.coordinates import Coordinates
from nerte.vector import Vector


class Ray:
    def __init__(self, start: Coordinates, direction: Vector):
        self.start = start
        self.direction = direction

    def __repr__(self):
        return "Ray({}, {})".format(repr(self.start), repr(self.direction))
