from nerte.coordinates import Coordinates
from nerte.vector import Vector


class Camera:
    def __init__(
        self,
        location: Coordinates,
        direction: Vector,
        width_vector: Vector,
        height_vector: Vector,
        canvas_width: int,
        canvas_height: int,
    ):
        self.location = location
        self.direction = direction
        self.width_vector = width_vector
        self.height_vector = height_vector
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def canvas_size(self):
        return self.canvas_width, self.canvas_height
