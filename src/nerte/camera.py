from nerte.coordinates import Coordinates
from nerte.vector import Vector


class Camera:
    def __init__(
        self,
        location: Coordinates,
        direction: Vector,
        detector_manifold: (Vector, Vector),  # TODO: must be generalized!
        canvas_dimensions: (int, int),
    ):
        self.location = location
        self.direction = direction
        self.detector_manifold = detector_manifold
        self.canvas_dimensions = canvas_dimensions
