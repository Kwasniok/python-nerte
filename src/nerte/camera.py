""""Module for representing a camera."""

from nerte.geometry.coordinates import Coordinates
from nerte.geometry.vector import Vector


class Camera:
    # pylint: disable=R0903
    """Represenation of a camera.
    Each camera defines its properties in the world via its detector and on the
    screen via its canvas.
    """

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
