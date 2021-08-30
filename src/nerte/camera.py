""""Module for representing a camera."""

from nerte.values.coordinates import Coordinates
from nerte.values.linalg import AbstractVector


class Camera:
    # pylint: disable=R0903
    """Represenation of a camera.
    Each camera defines its properties in the world via its detector and on the
    screen via its canvas.
    """

    def __init__(
        self,
        location: Coordinates,
        direction: AbstractVector,
        detector_manifold: tuple[
            AbstractVector,
            AbstractVector,
        ],  # TODO: must be generalized!
        canvas_dimensions: tuple[int, int],
    ) -> None:
        self.location = location
        self.direction = direction
        self.detector_manifold = detector_manifold
        self.canvas_dimensions = canvas_dimensions
