""""Module for representing a camera."""

from nerte.values.coordinates import Coordinates
from nerte.values.manifold import Manifold2D


class Camera:
    # pylint: disable=R0903
    """Represenation of a camera.
    Each camera defines its properties in the world via its detector and on the
    screen via its canvas.
    """

    def __init__(
        self,
        location: Coordinates,
        detector_manifold: Manifold2D,
        detector_manifold_ranges: tuple[
            tuple[float, float], tuple[float, float]
        ],
        canvas_dimensions: tuple[int, int],
    ) -> None:
        self.location = location
        self.detector_manifold = detector_manifold
        self.detector_manifold_ranges = detector_manifold_ranges
        self.canvas_dimensions = canvas_dimensions
