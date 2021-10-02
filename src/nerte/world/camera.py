""""Module for representing a camera."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.manifolds.chart_2_to_3 import Chart2DTo3D


class Camera:
    # pylint: disable=R0903
    """
    Represenation of a camera connecting the world to the screen.

    Each camera defines its properties in the world via its detector surface
    (manifold) and on the screen via its canvas dimensions.
    """

    def __init__(
        self,
        location: Coordinates3D,
        detector_manifold: Chart2DTo3D,
        canvas_dimensions: tuple[int, int],
    ) -> None:
        self.location = location
        self.detector_manifold = detector_manifold
        self.canvas_dimensions = canvas_dimensions
