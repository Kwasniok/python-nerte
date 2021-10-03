""""Module for representing a camera."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains import CartesianProduct2D
from nerte.values.charts import Chart2DTo3D


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
        detector_domain: CartesianProduct2D,
        detector_manifold: Chart2DTo3D,
        canvas_dimensions: tuple[int, int],
    ) -> None:
        self.location = location
        self.detector_domain = detector_domain
        self.detector_manifold = detector_manifold
        self.canvas_dimensions = canvas_dimensions
