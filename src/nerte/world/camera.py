""""Module for representing a camera."""

from typing import Optional

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains import Domain2D, CartesianProduct2D
from nerte.values.submanifolds import Submanifold2DIn3D


class Camera:
    # pylint: disable=R0903
    """
    Represenation of a camera connecting the world to the screen.

    Each camera defines its properties in the world via its detector surface
    (manifold) and on the screen via its canvas dimensions.
    """

    # pylint: disable=R0913
    def __init__(
        self,
        location: Coordinates3D,
        detector_domain: CartesianProduct2D,
        detector_manifold: Submanifold2DIn3D,
        canvas_dimensions: tuple[int, int],
        detector_domain_filter: Optional[Domain2D] = None,
    ) -> None:
        self.location = location
        self.detector_domain = detector_domain
        self.detector_domain_filter = detector_domain_filter
        self.detector_manifold = detector_manifold
        self.canvas_dimensions = canvas_dimensions
