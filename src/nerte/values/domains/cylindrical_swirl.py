"""Module for cylindrical swirl manifold domains."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains import Domain3D


class CylindricalSwirlDomain(Domain3D):
    """Domain of cylindrical swirl coordinates for fixed swirl value."""

    def __init__(self, swirl: float) -> None:

        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cylindrical swirl domain with swirl={swirl}."
                f" Value must be finite."
            )

        self.swirl = swirl

    def are_inside(self, coords: Coordinates3D) -> bool:
        # pylint: disable=C0103
        r, alpha, z = coords
        return (
            0 < r < math.inf
            and -math.pi < alpha + self.swirl * r * z < math.pi
            and math.isfinite(z)
        )

    def not_inside_reason(self, coords: Coordinates3D) -> str:
        return (
            f"Coordinates (r, alpha, z)={coords} are not inside domain of"
            f" the cylindrical swirl={self.swirl } domain."
            f" The following conditions must be met:"
            f" 0 < r < inf"
            f" and -pi < alpha + swirl * r * z < pi"
            f" and -inf < z < inf"
        )
