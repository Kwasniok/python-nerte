"""Module for cartesian swirl manifold domains."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.domains import Domain3D


class CartesianSwirlDomain(Domain3D):
    """Domain of cartesian swirl coordinates for fixed swirl value."""

    def __init__(self, swirl: float) -> None:

        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cylindrical swirl domain with swirl={swirl}."
                f" Value must be finite."
            )

        self.swirl = swirl

    def are_inside(self, coords: Coordinates3D) -> bool:
        # pylint: disable=C0103
        u, v, z = coords
        return 0 < abs(u) + abs(v) < math.inf and math.isfinite(z)

    def not_inside_reason(self, coords: Coordinates3D) -> str:
        return (
            f"Coordinates (u, v, z)={coords} are not inside domain of"
            f" the cartesian swirl={self.swirl } domain."
            f" The following conditions must be met:"
            f" 0 < abs(u) + abs(v) < inf and -inf < z < inf"
        )
