"""Module for representing manifolds in cylindrical swirl coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
)
from nerte.values.domains import Domain3D


class CylindricalSwirlDomain(Domain3D):
    """Domain of cylindrical swirl coordinates for fixed swirl value."""

    def __init__(self, swirl: float) -> None:
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


def _metric(swirl: float, coords: Coordinates3D) -> AbstractMatrix:
    """
    Returns the metric for the cylindrical swirl coordiantes (r, 𝛼, z).

    Note: No checks are performed. It is trusted that:
        0 < r < inf
        -pi < 𝛼 + swirl * r * z < pi
        -inf < z < inf
    """
    # pylint: disable=C0103
    a = swirl
    r, _, z = coords
    return AbstractMatrix(
        AbstractVector(
            (1 + (a * r * z) ** 2, a * r ** 2 * z, a ** 2 * r ** 3 * z)
        ),
        AbstractVector((a * r ** 2 * z, r ** 2, a * r ** 3)),
        AbstractVector((a ** 2 * r ** 3 * z, a * r ** 3, 1 + a ** 2 * r ** 4)),
    )


def metric(swirl: float, coords: Coordinates3D) -> Metric:
    """Returns the local metric in cylindrical swirl coordinates."""
    return Metric(_metric(swirl, coords))


def geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cylindrical swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2)
    - meaning the local rate of change in sapce and velocity.
    """
    # pylint: disable=C0103
    a = swirl
    r, _, z = tangent.point
    v_r, v_alpha, v_z = tangent.vector[0], tangent.vector[1], tangent.vector[2]
    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector(
            (
                r * (a * z * v_r + a * r * v_z + v_alpha) ** 2,
                -(
                    (2 * v_r * v_alpha) / r
                    + 2 * a ** 2 * r * v_alpha * z * (r * v_z + v_r * z)
                    + a ** 3 * r * z * (r * v_z + v_r * z) ** 2
                    + a
                    * (
                        4 * v_r * v_z
                        + (2 * v_r ** 2 * z) / r
                        + r * v_alpha ** 2 * z
                    )
                ),
                0,
            )
        ),
    )
