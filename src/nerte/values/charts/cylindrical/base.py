"""Base module for representing manifolds in cylindrical coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
)
from nerte.values.interval import Interval
from nerte.values.domains import CartesianProduct3D

DOMAIN = CartesianProduct3D(
    Interval(0, math.inf),
    Interval(-math.pi, math.pi),
    Interval(-math.inf, math.inf),
)


def metric(coords: Coordinates3D) -> Metric:
    """Returns the local metric for the given coordinates."""
    # pylint: disable=C0103
    r, _, _ = coords
    return Metric(
        AbstractMatrix(
            AbstractVector((1, 0, 0)),
            AbstractVector((0, r ** 2, 0)),
            AbstractVector((0, 0, 1)),
        )
    )


def geodesic_equation(
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cylindrical coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d^2x/d𝜆^2).
    """
    # pylint: disable=C0103
    r, _, _ = tangent.point
    v_r, v_phi, _ = tangent.vector[0], tangent.vector[1], tangent.vector[2]
    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector(
            (
                r * v_phi ** 2,
                -2 * v_r * v_phi / r,
                0,
            )
        ),
    )
