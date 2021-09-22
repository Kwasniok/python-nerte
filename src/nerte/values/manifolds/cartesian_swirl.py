"""Module for representing manifolds in cartesian swirl coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates1D, Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    is_zero_vector,
    cross,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.linalg import AbstractMatrix, Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.manifold import Manifold1D, Manifold2D, Manifold3D


def carthesian_swirl_metric(swirl: float, coords: Coordinates3D) -> Metric:
    # pylint: disable=C0103
    """Returns the local metric in carthesian swirl coordinates."""
    a = swirl
    x, y, z = coords
    r = math.sqrt(x ** 2 + y ** 2)
    if r == 0:
        raise ValueError(
            f"Cannot generate matric for cartesian swirl={swirl} coordinates"
            f" at (x, y, z)={coords}."
            f" Coordinate values must be restricted to "
            f" 0 < r = sqrt(x ** 2 + y ** 2)."
        )

    # frequent factors
    axyz2 = 2 * a * x * y * z
    r2z2 = r ** 2 + z ** 2
    R = x ** 2 - y ** 2
    u = -((a * (R * z + a * r * x * y * r2z2)) / r)
    w = a ** 2 * r2z2
    ary = a * r * y
    arx = a * r * x

    return Metric(
        AbstractMatrix(
            AbstractVector((1 + axyz2 / r + w * y ** 2, u, ary)),
            AbstractVector((u, 1 - axyz2 / r + w * x ** 2, -arx)),
            AbstractVector((ary, -arx, 1)),
        )
    )


def carthesian_swirl_geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    # pylint: disable=C0103,R0914
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    carthesian swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2).
    """
    a = swirl
    x, y, z = tangent.point
    vx, vy, vz = tangent.vector[0], tangent.vector[1], tangent.vector[2]

    # radial factors
    r = math.sqrt(x ** 2 + y ** 2)
    if r == 0:
        raise ValueError(
            f"Cannot generate geodesic equation for cartesian swirl={swirl}"
            f" coordinates at (x, y, z)={tangent.point}."
            f" Coordinate values must be restricted to "
            f" 0 < r = sqrt(x ** 2 + y ** 2)."
        )
    R = 1 / r

    # frequent factors
    uyz = vy * vz
    uxz = vx * vz

    # polynomial factors for vx
    faR_x = 2 * uyz * x ** 2 + 2 * uxz * x * y + 4 * uyz * y ** 2
    faR3_x = (
        2 * vx * vy * x ** 3 * z
        + 3 * vy ** 2 * x ** 2 * y * z
        + (vx ** 2 + 2 * vy ** 2) * y ** 3 * z
    )
    fa2r2_x = vz ** 2 * x
    fa2R2_x = (
        2 * uxz * x ** 4 * z
        + 4 * uyz * x ** 3 * y * z
        + 4 * uyz * x * y ** 3 * z
        - 2 * uxz * y ** 4 * z
    )
    fa2R4_x = (
        vx ** 2 * x ** 5 * z ** 2
        + 4 * vx * vy * x ** 4 * y * z ** 2
        + (-(vx ** 2) + 3 * vy ** 2) * x ** 3 * y ** 2 * z ** 2
        + 2 * vx * vy * x ** 2 * y ** 3 * z ** 2
        + (-2 * vx ** 2 + 3 * vy ** 2) * x * y ** 4 * z ** 2
        - 2 * vx * vy * y ** 5 * z ** 2
    )
    fa3r3_x = vz ** 2 * y * z
    fa3R_x = (
        2 * uxz * x ** 3 * y * z ** 2
        + 2 * uyz * x ** 2 * y ** 2 * z ** 2
        + 2 * uxz * x * y ** 3 * z ** 2
        + 2 * uyz * y ** 4 * z ** 2
    )
    fa3R3_x = (
        vx ** 2 * x ** 4 * y * z ** 3
        + 2 * vx * vy * x ** 3 * y ** 2 * z ** 3
        + (vx ** 2 + vy ** 2) * x ** 2 * y ** 3 * z ** 3
        + 2 * vx * vy * x * y ** 4 * z ** 3
        + vy ** 2 * y ** 5 * z ** 3
    )

    # polynomial factors for vy
    faR_y = -4 * uxz * x ** 2 - 2 * uyz * x * y - 2 * uxz * y ** 2
    faR3_y = (
        (-2 * vx ** 2 - vy ** 2) * x ** 3 * z
        - 3 * vx ** 2 * x * y ** 2 * z
        - 2 * vx * vy * y ** 3 * z
    )
    fa2r2_y = vz ** 2 * y
    fa2R2_y = (
        -2 * uyz * x ** 4 * z
        + 4 * uxz * x ** 3 * y * z
        + 4 * uxz * x * y ** 3 * z
        + 2 * uyz * y ** 4 * z
    )
    fa2R4_y = (
        -2 * vx * vy * x ** 5 * z ** 2
        + (3 * vx ** 2 - 2 * vy ** 2) * x ** 4 * y * z ** 2
        + 2 * vx * vy * x ** 3 * y ** 2 * z ** 2
        + (3 * vx ** 2 - vy ** 2) * x ** 2 * y ** 3 * z ** 2
        + 4 * vx * vy * x * y ** 4 * z ** 2
        + vy ** 2 * y ** 5 * z ** 2
    )
    fa3r3_y = -(vz ** 2) * x * z
    fa3R_y = (
        -2 * uxz * x ** 4 * z ** 2
        - 2 * uyz * x ** 3 * y * z ** 2
        - 2 * uxz * x ** 2 * y ** 2 * z ** 2
        - 2 * uyz * x * y ** 3 * z ** 2
    )
    fa3R3_y = (
        -(vx ** 2 * x ** 5 * z ** 3)
        - 2 * vx * vy * x ** 4 * y * z ** 3
        + (-(vx ** 2) - vy ** 2) * x ** 3 * y ** 2 * z ** 3
        - 2 * vx * vy * x ** 2 * y ** 3 * z ** 3
        - vy ** 2 * x * y ** 4 * z ** 3
    )

    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector(
            (
                a * (R * faR_x + R ** 3 * faR3_x)
                + a ** 2
                * (r ** 2 * fa2r2_x + R ** 2 * fa2R2_x + R ** 4 * fa2R4_x)
                + a ** 3 * (r ** 3 * fa3r3_x + R * fa3R_x + R ** 3 * fa3R3_x),
                a * (R * faR_y + R ** 3 * faR3_y)
                + a ** 2
                * (r ** 2 * fa2r2_y + R ** 2 * fa2R2_y + R ** 4 * fa2R4_y)
                + a ** 3 * (r ** 3 * fa3r3_y + R * fa3R_y + R ** 3 * fa3R3_y),
                0,
            )
        ),
    )
