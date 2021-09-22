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
