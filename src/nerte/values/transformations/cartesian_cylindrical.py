"""
Module for a transformation mediating between cartesian and cylindrical
coordinates.
"""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    mat_vec_mult,
)


def cartesian_to_cylindrical_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns cylindrical coordinates obtained from cartesian coordinates.

    :param coords: cartesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    """
    # pylint:disable=C0103
    x, y, z = coords
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    return Coordinates3D((r, phi, z))


def cylindrical_to_cartesian_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns cartesian coordinates obtained from cylindrical coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and -pi < phi < pi and -inf < z < inf
    """
    # pylint:disable=C0103
    r, phi, z = coords
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return Coordinates3D((x, y, z))


def cartesian_to_cylindrical_vector(
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cartesian to cylindrical
    coordinates.
    """
    # pylint:disable=C0103
    x, y, z = tangential_vector.point
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), math.sin(phi), 0.0)),
        AbstractVector((-math.sin(phi) / r, math.cos(phi) / r, 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return TangentialVector(
        point=Coordinates3D((r, phi, z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )


def cylindrical_to_cartesian_vector(
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from cylindrical to cartesian
    coordinates.
    """
    # pylint:disable=C0103
    r, phi, z = tangential_vector.point
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    jacobian = AbstractMatrix(
        AbstractVector((math.cos(phi), -r * math.sin(phi), 0.0)),
        AbstractVector((math.sin(phi), r * math.cos(phi), 0.0)),
        AbstractVector((0.0, 0.0, 1.0)),
    )
    return TangentialVector(
        point=Coordinates3D((x, y, z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )
