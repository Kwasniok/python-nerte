""""Module for conversion between vectors and coordinates."""

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector

# TODO: use this implementation everywhere


def coordinates_as_vector(coords: Coordinates3D) -> AbstractVector:
    """Returns reinterpretation of coordinates as a vector."""
    return AbstractVector(coords)


def vector_as_coordinates(vec: AbstractVector) -> Coordinates3D:
    """Returns reinterpretation of a vector as coordinates."""
    return Coordinates3D((vec[0], vec[1], vec[2]))


def carthesian_to_cylindric_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns cylindrical coordinates obtained from carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
    """
    # pylint:disable=C0103
    x, y, z = coords
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    r = math.sqrt(x ** 2 + y ** 2)
    phi = math.atan2(y, x)
    return Coordinates3D((r, phi, z))


def carthesian_to_cylindric_vector(
    coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of cylindircal coordinates from
    a vector in tangential vector space in carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
    :param vec: vector in tangential vector space of the carthesian coordinates
                (x, y, z) such that vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    x, y, z = coords
    assert -math.inf < x < math.inf, f"{x} is out of bounds"
    assert -math.inf < y < math.inf, f"{y} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    phi = math.atan2(y, x)
    e_x = AbstractVector((math.cos(phi), -math.sin(phi), 0.0))
    e_y = AbstractVector((math.sin(phi), math.cos(phi), 0.0))
    e_z = AbstractVector((0.0, 0.0, 1.0))
    return e_x * vec[0] + e_y * vec[1] + e_z * vec[2]


def cylindric_to_carthesian_coords(coords: Coordinates3D) -> Coordinates3D:
    """
    Returns carthesian coordinates obtained from cylindrical coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and 0 < phi < 2*pi and -inf < z < inf
    """
    # pylint:disable=C0103
    r, phi, z = coords
    assert 0 < r < math.inf, f"{r} is out of bounds"
    assert 0 < phi < 2 * math.pi, f"{phi} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    return Coordinates3D((x, y, z))


def cylindric_to_carthesian_vector(
    coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of carthesian coordinates from
    a vector in tangential vector space in cylindircal coordinates.

    :param coords: cylindrical coordinates (r, phi, z)
                   where 0 < r < inf and 0 < phi < 2*pi and -inf < z < inf
    :param vec: vector in tangential vector space of the cylindircal coordinates
                (r, phi, z) such that vec = e_r * r + e_phi * phi + e_z * z
    """
    # pylint:disable=C0103
    r, phi, z = coords
    assert 0 < r < math.inf, f"{r} is out of bounds"
    assert 0 < phi < 2 * math.pi, f"{phi} is out of bounds"
    assert -math.inf < z < math.inf, f"{z} is out of bounds"
    e_r = AbstractVector((math.cos(phi), math.sin(phi), 0.0))
    e_phi = AbstractVector((-math.sin(phi), math.cos(phi), 0.0))
    e_z = AbstractVector((0.0, 0.0, 1.0))
    return e_r * vec[0] + e_phi * vec[1] + e_z * vec[2]
