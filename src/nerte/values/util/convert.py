""""Module for conversion between vectors and coordinates."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractVector

# TODO: use this implementation everywhere


def coordinates_as_vector(coords: Coordinates3D) -> AbstractVector:
    """Returns reinterpretation of coordinates as a vector."""
    return AbstractVector(coords[0], coords[1], coords[2])


def vector_as_coordinates(vec: AbstractVector) -> Coordinates3D:
    """Returns reinterpretation of a vector as coordinates."""
    return Coordinates3D((vec[0], vec[1], vec[2]))
