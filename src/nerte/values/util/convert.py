""""Module for conversion between vectors and coordinates."""

from nerte.values.coordinates import Coordinates
from nerte.values.linalg import AbstractVector

# TODO: use this implementation everywhere


def coordinates_as_vector(coords: Coordinates) -> AbstractVector:
    """Returns reinterpretation of coordinates as a vector."""
    return AbstractVector(coords[0], coords[1], coords[2])


def vector_as_coordinates(vec: AbstractVector) -> Coordinates:
    """Returns reinterpretation of a vector as coordinates."""
    return Coordinates(vec[0], vec[1], vec[2])
