# pylint: disable=W0212

"""Module for vector representation and operations."""

import numpy as np


class AbstractVector:
    """
    Represents an abstract vector via its three real coefficients.
    Note: The basis of the vectorspace must be implicitly given by the contex
          the vector is used in.
    """

    def __init__(self, v1: float, v2: float, v3: float) -> None:
        self._v = np.array([v1, v2, v3])

    def __repr__(self) -> str:
        return "V(" + (",".join(repr(x) for x in self._v)) + ")"

    def __add__(self, other: "AbstractVector") -> "AbstractVector":
        return _abstract_vector_from_numpy(self._v + other._v)

    def __neg__(self) -> "AbstractVector":
        return _abstract_vector_from_numpy(-1 * self._v)

    def __sub__(self, other: "AbstractVector") -> "AbstractVector":
        return _abstract_vector_from_numpy(self._v - other._v)

    def __mul__(self, fac: float) -> "AbstractVector":
        return _abstract_vector_from_numpy(fac * self._v)

    def __truediv__(self, fac: float) -> "AbstractVector":
        return _abstract_vector_from_numpy((1 / fac) * self._v)

    def __getitem__(self, i: int) -> float:
        return self._v[i]


def _abstract_vector_from_numpy(np_array: np.ndarray) -> AbstractVector:
    """
    Auxiliar function to wrap an np.array into a vetor.
    Note: For internal usage only! The input is trusted to be valid and no
          checks are applied.
    """
    vec = AbstractVector.__new__(AbstractVector)
    vec._v = np_array
    return vec


class ContraVector(AbstractVector):
    """
    Represents a contra-variant vector via its three real coefficients.
    Note: The basis of the vectorspace must be implicitly given by the contex
          the vector is used in.
    """

    # pylint: disable=R0903,W0107
    pass


class CoVector(AbstractVector):
    """
    Represents a co-variant vector via its three real coefficients.
    Note: The basis of the vectorspace must be implicitly given by the contex
          the vector is used in.
    """

    # pylint: disable=R0903,W0107
    pass


def dot(vec1: AbstractVector, vec2: AbstractVector) -> float:
    """Returns the (orthonormal) dot product of both vectors."""
    # NOTE: SMALL performance improvments with hardcoded version!
    return (
        vec1._v[0] * vec2._v[0]
        + vec1._v[1] * vec2._v[1]
        + vec1._v[2] * vec2._v[2]
    )
    # NOTE: DON'T use this:
    # return np.dot(vec1._v, vec2._v)


def cross(vec1: AbstractVector, vec2: AbstractVector) -> AbstractVector:
    """Returns the (orthonormal) cross product of both vectors."""
    # NOTE: MASSIVE performance improvments with hardcoded version!
    return AbstractVector(
        vec1._v[1] * vec2._v[2] - vec1._v[2] * vec2._v[1],
        vec1._v[2] * vec2._v[0] - vec1._v[0] * vec2._v[2],
        vec1._v[0] * vec2._v[1] - vec1._v[1] * vec2._v[0],
    )
    # NOTE: DON'T use this:
    # return Vector.__from_numpy(np.cross(vec1._v, vec2._v))


def length(vec: AbstractVector) -> float:
    """
    Returns the length of the vector (with respect to an orthonormal basis).
    """
    return np.linalg.norm(vec._v)


def normalized(vec: AbstractVector) -> AbstractVector:
    """
    Returns the normalized vector (with respect to an orthonormal basis).
    """
    # NOTE: VERY SMALL performance improvments with hardcoded version!
    return _abstract_vector_from_numpy(vec._v * (dot(vec, vec) ** -0.5))
    # NOTE: DON'T use this:
    # return Vector.__from_numpy((1 / np.linalg.norm(vec._v)) * vec._v)
