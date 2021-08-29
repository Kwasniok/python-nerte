# pylint: disable=W0212

"""Module for vector representation and operations."""

import numpy as np


# TODO: separate tdot,cross etc. from this class
class Vector:
    """
    Represents a vector as three real coefficients.
    The basis is implicitly assumed to be orthonormal.
    Note: All vector operations are with respect to an orthonormal basis
          but they might be interpreted with respect to a non orthogonal
          basis as well! Therefore, use the methods dot, cross, length and
          normalize with caution.
    """

    def __init__(self, v1: float, v2: float, v3: float):
        self._v = np.array([v1, v2, v3])

    @classmethod
    def __from_numpy(cls, np_array) -> "Vector":
        vec = Vector.__new__(Vector)
        vec._v = np_array
        return vec

    def __repr__(self):
        return "V(" + (",".join(repr(x) for x in self._v)) + ")"

    def __add__(self, other: "Vector") -> "Vector":
        return Vector.__from_numpy(self._v + other._v)

    def __neg__(self) -> "Vector":
        return Vector.__from_numpy(-1 * self._v)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector.__from_numpy(self._v - other._v)

    def __mul__(self, fac: float) -> "Vector":
        return Vector.__from_numpy(fac * self._v)

    def __truediv__(self, fac: float) -> "Vector":
        return Vector.__from_numpy((1 / fac) * self._v)

    def __getitem__(self, i: int) -> float:
        return self._v[i]

    #       as they only apply for orthonormal spaces
    def dot(self, other: "Vector") -> "Vector":
        """Returns the (orthonormal) dot product of both vectors."""
        # NOTE: SMALL performance improvments with hardcoded version!
        return (
            self._v[0] * other._v[0]
            + self._v[1] * other._v[1]
            + self._v[2] * other._v[2]
        )
        # NOTE: DON'T use this:
        # return np.dot(self._v, other._v)

    def cross(self, other: "Vector") -> "Vector":
        """Returns the (orthonormal) cross product of both vectors."""
        # NOTE: MASSIVE performance improvments with hardcoded version!
        return Vector(
            self._v[1] * other._v[2] - self._v[2] * other._v[1],
            self._v[2] * other._v[0] - self._v[0] * other._v[2],
            self._v[0] * other._v[1] - self._v[1] * other._v[0],
        )
        # NOTE: DON'T use this:
        # return Vector.__from_numpy(np.cross(self._v, other._v))

    def length(self) -> float:
        """
        Returns the length of the vector (with respect to an orthonormal basis).
        """
        return np.linalg.norm(self._v)

    def normalized(self) -> "Vector":
        """
        Returns the normalized vector (with respect to an orthonormal basis).
        """
        # NOTE: VERY SMALL performance improvments with hardcoded version!
        length = self.dot(self) ** -0.5
        return Vector.__from_numpy(self._v * length)
        # NOTE: DON'T use this:
        # return Vector.__from_numpy((1 / np.linalg.norm(self._v)) * self._v)
