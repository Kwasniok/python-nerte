# pylint: disable=W0212

import numpy as np


class Vector:
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

    def dot(self, other: "Vector") -> "Vector":
        return np.dot(self._v, other._v)

    def cross(self, other: "Vector") -> "Vector":
        return Vector.__from_numpy(np.cross(self._v, other._v))

    def length(self) -> float:
        return np.linalg.norm(self._v)

    def normalized(self) -> "Vector":
        return Vector.__from_numpy((1 / np.linalg.norm(self._v)) * self._v)
