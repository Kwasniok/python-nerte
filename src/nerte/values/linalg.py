# pylint: disable=W0212

"""Module for vector representation and operations."""

from typing import Optional

import numpy as np


class AbstractVector:
    """
    Represents an abstract vector via its three real coefficients.
    Note: The basis of the vector space and its kindness (co- or contra-variant)
          must be implicitly given by the context the vector is used in.
    """

    def __init__(self, coeffs: tuple[float, float, float]) -> None:
        self._v = np.array(coeffs)

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


class AbstractMatrix:
    """
    Represents an abstract matrix via three vectors.
    Note: The basis of the vector space and the kindness (co- or contra-variant)
          of each rank must be implicitly given by the context the vector is
          used in.
    """

    def __init__(
        self, vec0: AbstractVector, vec1: AbstractVector, vec2: AbstractVector
    ) -> None:
        self._m = np.array((vec0._v, vec1._v, vec2._v))

    def __repr__(self) -> str:
        return "M(" + (",".join(repr(x) for x in self._m)) + ")"

    def __add__(self, other: "AbstractMatrix") -> "AbstractMatrix":
        return _abstract_matrix_from_numpy(self._m + other._m)

    def __neg__(self) -> "AbstractMatrix":
        return _abstract_matrix_from_numpy(-1 * self._m)

    def __sub__(self, other: "AbstractMatrix") -> "AbstractMatrix":
        return _abstract_matrix_from_numpy(self._m - other._m)

    def __mul__(self, fac: float) -> "AbstractMatrix":
        return _abstract_matrix_from_numpy(fac * self._m)

    def __truediv__(self, fac: float) -> "AbstractMatrix":
        return _abstract_matrix_from_numpy((1 / fac) * self._m)

    def __getitem__(self, i: int) -> AbstractVector:
        return _abstract_vector_from_numpy(self._m[i])

    def is_symmetric(self) -> bool:
        """
        Returns True, iff matrix is symmetric.

        Note: Small numerical deviations of the coefficients are allowed.
        """
        return (
            np.isclose(self._m[0][1], self._m[1][0])
            and np.isclose(self._m[0][2], self._m[2][0])
            and np.isclose(self._m[1][2], self._m[2][1])
        )


def _abstract_matrix_from_numpy(np_array: np.ndarray) -> AbstractMatrix:
    """
    Auxiliar function to wrap an np.array into a matrix.
    Note: For internal usage only! The input is trusted to be valid and no
          checks are applied.
    """
    mat = AbstractMatrix.__new__(AbstractMatrix)
    mat._m = np_array
    return mat


class AbstractSymmetricMatrix(AbstractMatrix):
    # pylint: disable=R0903
    """
    Represents an abstract symmetric matrix via three vectors.
    Note: The basis of the vector space and the kindness (co- or contra-variant)
          of each rank must be implicitly given by the context the vector is
          used in.
    """

    def __init__(
        self, vec0: AbstractVector, vec1: AbstractVector, vec2: AbstractVector
    ) -> None:
        if (
            not np.isclose(vec0[1], vec1[0])
            or not np.isclose(vec0[2], vec2[0])
            or not np.isclose(vec1[2], vec2[1])
        ):
            raise ValueError(
                f"Cannot construct a symmetric matrix for given vectors."
                f" Vectors given are {vec0}, {vec1} and {vec2}"
            )

        AbstractMatrix.__init__(self, vec0, vec1, vec2)


def _abstract_symmetric_matrix_from_numpy(
    np_array: np.ndarray,
) -> AbstractSymmetricMatrix:
    """
    Auxiliar function to wrap an np.array into a symmetric matrix.
    Note: For internal usage only! The input is trusted to be valid and no
    checks are applied.
    """
    mat = AbstractSymmetricMatrix.__new__(AbstractMatrix)
    mat._m = np_array
    return mat


def to_symmetric_matrix(matrix: AbstractMatrix) -> AbstractSymmetricMatrix:
    """
    Declares a matrix to be symmetric.
    :raises: ValueError, iff matrix is not symmetric
    """
    if not matrix.is_symmetric():
        raise ValueError(f"Matrix {matrix} cannot be declared symmetic.")
    return _abstract_symmetric_matrix_from_numpy(matrix._m)


class Metric:
    """
    Represents a metric as a matrix acting on contravariant representations of
    vectors and returning covariant representations of them.
    Note: A metric must be invertible.
    """

    def __init__(self, matrix: AbstractSymmetricMatrix) -> None:
        rank = np.linalg.matrix_rank(
            matrix._m,
            hermitian=True,
        )  # type: ignore[no-untyped-call]
        if rank != 3:
            raise ValueError(
                f"Cannot construct metric form non-invertible symmetric matrix"
                f" {matrix} - its rank is {rank}."
            )

        self._g = matrix
        self._g_inv: Optional[AbstractSymmetricMatrix] = None

    def matrix(self) -> AbstractSymmetricMatrix:
        """Returns the metric as a matrix."""
        return self._g

    def inverse_matrix(self) -> AbstractSymmetricMatrix:
        """Returns the inverse of the metric as a matrix."""
        if self._g_inv is None:
            try:
                self._g_inv = to_symmetric_matrix(inverted(self._g))
            except ArithmeticError as ex:
                raise ValueError(
                    "Cannot construct a metric from non-invertible matrix."
                ) from ex
        return self._g_inv


def covariant(metric: Metric, contra_vec: AbstractVector) -> AbstractVector:
    """
    Returns the co-variant vector.

    :param metric: metric of the tangential vector space
    :param contra_vec: contra-variant vector of the tangential vetor space
    """

    return _abstract_vector_from_numpy(
        np.dot(metric.matrix()._m, contra_vec._v)  # type: ignore[no-untyped-call]
    )


def contravariant(metric: Metric, co_vec: AbstractVector) -> AbstractVector:
    """
    Returns the contra-variant vector.

    :param inv_metric: inverse of the metric of the tangential vector space
    :param contra_vec: co-variant vector of the tangential vetor space
    """
    return _abstract_vector_from_numpy(
        np.dot(metric.inverse_matrix()._m, co_vec._v)  # type: ignore[no-untyped-call]
    )


def is_zero_vector(vec: AbstractVector) -> bool:
    """Retruns True, iff the vector is a zero vector."""
    return vec[0] == vec[1] == vec[2] == 0.0


def dot(
    vec1: AbstractVector,
    vec2: AbstractVector,
    metric: Optional[Metric] = None,
) -> float:
    """Returns the (orthonormal) dot product of both vectors."""
    if metric is None:
        # NOTE: SMALL performance improvments with hardcoded version!
        return (
            vec1._v[0] * vec2._v[0]
            + vec1._v[1] * vec2._v[1]
            + vec1._v[2] * vec2._v[2]
        )
        # NOTE: DON'T use this:
        # return np.dot(vec1._v, vec2._v)
    return np.dot(
        vec1._v,
        np.dot(metric.matrix()._m, vec2._v),  # type: ignore[no-untyped-call]
    )  # TODO: optimize


# TODO: include metric and transformation
def cross(
    vec1: AbstractVector,
    vec2: AbstractVector,
) -> AbstractVector:
    """Returns the (orthonormal) cross product of both vectors."""
    # NOTE: MASSIVE performance improvments with hardcoded version!
    return AbstractVector(
        (
            vec1._v[1] * vec2._v[2] - vec1._v[2] * vec2._v[1],
            vec1._v[2] * vec2._v[0] - vec1._v[0] * vec2._v[2],
            vec1._v[0] * vec2._v[1] - vec1._v[1] * vec2._v[0],
        )
    )
    # NOTE: DON'T use this:
    # return Vector.__from_numpy(np.cross(vec1._v, vec2._v))


def length(vec: AbstractVector, metric: Optional[Metric] = None) -> float:
    """
    Returns the length of the vector (with respect to an orthonormal basis).
    """
    if metric is None:
        return np.linalg.norm(vec._v)  # type: ignore[no-untyped-call]
    return dot(vec, vec, metric) ** 0.5  # TODO: optimize


def normalized(
    vec: AbstractVector, metric: Optional[Metric] = None
) -> AbstractVector:
    """
    Returns the normalized vector (with respect to an orthonormal basis).
    """
    if metric is None:
        # NOTE: VERY SMALL performance improvments with hardcoded version!
        return _abstract_vector_from_numpy(vec._v * (dot(vec, vec) ** -0.5))
        # NOTE: DON'T use this:
        # return Vector.__from_numpy((1 / np.linalg.norm(vec._v)) * vec._v)
    return vec / length(vec, metric)  # TODO: optimize


def inverted(mat: AbstractMatrix) -> AbstractMatrix:
    """
    Returns the inverse of a matrix.
    :raises: ArithmeticError
    """
    try:
        mat = _abstract_matrix_from_numpy(
            np.linalg.inv(mat._m)  # type: ignore[no-untyped-call]
        )
    except np.linalg.LinAlgError as ex:
        raise ArithmeticError from ex
    else:
        return mat
