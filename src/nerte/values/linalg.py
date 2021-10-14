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


ZERO_VECTOR = AbstractVector((0.0, 0.0, 0.0))
UNIT_VECTOR0 = AbstractVector((1.0, 0.0, 0.0))
UNIT_VECTOR1 = AbstractVector((0.0, 1.0, 0.0))
UNIT_VECTOR2 = AbstractVector((0.0, 0.0, 1.0))
STANDARD_BASIS = (UNIT_VECTOR0, UNIT_VECTOR1, UNIT_VECTOR2)


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
        self._is_invertible: Optional[bool] = None  # cache

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

    def is_invertible(self) -> bool:
        """
        Returns True, iff matrix is invertible.
        """
        # NOTE: Calculating the determinant is faster than calculating the rank!
        if self._is_invertible is None:
            # NOTE: np.isclose is significantly slower
            self._is_invertible = (
                np.linalg.det(self._m) != 0.0  # type: ignore[no-untyped-call]
            )
        return self._is_invertible


def _abstract_matrix_from_numpy(np_array: np.ndarray) -> AbstractMatrix:
    """
    Auxiliar function to wrap an np.array into a matrix.
    Note: For internal usage only! The input is trusted to be valid and no
          checks are applied.
    """
    mat = AbstractMatrix.__new__(AbstractMatrix)
    mat._m = np_array
    mat._is_invertible = None
    return mat


ZERO_MATRIX = AbstractMatrix(ZERO_VECTOR, ZERO_VECTOR, ZERO_VECTOR)
IDENTITY_MATRIX = AbstractMatrix(UNIT_VECTOR0, UNIT_VECTOR1, UNIT_VECTOR2)

# TODO: remove dedicated class
class Metric:
    """
    Represents a metric as a matrix acting on contravariant representations of
    vectors and returning covariant representations of them.
    Note: A metric must be invertible.
    """

    def __init__(self, matrix: AbstractMatrix) -> None:

        if not matrix.is_invertible():
            raise ValueError(
                f"Cannot construct metric form non-invertible symmetric matrix"
                f" {matrix}."
            )

        self._g = matrix
        self._g_inv: Optional[AbstractMatrix] = None  # cache

    def __repr__(self) -> str:
        return repr(self._g)

    def matrix(self) -> AbstractMatrix:
        """Returns the metric as a matrix."""
        return self._g

    def inverse_matrix(self) -> AbstractMatrix:
        """Returns the inverse of the metric as a matrix."""
        if self._g_inv is None:
            try:
                self._g_inv = inverted(self._g)
            except ArithmeticError as ex:
                raise ValueError(
                    "Cannot construct a metric from non-invertible matrix."
                ) from ex
        return self._g_inv


IDENTITY_METRIC = Metric(IDENTITY_MATRIX)


class Rank3Tensor:
    """
    Represents an abstract rank three tensor.
    Note: The basis of the vector space and the kindness (co- or contra-variant)
          of each rank must be implicitly given by the context the vector is
          used in.
    """

    def __init__(
        self, mat0: AbstractMatrix, mat1: AbstractMatrix, mat2: AbstractMatrix
    ) -> None:
        self._data = np.array((mat0._m, mat1._m, mat2._m))

    def __repr__(self) -> str:
        return f"T3({self._data})"

    def __add__(self, other: "Rank3Tensor") -> "Rank3Tensor":
        return _rank_3_tensor_from_numpy(self._data + other._data)

    def __neg__(self) -> "Rank3Tensor":
        return _rank_3_tensor_from_numpy(-self._data)

    def __sub__(self, other: "Rank3Tensor") -> "Rank3Tensor":
        return _rank_3_tensor_from_numpy(self._data - other._data)

    def __mul__(self, fac: float) -> "Rank3Tensor":
        return _rank_3_tensor_from_numpy(self._data * fac)

    def __truediv__(self, fac: float) -> "Rank3Tensor":
        return _rank_3_tensor_from_numpy(self._data / fac)

    def __getitem__(self, i: int) -> AbstractMatrix:
        return _abstract_matrix_from_numpy(self._data[i])


def _rank_3_tensor_from_numpy(np_array: np.ndarray) -> Rank3Tensor:
    """
    Auxiliar function to wrap an np.array into a rank 3 tensor.
    Note: For internal usage only! The input is trusted to be valid and no
          checks are applied.
    """
    ten = Rank3Tensor.__new__(Rank3Tensor)
    ten._data = np_array
    return ten


ZERO_RANK3TENSOR = Rank3Tensor(ZERO_MATRIX, ZERO_MATRIX, ZERO_MATRIX)
IDENTITY_RANK3TENSOR = Rank3Tensor(
    IDENTITY_MATRIX, IDENTITY_MATRIX, IDENTITY_MATRIX
)


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


def mat_vec_mult(
    matrix: AbstractMatrix, vector: AbstractVector
) -> AbstractVector:
    """Return the product p = m.v of the matrix m and the vector v."""
    return _abstract_vector_from_numpy(
        np.dot(matrix._m, vector._v)  # type: ignore[no-untyped-call]
    )


def mat_mult(
    matrix1: AbstractMatrix, matrix2: AbstractMatrix
) -> AbstractMatrix:
    """Return the product m3 = m1.m2 of the matrices m1 and m2."""
    return _abstract_matrix_from_numpy(
        np.dot(matrix1._m, matrix2._m)  # type: ignore[no-untyped-call]
    )


def tensor_3_vec_contract(
    tensor: Rank3Tensor, vector: AbstractVector, tensor_index: int
) -> AbstractMatrix:
    """
    Returns contracted matrix from tansor of rank 3 and vector.

    :param tensor_index: Index of tesnor to contract (starts with 0)

    Example:
            M_{ijk} = T_{ilj} V^{l}
        for tensor T with index 1 and vector V.
    """
    if not 0 <= tensor_index <= 2:
        raise ValueError(
            f"Cannot contract rank 3 tensor with matrix for"
            f" tensor_index={tensor_index}. Index must be either 0, 1 or 2."
        )
    return _abstract_matrix_from_numpy(
        np.tensordot(tensor._data, vector._v, axes=(tensor_index, 0))
    )


def tensor_3_mat_contract(
    tensor: Rank3Tensor,
    matrix: AbstractMatrix,
    tensor_index: int,
    matrix_index: int,
) -> Rank3Tensor:
    """
    Returns contracted tensor of rank 3 from tansor of rank 3 and matrix.

    :param tensor_index: Index of tesnor to contract (starts with 0)
    :param matrix_index: Index of matrix to contract (starts with 0)

    Example:
            R_{ijk} = T_{ilj} M^{l}_{k}
        for tensor T with index 1 and matrix M with index 0.
    """
    if not 0 <= tensor_index <= 2:
        raise ValueError(
            f"Cannot contract rank 3 tensor with matrix for"
            f" tensor_index={tensor_index}. Index must be either 0, 1 or 2."
        )
    if not 0 <= matrix_index <= 2:
        raise ValueError(
            f"Cannot contract rank 3 tensor with matrix for"
            f" matrix_index={matrix_index}. Index must be either 0 or 1."
        )
    return _rank_3_tensor_from_numpy(
        np.tensordot(tensor._data, matrix._m, axes=(tensor_index, matrix_index))
    )


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
    )  # POSSIBLE-OPTIMIZATION: hard code


def cross(
    vec1: AbstractVector,
    vec2: AbstractVector,
    jacobian: Optional[AbstractMatrix] = None,
) -> AbstractVector:
    """
    Returns the (orthonormal) cross product of both vectors.

    :param jacobian: Jacobian matric which transforms a vector to one in an
                     orthonormal basis.
    """
    if jacobian is None:
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
    vec1 = mat_vec_mult(jacobian, vec1)
    vec2 = mat_vec_mult(jacobian, vec2)
    vec_res = cross(vec1, vec2)
    return mat_vec_mult(inverted(jacobian), vec_res)


def length(vec: AbstractVector, metric: Optional[Metric] = None) -> float:
    """
    Returns the length of the vector (with respect to an orthonormal basis).
    """
    if metric is None:
        # POSSIBLE-OPTIMIZATION: hard code
        return np.linalg.norm(vec._v)  # type: ignore[no-untyped-call]
    return dot(vec, vec, metric) ** 0.5


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
    return vec / length(vec, metric)  # POSSIBLE-OPTIMIZATION: hard code


def are_linear_dependent(vectors: tuple[AbstractVector, ...]) -> bool:
    """
    Returns True iff, the vectors have a linear dependecy i.e. iff
    there exist non-trivial coefficients a_i such that
    v_1 * a_1 + v_2 * a_2 + ... + v_n * a_n = 0
    """
    if 1 <= len(vectors) <= 3:
        matrix = np.array(tuple(v._v for v in vectors))
        return np.linalg.matrix_rank(matrix) != len(vectors)  # type: ignore[no-untyped-call]
    return True


def transposed(mat: AbstractMatrix) -> AbstractMatrix:
    """
    Returns the transposed of a matrix.
    """
    return AbstractMatrix(
        AbstractVector((mat[0][0], mat[1][0], mat[2][0])),
        AbstractVector((mat[0][1], mat[1][1], mat[2][1])),
        AbstractVector((mat[0][2], mat[1][2], mat[2][2])),
    )


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
