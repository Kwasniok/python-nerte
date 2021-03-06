# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144
# pylint: disable=C0302

import unittest

from typing import Callable, Optional

import math

from nerte.base_test_case import BaseTestCase, float_almost_equal

from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Rank3Tensor,
    covariant,
    contravariant,
    is_zero_vector,
    mat_vec_mult,
    mat_mult,
    tensor_3_vec_contract,
    tensor_3_mat_contract,
    dot,
    cross,
    length,
    normalized,
    are_linear_dependent,
    transposed,
    inverted,
)


def scalar_equiv(x: float, y: float) -> bool:
    """Returns true iff both scalars are considered equivalent."""
    return math.isclose(x, y)


scalar_almost_equal = float_almost_equal


def vec_equiv(x: AbstractVector, y: AbstractVector) -> bool:
    """Returns true iff both vectors are considered equivalent."""
    return (
        scalar_equiv(x[0], y[0])
        and scalar_equiv(x[1], y[1])
        and scalar_equiv(x[2], y[2])
    )


def vec_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[AbstractVector, AbstractVector], bool]:
    """
    Returns a function which true iff both vectors are considered almost equal.
    """

    # pylint: disable=W0621
    def vec_almost_equal(x: AbstractVector, y: AbstractVector) -> bool:
        pred = scalar_almost_equal(places=places, delta=delta)
        return pred(x[0], y[0]) and pred(x[1], y[1]) and pred(x[2], y[2])

    return vec_almost_equal


def mat_equiv(x: AbstractMatrix, y: AbstractMatrix) -> bool:
    """Returns true iff both matrices are considered equivalent."""
    return (
        vec_equiv(x[0], y[0])
        and vec_equiv(x[1], y[1])
        and vec_equiv(x[2], y[2])
    )


def mat_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[AbstractMatrix, AbstractMatrix], bool]:
    """
    Returns a function which true iff both matrices are considered almost equal.
    """

    # pylint: disable=W0621
    def mat_almost_equal(x: AbstractMatrix, y: AbstractMatrix) -> bool:
        pred = vec_almost_equal(places=places, delta=delta)
        return pred(x[0], y[0]) and pred(x[1], y[1]) and pred(x[2], y[2])

    return mat_almost_equal


def rank3tensor_equiv(x: Rank3Tensor, y: Rank3Tensor) -> bool:
    """Returns true iff both rank 3 tensors are considered equivalent."""
    return (
        mat_equiv(x[0], y[0])
        and mat_equiv(x[1], y[1])
        and mat_equiv(x[2], y[2])
    )


def rank3tensor_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[Rank3Tensor, Rank3Tensor], bool]:
    """
    Returns a function which true iff both rank 3 tensors are considered almost
    equal.
    """

    # pylint: disable=W0621
    def rank3tensor_almost_equal(x: Rank3Tensor, y: Rank3Tensor) -> bool:
        pred = mat_almost_equal(places=places, delta=delta)
        return pred(x[0], y[0]) and pred(x[1], y[1]) and pred(x[2], y[2])

    return rank3tensor_almost_equal


class AbstractVectorTestItem(BaseTestCase):
    def setUp(self) -> None:
        self.cs = (1.0, 2.0, 3.0)

    def test_vector_item(self) -> None:
        """Tests item related operations."""
        v = AbstractVector(self.cs)
        for i, c in zip(range(3), self.cs):
            self.assertPredicate2(scalar_equiv, v[i], c)


class AbstractVectorMathTest(BaseTestCase):
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.1, 2.2, 3.3))
        self.v2 = AbstractVector((4.4, 5.5, 6.6))
        self.v3 = AbstractVector((5.5, 7.7, 9.9))
        self.v4 = AbstractVector((1.0, 1.0, 1.0))
        self.v5 = AbstractVector((3.3, 3.3, 3.3))

    def test_vector_linear(self) -> None:
        """Tests linear operations on vectors."""

        self.assertPredicate2(vec_equiv, self.v1 + self.v2, self.v3)
        self.assertPredicate2(vec_equiv, self.v4 * 3.3, self.v5)
        self.assertPredicate2(vec_equiv, self.v2 - self.v1, self.v5)
        self.assertPredicate2(vec_equiv, self.v5 / 3.3, self.v4)


class AbstractMatrixTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.1, 2.2, 3.3))
        self.v2 = AbstractVector((4.4, 5.5, 6.6))
        self.v3 = AbstractVector((5.5, 7.7, 9.9))

    def test_matrix_constructor(self) -> None:
        """Test matrix constructor."""
        AbstractMatrix(self.v0, self.v1, self.v2)
        AbstractMatrix(self.v1, self.v1, self.v1)
        AbstractMatrix(self.v1, self.v2, self.v3)


class AbstractMatrixTestItem(BaseTestCase):
    def setUp(self) -> None:
        self.vs = (
            AbstractVector((1.0, 2.0, 3.0)),
            AbstractVector((4.0, 5.0, 6.0)),
            AbstractVector((7.0, 8.0, 9.0)),
        )

    def test_matrix_item(self) -> None:
        """Tests item related operations."""
        m = AbstractMatrix(*self.vs)
        for i, v in zip(range(3), self.vs):
            self.assertPredicate2(vec_equiv, m[i], v)


class AbstractMatrixIsInvertibleTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        v1 = AbstractVector((2.0, 3.0, 5.0))
        v2 = AbstractVector((7.0, 11.0, 13.0))
        v3 = AbstractVector((17.0, 23.0, 27))
        self.invertible_mats = (
            AbstractMatrix(v1, v2, v3),
            AbstractMatrix(v2, v3, v1),
            AbstractMatrix(v3, v1, v2),
            AbstractMatrix(v3, v2, v1),
            AbstractMatrix(v2, v1, v3),
            AbstractMatrix(v1, v3, v2),
        )

        self.non_invertible_mats = (
            AbstractMatrix(v0, v0, v0),
            AbstractMatrix(v1, v1, v1),
            AbstractMatrix(v1, v1, v3),
            AbstractMatrix(v1, v2, v2),
            AbstractMatrix(v3, v2, v3),
        )

    def test_matrix_is_invertible(self) -> None:
        """Tests matix invertibility property."""
        for mat in self.invertible_mats:
            self.assertTrue(mat.is_invertible())
        for mat in self.non_invertible_mats:
            self.assertFalse(mat.is_invertible())


class AbstractMatrixMathTest(BaseTestCase):
    def setUp(self) -> None:
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((4.4, 5.5, 6.6))
        v3 = AbstractVector((5.5, 7.7, 9.9))
        v4 = AbstractVector((1.0, 1.0, 1.0))
        v5 = AbstractVector((3.3, 3.3, 3.3))
        self.m1 = AbstractMatrix(v1, v1, v1)
        self.m2 = AbstractMatrix(v2, v2, v2)
        self.m3 = AbstractMatrix(v3, v3, v3)
        self.m4 = AbstractMatrix(v4, v4, v4)
        self.m5 = AbstractMatrix(v5, v5, v5)

    def test_matrix_linear(self) -> None:
        """Tests linear operations on matrices."""

        self.assertPredicate2(mat_equiv, self.m1 + self.m2, self.m3)
        self.assertPredicate2(mat_equiv, self.m4 * 3.3, self.m5)
        self.assertPredicate2(mat_equiv, self.m2 - self.m1, self.m5)
        self.assertPredicate2(mat_equiv, self.m5 / 3.3, self.m4)


class Rank3TensorTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((4.4, 5.5, 6.6))
        v3 = AbstractVector((5.5, 7.7, 9.9))
        v4 = AbstractVector((1.0, 1.1, 1.2))
        v5 = AbstractVector((1.3, 1.4, 1.5))
        self.m0 = AbstractMatrix(v0, v1, v2)
        self.m1 = AbstractMatrix(v1, v2, v3)
        self.m2 = AbstractMatrix(v2, v3, v4)
        self.m3 = AbstractMatrix(v3, v4, v5)

    def test_constructor(self) -> None:
        """Test rank 3 tensor constructor."""
        Rank3Tensor(self.m0, self.m1, self.m2)
        Rank3Tensor(self.m1, self.m1, self.m1)
        Rank3Tensor(self.m1, self.m2, self.m3)


class Rank3TensorTestItem(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((4.4, 5.5, 6.6))
        v3 = AbstractVector((5.5, 7.7, 9.9))
        v4 = AbstractVector((1.0, 1.1, 1.2))
        m0 = AbstractMatrix(v0, v1, v2)
        m1 = AbstractMatrix(v1, v2, v3)
        m2 = AbstractMatrix(v2, v3, v4)
        self.ms = (m0, m1, m2)
        self.r3tensor = Rank3Tensor(m0, m1, m2)

    def test_item(self) -> None:
        """Tests item related operations or a rank 3 tensor."""
        for i, m in zip(range(3), self.ms):
            self.assertPredicate2(mat_equiv, self.r3tensor[i], m)


class Rank3TensorMathTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((4.4, 5.5, 6.6))
        v3 = AbstractVector((5.5, 7.7, 9.9))
        v4 = AbstractVector((1.0, 1.1, 1.2))
        v5 = AbstractVector((1.1, 7.7, 16.5))
        v5 = AbstractVector((5.5, 7.7, 19.9))
        m0 = AbstractMatrix(v0, v1, v2)
        m1 = AbstractMatrix(v1, v2, v3)
        m2 = AbstractMatrix(v2, v3, v4)
        m3 = AbstractMatrix(v3, v4, v5)
        self.r3t1 = Rank3Tensor(m0, m1, m2)
        self.r3t2 = Rank3Tensor(m1, m2, m3)
        self.r3t3 = Rank3Tensor(m0 + m1, m1 + m2, m2 + m3)
        self.r3t4 = Rank3Tensor(m0 * 3.3, m1 * 3.3, m2 * 3.3)

    def test_linear(self) -> None:
        """Tests linear operations on rank 3 tensors."""

        self.assertPredicate2(
            rank3tensor_equiv, self.r3t1 + self.r3t2, self.r3t3
        )
        self.assertPredicate2(
            rank3tensor_equiv, self.r3t3 - self.r3t2, self.r3t1
        )
        self.assertPredicate2(rank3tensor_equiv, self.r3t1 * 3.3, self.r3t4)
        self.assertPredicate2(rank3tensor_equiv, self.r3t4 / 3.3, self.r3t1)


class AbstractVectorIsZero(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 2.0, -3.0))

    def test_is_zero_vector(self) -> None:
        """Tests zero vector test."""

        self.assertTrue(is_zero_vector(self.v0))
        self.assertFalse(is_zero_vector(self.v1))


class MatVecMultTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.vec0 = AbstractVector((0.0, 0.0, 0.0))
        self.vec1 = AbstractVector((2.0, 3.0, 5.0))
        self.vec2 = AbstractVector((7.0, 11.0, 13.0))
        self.vec3 = AbstractVector((17.0, 19.0, 23.0))
        self.vec4 = AbstractVector((29.0, 31.0, 37.0))
        self.vec5 = AbstractVector((336, 1025, 1933))
        self.mat0 = AbstractMatrix(self.vec0, self.vec0, self.vec0)
        self.mat1 = AbstractMatrix(self.vec1, self.vec2, self.vec3)

    def test_mat_vec_mult(self) -> None:
        """Tests the multiplication of a matrix and a vector."""
        v = mat_vec_mult(self.mat0, self.vec0)
        self.assertPredicate2(vec_equiv, v, self.vec0)
        v = mat_vec_mult(self.mat1, self.vec0)
        self.assertPredicate2(vec_equiv, v, self.vec0)
        v = mat_vec_mult(self.mat0, self.vec1)
        self.assertPredicate2(vec_equiv, v, self.vec0)
        v = mat_vec_mult(self.mat1, self.vec4)
        self.assertPredicate2(vec_equiv, v, self.vec5)


class MatMultTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.vec0 = AbstractVector((0.0, 0.0, 0.0))
        self.mat0 = AbstractMatrix(self.vec0, self.vec0, self.vec0)
        self.mat1 = AbstractMatrix(
            AbstractVector((02.0, 03.0, 05.0)),
            AbstractVector((07.0, 11.0, 13.0)),
            AbstractVector((17.0, 19.0, 23.0)),
        )
        self.mat2 = AbstractMatrix(
            AbstractVector((29.0, 31.0, 37.0)),
            AbstractVector((41.0, 43.0, 47.0)),
            AbstractVector((53.0, 59.0, 61)),
        )
        self.mat3 = AbstractMatrix(
            AbstractVector((446, 486, 520)),
            AbstractVector((1343, 1457, 1569)),
            AbstractVector((2491, 2701, 2925)),
        )

    def test_mat_mult(self) -> None:
        """Tests the multiplication of two matrices."""
        self.assertPredicate2(
            mat_equiv, mat_mult(self.mat1, self.mat0), self.mat0
        )
        self.assertPredicate2(
            mat_equiv, mat_mult(self.mat0, self.mat1), self.mat0
        )
        self.assertPredicate2(
            mat_equiv, mat_mult(self.mat1, self.mat2), self.mat3
        )


class Rank3TensorVectorContractionTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        self.v0 = AbstractVector((0, 0, 0))
        self.mat0 = AbstractMatrix(self.v0, self.v0, self.v0)
        self.ten0 = Rank3Tensor(self.mat0, self.mat0, self.mat0)
        self.ten = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((2, 3, 5)),
                AbstractVector((7, 11, 13)),
                AbstractVector((17, 19, 23)),
            ),
            AbstractMatrix(
                AbstractVector((29, 31, 37)),
                AbstractVector((41, 43, 47)),
                AbstractVector((53, 59, 61)),
            ),
            AbstractMatrix(
                AbstractVector((67, 71, 73)),
                AbstractVector((79, 83, 89)),
                AbstractVector((97, 101, 103)),
            ),
        )
        self.vec = AbstractVector((107, 109, 113))
        self.mat00 = AbstractMatrix(
            AbstractVector((10946, 11723, 12817)),
            AbstractVector((14145, 15243, 16571)),
            AbstractVector((18557, 19877, 20749)),
        )
        self.mat10 = AbstractMatrix(
            AbstractVector((2898, 3667, 4551)),
            AbstractVector((13561, 14671, 15975)),
            AbstractVector((26741, 28057, 29151)),
        )
        self.mat20 = AbstractMatrix(
            AbstractVector((1106, 3417, 6489)),
            AbstractVector((10663, 14385, 18995)),
            AbstractVector((23157, 27557, 33027)),
        )

    def test_mat_mult(self) -> None:
        """Tests the contraction of a rank 3 tensor and a vector."""
        self.assertPredicate2(
            mat_equiv, tensor_3_vec_contract(self.ten0, self.v0, 0), self.mat0
        )
        self.assertPredicate2(
            mat_equiv, tensor_3_vec_contract(self.ten0, self.v0, 1), self.mat0
        )
        self.assertPredicate2(
            mat_equiv, tensor_3_vec_contract(self.ten0, self.v0, 2), self.mat0
        )
        self.assertPredicate2(
            mat_equiv, tensor_3_vec_contract(self.ten, self.vec, 0), self.mat00
        )
        self.assertPredicate2(
            mat_equiv, tensor_3_vec_contract(self.ten, self.vec, 1), self.mat10
        )
        self.assertPredicate2(
            mat_equiv, tensor_3_vec_contract(self.ten, self.vec, 2), self.mat20
        )


class Rank3TensorMatrixContractionTest(BaseTestCase):
    # pylint: disable=R0902
    def setUp(self) -> None:
        v0 = AbstractVector((0, 0, 0))
        self.mat0 = AbstractMatrix(v0, v0, v0)
        self.ten0 = Rank3Tensor(self.mat0, self.mat0, self.mat0)
        self.ten = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((2, 3, 5)),
                AbstractVector((7, 11, 13)),
                AbstractVector((17, 19, 23)),
            ),
            AbstractMatrix(
                AbstractVector((29, 31, 37)),
                AbstractVector((41, 43, 47)),
                AbstractVector((53, 59, 61)),
            ),
            AbstractMatrix(
                AbstractVector((67, 71, 73)),
                AbstractVector((79, 83, 89)),
                AbstractVector((97, 101, 103)),
            ),
        )
        self.mat = AbstractMatrix(
            AbstractVector((107, 109, 113)),
            AbstractVector((127, 131, 137)),
            AbstractVector((139, 149, 151)),
        )
        self.ten00 = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((13210, 14000, 14316)),
                AbstractVector((14127, 14967, 15307)),
                AbstractVector((15381, 16269, 16657)),
            ),
            AbstractMatrix(
                AbstractVector((16937, 17905, 18337)),
                AbstractVector((18175, 19199, 19667)),
                AbstractVector((19731, 20835, 21347)),
            ),
            AbstractMatrix(
                AbstractVector((22033, 23249, 23829)),
                AbstractVector((23565, 24849, 25481)),
                AbstractVector((24525, 25845, 26509)),
            ),
        )
        self.ten01 = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((10946, 13232, 14716)),
                AbstractVector((11723, 14169, 15757)),
                AbstractVector((12817, 15483, 17231)),
            ),
            AbstractMatrix(
                AbstractVector((14145, 17083, 19011)),
                AbstractVector((15243, 18401, 20469)),
                AbstractVector((16571, 20001, 22249)),
            ),
            AbstractMatrix(
                AbstractVector((18557, 22391, 24907)),
                AbstractVector((19877, 23979, 26683)),
                AbstractVector((20749, 25023, 27839)),
            ),
        )
        self.ten10 = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((3466, 3668, 3752)),
                AbstractVector((4359, 4599, 4715)),
                AbstractVector((5383, 5675, 5819)),
            ),
            AbstractMatrix(
                AbstractVector((15677, 16429, 16897)),
                AbstractVector((16979, 17803, 18303)),
                AbstractVector((18407, 19279, 19831)),
            ),
            AbstractMatrix(
                AbstractVector((30685, 32105, 33041)),
                AbstractVector((32177, 33661, 34645)),
                AbstractVector((33431, 34963, 35995)),
            ),
        )
        self.ten11 = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((2898, 3500, 3888)),
                AbstractVector((3667, 4425, 4925)),
                AbstractVector((4551, 5489, 6105)),
            ),
            AbstractMatrix(
                AbstractVector((13561, 16315, 18143)),
                AbstractVector((14671, 17653, 19625)),
                AbstractVector((15975, 19213, 21357)),
            ),
            AbstractMatrix(
                AbstractVector((26741, 32147, 35731)),
                AbstractVector((28057, 33727, 37487)),
                AbstractVector((29151, 35041, 38961)),
            ),
        )
        self.ten20 = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((1290, 1356, 1392)),
                AbstractVector((3953, 4141, 4261)),
                AbstractVector((7429, 7769, 7997)),
            ),
            AbstractMatrix(
                AbstractVector((12183, 12735, 13111)),
                AbstractVector((16381, 17105, 17621)),
                AbstractVector((21643, 22595, 23283)),
            ),
            AbstractMatrix(
                AbstractVector((26333, 27481, 28321)),
                AbstractVector((31365, 32745, 33737)),
                AbstractVector((37523, 39151, 40351)),
            ),
        )
        self.ten21 = Rank3Tensor(
            AbstractMatrix(
                AbstractVector((1106, 1332, 1480)),
                AbstractVector((3417, 4111, 4575)),
                AbstractVector((6489, 7799, 8667)),
            ),
            AbstractMatrix(
                AbstractVector((10663, 12813, 14237)),
                AbstractVector((14385, 17279, 19203)),
                AbstractVector((18995, 22817, 25369)),
            ),
            AbstractMatrix(
                AbstractVector((23157, 27811, 30915)),
                AbstractVector((27557, 33099, 36787)),
                AbstractVector((33027, 39661, 44085)),
            ),
        )

    def test_tensor_3_mat_contract(self) -> None:
        """Tests the contraction of a rank 3 tensor and a matrix."""
        for i in range(3):
            for j in range(2):
                self.assertPredicate2(
                    rank3tensor_equiv,
                    tensor_3_mat_contract(self.ten0, self.mat0, i, j),
                    self.ten0,
                )
        self.assertPredicate2(
            rank3tensor_equiv,
            tensor_3_mat_contract(self.ten, self.mat, 0, 0),
            self.ten00,
        )
        self.assertPredicate2(
            rank3tensor_equiv,
            tensor_3_mat_contract(self.ten, self.mat, 0, 1),
            self.ten01,
        )
        self.assertPredicate2(
            rank3tensor_equiv,
            tensor_3_mat_contract(self.ten, self.mat, 1, 0),
            self.ten10,
        )
        self.assertPredicate2(
            rank3tensor_equiv,
            tensor_3_mat_contract(self.ten, self.mat, 1, 1),
            self.ten11,
        )
        self.assertPredicate2(
            rank3tensor_equiv,
            tensor_3_mat_contract(self.ten, self.mat, 2, 0),
            self.ten20,
        )
        self.assertPredicate2(
            rank3tensor_equiv,
            tensor_3_mat_contract(self.ten, self.mat, 2, 1),
            self.ten21,
        )


class LengthTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 2.0, -3.0))
        # metric
        self.metric = AbstractMatrix(
            AbstractVector((5.0, 0.0, 0.0)),
            AbstractVector((0.0, 7.0, 0.0)),
            AbstractVector((0.0, 0.0, 11.0)),
        )

    def test_length(self) -> None:
        """Tests vector length."""

        self.assertPredicate2(scalar_equiv, length(self.v0), 0.0)
        self.assertPredicate2(scalar_equiv, length(self.v1) ** 2, 14.0)

    def test_length_metric(self) -> None:
        """Tests vector length with metric."""

        self.assertPredicate2(
            scalar_equiv,
            length(self.v0, metric=self.metric),
            0.0,
        )
        self.assertPredicate2(
            scalar_equiv,
            length(self.v1, metric=self.metric) ** 2,
            132.0,
        )


class NormalizedTest(BaseTestCase):
    def setUp(self) -> None:
        self.n = AbstractVector((1.0, 1.0, 1.0)) / math.sqrt(3)
        self.w = AbstractVector((7.0, 7.0, 7.0))
        # metric
        self.metric = AbstractMatrix(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 2.0, 0.0)),
            AbstractVector((0.0, 0.0, 3.0)),
        )

        self.n_metric = AbstractVector((1.0, 1.0, 1.0)) / math.sqrt(6)

    def test_normalized(self) -> None:
        """Tests vector normalization."""
        self.assertPredicate2(vec_equiv, normalized(self.w), self.n)

    def test_normized_metric(self) -> None:
        """Tests vetor normalization with metric."""
        self.assertPredicate2(
            vec_equiv,
            normalized(self.w, metric=self.metric),
            self.n_metric,
        )


class DotTest(BaseTestCase):
    def setUp(self) -> None:
        # standart Cartesian basis
        self.orth_norm_basis = (
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )
        # arbitrary factors
        self.scalar_factors = (0.0, 1.2345, -0.98765)
        # metric
        self.metric = AbstractMatrix(
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 2.0, 0.0)),
            AbstractVector((0.0, 0.0, 3.0)),
        )

    def test_math_dot_orthonormality(self) -> None:
        """Tests dot product acting on orthonormal basis."""
        for v in self.orth_norm_basis:
            for w in self.orth_norm_basis:
                if v is w:
                    self.assertPredicate2(scalar_equiv, dot(v, w), 1.0)
                else:
                    self.assertPredicate2(scalar_equiv, dot(v, w), 0.0)

    def test_math_dot_linearity_left(self) -> None:
        """Tests dot product's linearity in the left argument."""
        for u in self.orth_norm_basis:
            for v in self.orth_norm_basis:
                for w in self.orth_norm_basis:
                    for a in self.scalar_factors:
                        for b in self.scalar_factors:
                            self.assertPredicate2(
                                scalar_equiv,
                                dot((v * a) + (w * b), u),
                                dot(v, u) * a + dot(w, u) * b,
                            )

    def test_math_dot_linearity_right(self) -> None:
        """Tests dot product's linearity in the right argument."""
        for u in self.orth_norm_basis:
            for v in self.orth_norm_basis:
                for w in self.orth_norm_basis:
                    for a in self.scalar_factors:
                        for b in self.scalar_factors:
                            self.assertPredicate2(
                                scalar_equiv,
                                dot(u, (v * a) + (w * b)),
                                dot(u, v) * a + dot(u, w) * b,
                            )

    def test_math_dot_metric(self) -> None:
        """Tests dot with metric."""
        for i, v in enumerate(self.orth_norm_basis):
            for j, w in enumerate(self.orth_norm_basis):
                self.assertPredicate2(
                    scalar_equiv,
                    dot(v, w, metric=self.metric),
                    self.metric[i][j],
                )


class CrossTest(BaseTestCase):
    def setUp(self) -> None:
        # standart Cartesian basis
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.orth_norm_basis = (
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )
        # result of corss product for orthonormal vectors
        self.ortho_norm_res = {
            (0, 0): v0,
            (0, 1): self.orth_norm_basis[2],
            (0, 2): -self.orth_norm_basis[1],
            (1, 0): -self.orth_norm_basis[2],
            (1, 1): v0,
            (1, 2): self.orth_norm_basis[0],
            (2, 0): self.orth_norm_basis[1],
            (2, 1): -self.orth_norm_basis[0],
            (2, 2): v0,
        }
        # arbitrary factors
        self.scalar_factors = (0.0, 1.2345, -0.98765)
        # vector transformation (permutation (0,1,2)->(2,1,0))
        self.jacobian = AbstractMatrix(
            AbstractVector((0.0, 0.0, 1.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            AbstractVector((1.0, 0.0, 0.0)),
        )
        # result of corss product for transformed vectors
        self.jacobian_res = {
            (0, 0): v0,
            (0, 1): -self.orth_norm_basis[2],
            (0, 2): self.orth_norm_basis[1],
            (1, 0): self.orth_norm_basis[2],
            (1, 1): v0,
            (1, 2): -self.orth_norm_basis[0],
            (2, 0): -self.orth_norm_basis[1],
            (2, 1): self.orth_norm_basis[0],
            (2, 2): v0,
        }

    def test_math_cross_orthonormality(self) -> None:
        """Tests cross product acting on orthonormal basis."""
        for i, v in enumerate(self.orth_norm_basis):
            for j, w in enumerate(self.orth_norm_basis):
                self.assertPredicate2(
                    vec_equiv,
                    cross(v, w),
                    self.ortho_norm_res[(i, j)],
                )

    def test_math_cross_antisymmetry(self) -> None:
        """Tests cross product's antisymmetry."""
        for v in self.orth_norm_basis:
            for w in self.orth_norm_basis:
                self.assertPredicate2(vec_equiv, cross(v, w), -cross(w, v))

    def test_math_cross_linearity_left(self) -> None:
        """Tests cross product's linearity in the left argument."""
        for u in self.orth_norm_basis:
            for v in self.orth_norm_basis:
                for w in self.orth_norm_basis:
                    for a in self.scalar_factors:
                        for b in self.scalar_factors:
                            self.assertPredicate2(
                                vec_equiv,
                                cross((v * a) + (w * b), u),
                                cross(v, u) * a + cross(w, u) * b,
                            )

    def test_math_cross_linearity_right(self) -> None:
        """Tests cross product's linearity in the right argument."""
        for u in self.orth_norm_basis:
            for v in self.orth_norm_basis:
                for w in self.orth_norm_basis:
                    for a in self.scalar_factors:
                        for b in self.scalar_factors:
                            self.assertPredicate2(
                                vec_equiv,
                                cross(u, (v * a) + (w * b)),
                                cross(u, v) * a + cross(u, w) * b,
                            )

    def test_math_cross_jacobian(self) -> None:
        """Tests cross with jacobian matrix."""
        for i, v in enumerate(self.orth_norm_basis):
            for j, w in enumerate(self.orth_norm_basis):
                self.assertPredicate2(
                    vec_equiv,
                    cross(v, w, jacobian=self.jacobian),
                    self.jacobian_res[(i, j)],
                )


class AreLinearDependentTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.v3 = AbstractVector((0.0, 0.0, 1.0))
        self.v4 = AbstractVector((1.0, 2.0, -3.0))

    def test_are_linear_dependent(self) -> None:
        """Tests linear dependecy check."""
        # are not dependent
        self.assertFalse(are_linear_dependent((self.v1,)))
        self.assertFalse(are_linear_dependent((self.v1, self.v2)))
        self.assertFalse(are_linear_dependent((self.v1, self.v2, self.v3)))
        # are dependent
        self.assertTrue(are_linear_dependent(()))
        self.assertTrue(are_linear_dependent((self.v1, self.v1)))
        self.assertTrue(are_linear_dependent((self.v1, self.v2, self.v1)))
        self.assertTrue(are_linear_dependent((self.v0,)))
        self.assertTrue(are_linear_dependent((self.v1, self.v0)))
        self.assertTrue(are_linear_dependent((self.v1, self.v0, self.v1)))
        self.assertTrue(are_linear_dependent((self.v1, self.v0, self.v2)))
        self.assertTrue(
            are_linear_dependent((self.v1, self.v2, self.v3, self.v4))
        )


class InvertedTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        e0 = AbstractVector((1.0, 0.0, 0.0))
        e1 = AbstractVector((0.0, 1.0, 0.0))
        e2 = AbstractVector((0.0, 0.0, 1.0))
        orth_norm_basis = (e0, e1, e2)
        self.m0 = AbstractMatrix(v0, v0, v0)
        self.mI = AbstractMatrix(*orth_norm_basis)  # own inverse
        self.m2 = AbstractMatrix(e0 * 1.0, e1 * 2.0, e2 * 3.0)
        self.m2_inv = AbstractMatrix(e0 / 1.0, e1 / 2.0, e2 / 3.0)
        self.m3 = AbstractMatrix(e0, e1, e0 * -1.0)  # linear dependency
        self.m4 = AbstractMatrix(
            AbstractVector((2, 3, 5)),
            AbstractVector((7, 11, 13)),
            AbstractVector((17, 19, 23)),
        )
        self.m4_inv = (
            AbstractMatrix(
                AbstractVector((-6, -26, 16)),
                AbstractVector((-60, 39, -9)),
                AbstractVector((54, -13, -1)),
            )
            / 78
        )

    def test_inverted(self) -> None:
        """Tests matrix inversion."""
        with self.assertRaises(ArithmeticError):
            inverted(self.m0)
        self.assertPredicate2(mat_equiv, inverted(self.mI), self.mI)
        self.assertPredicate2(mat_equiv, inverted(self.m2), self.m2_inv)
        with self.assertRaises(ArithmeticError):
            inverted(self.m3)
        self.assertPredicate2(mat_equiv, inverted(self.m4), self.m4_inv)


class TransposedTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((1.0, 2.0, 3.0))
        v1 = AbstractVector((4.0, 5.0, 6.0))
        v2 = AbstractVector((7.0, 8.0, 9.0))
        w0 = AbstractVector((1.0, 4.0, 7.0))
        w1 = AbstractVector((2.0, 5.0, 8.0))
        w2 = AbstractVector((3.0, 6.0, 9.0))
        self.m1 = AbstractMatrix(v0, v1, v2)
        self.m2 = AbstractMatrix(w0, w1, w2)

    def test_transposed(self) -> None:
        """Tests matrix trasposition."""
        self.assertPredicate2(mat_equiv, transposed(self.m1), self.m2)


class CoAndCoraviantTest(BaseTestCase):
    # pylint: disable=R0902

    def setUp(self) -> None:
        e0 = AbstractVector((1.0, 0.0, 0.0))
        e1 = AbstractVector((0.0, 1.0, 0.0))
        e2 = AbstractVector((0.0, 0.0, 1.0))
        orth_norm_basis = (e0, e1, e2)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1_con = AbstractVector((1.0, 2.0, 3.0))

        self.gI = AbstractMatrix(*orth_norm_basis)

        self.g1 = AbstractMatrix(e0 * 2.0, e1 * 5.0, e2 * 7.0)
        self.v11_co = AbstractVector((2.0, 10.0, 21.0))

        self.g2 = AbstractMatrix(
            AbstractVector((2, 3, 5)),
            AbstractVector((3, 11, 13)),
            AbstractVector((5, 13, 23)),
        )
        self.v12_co = AbstractVector((23, 64, 100))

        self.gs = (self.gI, self.g1, self.g2)
        self.vs = (self.v0, self.v1_con)

    def test_co_variant(self) -> None:
        """Tests covariant conversion."""
        self.assertPredicate2(vec_equiv, covariant(self.gI, self.v0), self.v0)
        self.assertPredicate2(vec_equiv, covariant(self.g1, self.v0), self.v0)
        self.assertPredicate2(
            vec_equiv,
            covariant(self.g1, self.v1_con),
            self.v11_co,
        )
        self.assertPredicate2(
            vec_equiv,
            covariant(self.g2, self.v1_con),
            self.v12_co,
        )

    def test_contra_variant(self) -> None:
        """Tests contravariant conversion."""
        self.assertPredicate2(
            vec_equiv,
            contravariant(self.gI, self.v0),
            self.v0,
        )
        self.assertPredicate2(
            vec_equiv,
            contravariant(self.g1, self.v0),
            self.v0,
        )
        self.assertPredicate2(
            vec_equiv,
            contravariant(self.g1, self.v11_co),
            self.v1_con,
        )
        self.assertPredicate2(
            vec_equiv,
            contravariant(self.g2, self.v12_co),
            self.v1_con,
        )

    def test_co_contra_variant_invertibility(self) -> None:
        """Tests if co- and contravariant invert each other."""
        for g in self.gs:
            for v in self.vs:
                self.assertPredicate2(
                    vec_equiv,
                    covariant(g, contravariant(g, v)),
                    v,
                )
                self.assertPredicate2(
                    vec_equiv,
                    contravariant(g, covariant(g, v)),
                    v,
                )


if __name__ == "__main__":
    unittest.main()
