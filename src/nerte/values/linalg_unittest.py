# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Callable, Optional

import math

from nerte.base_test_case import BaseTestCase, float_almost_equal

from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    covariant,
    contravariant,
    is_zero_vector,
    mat_vec_mult,
    dot,
    cross,
    length,
    normalized,
    are_linear_dependent,
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


def metric_equiv(x: Metric, y: Metric) -> bool:
    """Returns true iff both metrics are considered equivalent."""
    return mat_equiv(x.matrix(), y.matrix())


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


class AbstractMatrixIsSymmetricTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 1e-9, 0.0))  # simulated small numerical error
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((2.2, 5.5, 6.6))
        v3 = AbstractVector((3.3, 6.6, 9.9))
        v2_anti = AbstractVector((-2.2, 5.5, 6.6))
        v3_anti = AbstractVector((-3.3, -6.6, 9.9))

        self.symmetic_mats = (
            AbstractMatrix(v1, v2, v3),
            AbstractMatrix(v1 + v0, v2, v3 + v0),
        )

        self.non_symmetic_mats = (
            AbstractMatrix(v0, v0, v0),
            AbstractMatrix(v1, v1, v1),
            AbstractMatrix(v1, v2_anti, v3_anti),
        )

    def test_matrix_is_symmetric(self) -> None:
        """Tests matix symmetry property."""
        for mat in self.symmetic_mats:
            self.assertTrue(mat.is_symmetric())
        for mat in self.non_symmetic_mats:
            self.assertFalse(mat.is_symmetric())


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


class MetricTest(BaseTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.m0 = AbstractMatrix(v0, v0, v0)
        self.m = AbstractMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((3.0, 11.0, 13.0)),
            AbstractVector((5.0, 13.0, 23.0)),
        )
        self.m_inv = inverted(self.m)
        self.m_non_symm = AbstractMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((7.0, 11.0, 13.0)),
            AbstractVector((17.0, 19.0, 23.0)),
        )

    def test_metric_constructor(self) -> None:
        """Tests metric constructor."""
        Metric(self.m)
        with self.assertRaises(ValueError):
            Metric(self.m0)
        with self.assertRaises(ValueError):
            Metric(self.m_non_symm)

    def test_metric_matrix_getters(self) -> None:
        """Tests metric matrix getters."""
        g = Metric(self.m)
        self.assertPredicate2(mat_equiv, g.matrix(), self.m)
        self.assertPredicate2(mat_equiv, g.inverse_matrix(), self.m_inv)


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


class LengthTest(BaseTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 2.0, -3.0))
        # metric
        self.metric = Metric(
            AbstractMatrix(
                AbstractVector((5.0, 0.0, 0.0)),
                AbstractVector((0.0, 7.0, 0.0)),
                AbstractVector((0.0, 0.0, 11.0)),
            )
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
        self.metric = Metric(
            AbstractMatrix(
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 2.0, 0.0)),
                AbstractVector((0.0, 0.0, 3.0)),
            )
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
        # standart Carthesian basis
        self.orth_norm_basis = (
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )
        # arbitrary factors
        self.scalar_factors = (0.0, 1.2345, -0.98765)
        # metric
        self.metric = Metric(
            AbstractMatrix(
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 2.0, 0.0)),
                AbstractVector((0.0, 0.0, 3.0)),
            )
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
                    self.metric.matrix()[i][j],
                )


class CrossTest(BaseTestCase):
    def setUp(self) -> None:
        # standart Carthesian basis
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


class CoAndCoraviantTest(BaseTestCase):
    # pylint: disable=R0902

    def setUp(self) -> None:
        e0 = AbstractVector((1.0, 0.0, 0.0))
        e1 = AbstractVector((0.0, 1.0, 0.0))
        e2 = AbstractVector((0.0, 0.0, 1.0))
        orth_norm_basis = (e0, e1, e2)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1_con = AbstractVector((1.0, 2.0, 3.0))

        mI = AbstractMatrix(*orth_norm_basis)
        self.gI = Metric(mI)

        m1 = AbstractMatrix(e0 * 2.0, e1 * 5.0, e2 * 7.0)
        self.g1 = Metric(m1)
        self.v11_co = AbstractVector((2.0, 10.0, 21.0))

        m2 = AbstractMatrix(
            AbstractVector((2, 3, 5)),
            AbstractVector((3, 11, 13)),
            AbstractVector((5, 13, 23)),
        )
        self.g2 = Metric(m2)
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
