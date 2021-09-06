# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest

import math

from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    AbstractSymmetricMatrix,
    Metric,
    to_symmetric_matrix,
    covariant,
    contravariant,
    is_zero_vector,
    dot,
    cross,
    length,
    normalized,
    inverted,
)


# True, iff two floats are equivalent
def _equiv(x: float, y: float) -> bool:
    return math.isclose(x, y)


# True, iff two vectors component-wise agree up to the (absolute) precision 𝜀
def _vec_equiv(x: AbstractVector, y: AbstractVector) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two matrices component-wise agree up to the (absolute) precision 𝜀
def _mat_equiv(x: AbstractMatrix, y: AbstractMatrix) -> bool:
    return (
        _vec_equiv(x[0], y[0])
        and _vec_equiv(x[1], y[1])
        and _vec_equiv(x[2], y[2])
    )


class LinAlgTestCase(unittest.TestCase):
    def assertEquiv(self, x: float, y: float) -> None:
        """
        Asserts the equivalence of two floats.
        Note: This replaces assertTrue(x == y) for float.
        """
        try:
            self.assertTrue(_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Scalar {} is not equivalent to {}.".format(x, y)
            ) from ae

    def assertVectorEquiv(self, x: AbstractVector, y: AbstractVector) -> None:
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_vec_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Vector {} is not equivalent to {}.".format(x, y)
            ) from ae

    def assertMatrixEquiv(self, x: AbstractMatrix, y: AbstractMatrix) -> None:
        """
        Asserts ths equivalence of two matrices.
        Note: This replaces assertTrue(x == y) for matrices.
        """
        try:
            self.assertTrue(_mat_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Matrix {} is not equivalent to {}.".format(x, y)
            ) from ae


class AbstractVectorTestItem(LinAlgTestCase):
    def setUp(self) -> None:
        self.cs = (1.0, 2.0, 3.0)

    def test_vector_item(self) -> None:
        """Tests item related operations."""
        v = AbstractVector(self.cs)
        for i, c in zip(range(3), self.cs):
            self.assertEquiv(v[i], c)


class AbstractVectorMathTest(LinAlgTestCase):
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.1, 2.2, 3.3))
        self.v2 = AbstractVector((4.4, 5.5, 6.6))
        self.v3 = AbstractVector((5.5, 7.7, 9.9))
        self.v4 = AbstractVector((1.0, 1.0, 1.0))
        self.v5 = AbstractVector((3.3, 3.3, 3.3))

    def test_vector_linear(self) -> None:
        """Tests linear operations on vectors."""

        self.assertVectorEquiv(self.v1 + self.v2, self.v3)
        self.assertVectorEquiv(self.v4 * 3.3, self.v5)
        self.assertVectorEquiv(self.v2 - self.v1, self.v5)
        self.assertVectorEquiv(self.v5 / 3.3, self.v4)


class AbstractMatrixTest(LinAlgTestCase):
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


class AbstractMatrixTestItem(LinAlgTestCase):
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
            self.assertVectorEquiv(m[i], v)


class AbstractMatrixIsSymmetricTest(LinAlgTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 1e-9, 0.0))  # simulated small numerical error
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((2.2, 5.5, 6.6))
        v3 = AbstractVector((3.3, 6.6, 9.9))
        v2_anti = AbstractVector((-2.2, 5.5, 6.6))
        v3_anti = AbstractVector((-3.3, -6.6, 9.9))

        self.symmetic_mats = (
            AbstractMatrix(v0, v0, v0),
            AbstractMatrix(v1, v2, v3),
            AbstractMatrix(v1 + v0, v2, v3 + v0),
        )

        self.non_symmetic_mats = (
            AbstractMatrix(v1, v1, v1),
            AbstractMatrix(v1, v2_anti, v3_anti),
        )

    def test_matrix_is_symmetric(self) -> None:
        """Tests matix symmetry property."""
        for mat in self.symmetic_mats:
            self.assertTrue(mat.is_symmetric())
        for mat in self.non_symmetic_mats:
            self.assertFalse(mat.is_symmetric())


class AbstractMatrixMathTest(LinAlgTestCase):
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

        self.assertMatrixEquiv(self.m1 + self.m2, self.m3)
        self.assertMatrixEquiv(self.m4 * 3.3, self.m5)
        self.assertMatrixEquiv(self.m2 - self.m1, self.m5)
        self.assertMatrixEquiv(self.m5 / 3.3, self.m4)


class AbstractSymetricMatrixTest(LinAlgTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector(
            (0.0, 1e-9, 0.0)
        )  # simulated small numerical error
        self.v1 = AbstractVector((1.1, 2.2, 3.3))
        self.v2 = AbstractVector((2.2, 5.5, 6.6))
        self.v3 = AbstractVector((3.3, 6.6, 9.9))
        self.v2_anti = AbstractVector((-2.2, 5.5, 6.6))
        self.v3_anti = AbstractVector((-3.3, -6.6, 9.9))

    def test_matrix_constructor(self) -> None:
        """Test symmetric matrix constructor."""
        AbstractSymmetricMatrix(self.v0, self.v0, self.v0)
        AbstractSymmetricMatrix(self.v1, self.v2, self.v3)
        AbstractSymmetricMatrix(self.v1 + self.v0, self.v2, self.v3 + self.v0)
        with self.assertRaises(ValueError):
            AbstractSymmetricMatrix(self.v1, self.v1, self.v1)
        with self.assertRaises(ValueError):
            AbstractSymmetricMatrix(self.v1, self.v2_anti, self.v3_anti)


class AbstractSymmetricMatrixTestItem(LinAlgTestCase):
    def setUp(self) -> None:
        self.vs = (
            AbstractVector((1.0, 2.0, 3.0)),
            AbstractVector((2.0, 5.0, 6.0)),
            AbstractVector((3.0, 6.0, 9.0)),
        )

    def test_matrix_item(self) -> None:
        """Tests item related operations."""
        m = AbstractMatrix(*self.vs)
        for i, v in zip(range(3), self.vs):
            self.assertVectorEquiv(m[i], v)


class AbstractSymmetricMatrixMathTest(LinAlgTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 1e-9, 0.0))  # simulated small numerical error
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((2.2, 5.5, 6.6))
        v3 = AbstractVector((3.3, 6.6, 9.9))
        self.m1 = AbstractSymmetricMatrix(v0, v0, v0)
        self.m2 = AbstractSymmetricMatrix(v1, v2, v3)
        self.m3 = AbstractSymmetricMatrix(v1 + v0, v2, v3 + v0)
        self.m23 = AbstractSymmetricMatrix(v1 * 3.3, v2 * 3.3, v3 * 3.3)

    def test_matrix_linear(self) -> None:
        """Tests linear operations on symmetric matrices."""

        self.assertMatrixEquiv(self.m1 + self.m2, self.m2)
        self.assertMatrixEquiv(self.m2 * 3.3, self.m23)
        self.assertMatrixEquiv(self.m2 - self.m1, self.m2)
        self.assertMatrixEquiv(self.m23 / 3.3, self.m2)


class ToSymmetricMatrixTest(LinAlgTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 1e-9, 0.0))  # simulated small numerical error
        v1 = AbstractVector((1.1, 2.2, 3.3))
        v2 = AbstractVector((2.2, 5.5, 6.6))
        v3 = AbstractVector((3.3, 6.6, 9.9))
        v2_anti = AbstractVector((-2.2, 5.5, 6.6))
        v3_anti = AbstractVector((-3.3, -6.6, 9.9))

        self.symmetic_mats = (
            AbstractMatrix(v0, v0, v0),
            AbstractMatrix(v1, v2, v3),
            AbstractMatrix(v1 + v0, v2, v3 + v0),
        )

        self.non_symmetic_mats = (
            AbstractMatrix(v1, v1, v1),
            AbstractMatrix(v1, v2_anti, v3_anti),
        )

    def test_to_symmetric_matrix(self) -> None:
        """Test matrix to symmetric matrix conversion."""
        for mat in self.symmetic_mats:
            sym_mat = to_symmetric_matrix(mat)
            self.assertMatrixEquiv(sym_mat, mat)
        for mat in self.non_symmetic_mats:
            with self.assertRaises(ValueError):
                to_symmetric_matrix(mat)


class MetricTest(LinAlgTestCase):
    def setUp(self) -> None:
        v0 = AbstractVector((0.0, 0.0, 0.0))
        self.m0 = AbstractSymmetricMatrix(v0, v0, v0)
        self.m = AbstractSymmetricMatrix(
            AbstractVector((2.0, 3.0, 5.0)),
            AbstractVector((3.0, 11.0, 13.0)),
            AbstractVector((5.0, 13.0, 23.0)),
        )
        self.m_inv = inverted(self.m)

    def test_metric_constructor(self) -> None:
        """Tests metric constructor."""
        Metric(self.m)
        with self.assertRaises(ValueError):
            Metric(self.m0)

    def test_metric_matrix_getters(self) -> None:
        """Tests metric matrix getters."""
        g = Metric(self.m)
        self.assertMatrixEquiv(g.matrix(), self.m)
        self.assertMatrixEquiv(g.inverse_matrix(), self.m_inv)


class AbstractVectorIsZero(LinAlgTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 2.0, -3.0))

    def test_is_zero_vector(self) -> None:
        """Tests zero vector test."""

        self.assertTrue(is_zero_vector(self.v0))
        self.assertFalse(is_zero_vector(self.v1))


class LengthTest(LinAlgTestCase):
    def setUp(self) -> None:
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 2.0, -3.0))
        # metric
        self.metric = Metric(
            AbstractSymmetricMatrix(
                AbstractVector((5.0, 0.0, 0.0)),
                AbstractVector((0.0, 7.0, 0.0)),
                AbstractVector((0.0, 0.0, 11.0)),
            )
        )

    def test_length(self) -> None:
        """Tests vector length."""

        self.assertEquiv(length(self.v0), 0.0)
        self.assertEquiv(length(self.v1) ** 2, 14.0)

    def test_length_metric(self) -> None:
        """Tests vector length with metric."""

        self.assertEquiv(length(self.v0, metric=self.metric), 0.0)
        self.assertEquiv(length(self.v1, metric=self.metric) ** 2, 132.0)


class NormalizedTest(LinAlgTestCase):
    def setUp(self) -> None:
        self.n = AbstractVector((1.0, 1.0, 1.0)) / math.sqrt(3)
        self.w = AbstractVector((7.0, 7.0, 7.0))
        # metric
        self.metric = Metric(
            AbstractSymmetricMatrix(
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 2.0, 0.0)),
                AbstractVector((0.0, 0.0, 3.0)),
            )
        )
        self.n_metric = AbstractVector((1.0, 1.0, 1.0)) / math.sqrt(6)

    def test_normalized(self) -> None:
        """Tests vector normalization."""
        self.assertVectorEquiv(normalized(self.w), self.n)

    def test_normized_metric(self) -> None:
        """Tests vetor normalization with metric."""
        self.assertVectorEquiv(
            normalized(self.w, metric=self.metric), self.n_metric
        )


class DotTest(LinAlgTestCase):
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
            AbstractSymmetricMatrix(
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
                    self.assertEquiv(dot(v, w), 1.0)
                else:
                    self.assertEquiv(dot(v, w), 0.0)

    def test_math_dot_linearity_left(self) -> None:
        """Tests dot product's linearity in the left argument."""
        for u in self.orth_norm_basis:
            for v in self.orth_norm_basis:
                for w in self.orth_norm_basis:
                    for a in self.scalar_factors:
                        for b in self.scalar_factors:
                            self.assertEquiv(
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
                            self.assertEquiv(
                                dot(u, (v * a) + (w * b)),
                                dot(u, v) * a + dot(u, w) * b,
                            )

    def test_math_dot_metric(self) -> None:
        """Tests dot with metric."""
        for i, v in enumerate(self.orth_norm_basis):
            for j, w in enumerate(self.orth_norm_basis):
                self.assertEquiv(
                    dot(v, w, metric=self.metric), self.metric.matrix()[i][j]
                )


class CrossTest(LinAlgTestCase):
    def setUp(self) -> None:
        # standart Carthesian basis
        self.orth_norm_basis = (
            AbstractVector((1.0, 0.0, 0.0)),
            AbstractVector((0.0, 1.0, 0.0)),
            AbstractVector((0.0, 0.0, 1.0)),
        )
        # arbitrary factors
        self.scalar_factors = (0.0, 1.2345, -0.98765)

    def test_math_cross_orthonormality(self) -> None:
        """Tests cross product acting on orthonormal basis."""
        for v in self.orth_norm_basis:
            for w in self.orth_norm_basis:
                self.assertVectorEquiv(cross(v, w), -cross(w, v))

    def test_math_cross_linearity_left(self) -> None:
        """Tests cross product's linearity in the left argument."""
        for u in self.orth_norm_basis:
            for v in self.orth_norm_basis:
                for w in self.orth_norm_basis:
                    for a in self.scalar_factors:
                        for b in self.scalar_factors:
                            self.assertVectorEquiv(
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
                            self.assertVectorEquiv(
                                cross(u, (v * a) + (w * b)),
                                cross(u, v) * a + cross(u, w) * b,
                            )


class InvertedTest(LinAlgTestCase):
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
        self.assertMatrixEquiv(inverted(self.mI), self.mI)
        self.assertMatrixEquiv(inverted(self.m2), self.m2_inv)
        with self.assertRaises(ArithmeticError):
            inverted(self.m3)
        self.assertMatrixEquiv(inverted(self.m4), self.m4_inv)


class CoAndCoraviantTest(LinAlgTestCase):
    # pylint: disable=R0902

    def setUp(self) -> None:
        e0 = AbstractVector((1.0, 0.0, 0.0))
        e1 = AbstractVector((0.0, 1.0, 0.0))
        e2 = AbstractVector((0.0, 0.0, 1.0))
        orth_norm_basis = (e0, e1, e2)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1_con = AbstractVector((1.0, 2.0, 3.0))

        mI = AbstractSymmetricMatrix(*orth_norm_basis)
        self.gI = Metric(mI)

        m1 = AbstractSymmetricMatrix(e0 * 2.0, e1 * 5.0, e2 * 7.0)
        self.g1 = Metric(m1)
        self.v11_co = AbstractVector((2.0, 10.0, 21.0))

        m2 = AbstractSymmetricMatrix(
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
        self.assertVectorEquiv(covariant(self.gI, self.v0), self.v0)
        self.assertVectorEquiv(covariant(self.g1, self.v0), self.v0)
        self.assertVectorEquiv(covariant(self.g1, self.v1_con), self.v11_co)
        self.assertVectorEquiv(covariant(self.g2, self.v1_con), self.v12_co)

    def test_contra_variant(self) -> None:
        """Tests contravariant conversion."""
        self.assertVectorEquiv(contravariant(self.gI, self.v0), self.v0)
        self.assertVectorEquiv(contravariant(self.g1, self.v0), self.v0)
        self.assertVectorEquiv(contravariant(self.g1, self.v11_co), self.v1_con)
        self.assertVectorEquiv(contravariant(self.g2, self.v12_co), self.v1_con)

    def test_co_contra_variant_invertibility(self) -> None:
        """Tests if co- and contravariant invert each other."""
        for g in self.gs:
            for v in self.vs:
                self.assertVectorEquiv(covariant(g, contravariant(g, v)), v)
                self.assertVectorEquiv(contravariant(g, covariant(g, v)), v)


if __name__ == "__main__":
    unittest.main()
