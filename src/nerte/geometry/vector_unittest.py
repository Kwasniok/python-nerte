# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144


import unittest
import math
from nerte.geometry.vector import AbstractVector, dot, cross, length, normalized


# equivalence of floating point representations with finite precision
𝜀 = 1e-8
# True, iff two floats agree up to the (absolute) precision 𝜀
equiv = lambda x, y: abs(x - y) < 𝜀
# True, iff two vectors component-wise agree up to the (absolute) precision 𝜀
vec_equiv = lambda x, y: all(equiv(i, j) for i, j in zip(x, y))

# standart Carthesian basis
orth_norm_basis = (
    AbstractVector(1.0, 0.0, 0.0),
    AbstractVector(0.0, 1.0, 0.0),
    AbstractVector(0.0, 0.0, 1.0),
)

# arbitrary factors
scalar_factors = (0.0, 1.2345, -0.98765)


class VectorTest(unittest.TestCase):
    def assertEquiv(self, x, y):
        """
        Asserts the equivalence of two floats.
        Note: This replaces assertTrue(x == y) for float.
        """
        try:
            self.assertTrue(equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Scalar {} is not equivalent to {}.".format(x, y)
            ) from ae

    def assertVectorEquiv(self, x, y):
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for nerte.Vector.
        """
        try:
            self.assertTrue(vec_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Vector {} is not equivalent to {}.".format(x, y)
            ) from ae

    def test_item(self):
        """Tests item related operations."""
        cs = (1.0, 2.0, 3.0)
        v = AbstractVector(*cs)
        for x, i in zip(iter(v), range(3)):
            self.assertEquiv(x, v[i])
        for x, y in zip(iter(v), cs):
            self.assertEquiv(x, y)

    def test_math_linear(self):
        """Tests linear operations on vectors."""
        v1 = AbstractVector(1.1, 2.2, 3.3)
        v2 = AbstractVector(4.4, 5.5, 6.6)
        v3 = AbstractVector(5.5, 7.7, 9.9)
        v4 = AbstractVector(1.0, 1.0, 1.0)
        v5 = AbstractVector(3.3, 3.3, 3.3)

        self.assertVectorEquiv(v1 + v2, v3)
        self.assertVectorEquiv(v4 * 3.3, v5)
        self.assertVectorEquiv(v2 - v1, v5)
        self.assertVectorEquiv(v5 / 3.3, v4)

    def test_math_length(self):
        """Tests vector length."""
        v0 = AbstractVector(0.0, 0.0, 0.0)
        v1 = AbstractVector(1.0, 2.0, -3.0)

        self.assertEquiv(length(v0), 0.0)
        self.assertEquiv(length(v1) ** 2, 14.0)

    def test_math_normalized(self):
        """Tests vector normalization."""
        n = AbstractVector(1.0, 1.0, 1.0) / math.sqrt(3)
        w = AbstractVector(7.0, 7.0, 7.0)
        self.assertVectorEquiv(normalized(w), n)

    def test_math_dot_orthonormality(self):
        """Tests dot product acting on orthonormal basis."""
        for v in orth_norm_basis:
            for w in orth_norm_basis:
                if v is w:
                    self.assertEquiv(dot(v, w), 1.0)
                else:
                    self.assertEquiv(dot(v, w), 0.0)

    def test_math_dot_linearity_left(self):
        """Tests dot product's linearity in the left argument."""
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertEquiv(
                                dot((v * a) + (w * b), u),
                                dot(v, u) * a + dot(w, u) * b,
                            )

    def test_math_dot_linearity_right(self):
        """Tests dot product's linearity in the right argument."""
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertEquiv(
                                dot(u, (v * a) + (w * b)),
                                dot(u, v) * a + dot(u, w) * b,
                            )

    def test_math_cross_orthonormality(self):
        """Tests cross product acting on orthonormal basis."""
        for v in orth_norm_basis:
            for w in orth_norm_basis:
                self.assertVectorEquiv(cross(v, w), -cross(w, v))

    def test_math_cross_linearity_left(self):
        """Tests cross product's linearity in the left argument."""
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertVectorEquiv(
                                cross((v * a) + (w * b), u),
                                cross(v, u) * a + cross(w, u) * b,
                            )

    def test_math_cross_linearity_right(self):
        """Tests cross product's linearity in the right argument."""
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertVectorEquiv(
                                cross(u, (v * a) + (w * b)),
                                cross(u, v) * a + cross(u, w) * b,
                            )


if __name__ == "__main__":
    unittest.main()
