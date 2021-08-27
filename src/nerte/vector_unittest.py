import unittest
import math
from nerte.vector import Vector


# equivalence of floating point representations with finite precision
𝜀 = 1e-8
equiv = lambda x, y: abs(x - y) < 𝜀
vec_equiv = lambda x, y: all(equiv(i, j) for i, j in zip(x, y))

orth_norm_basis = (
    Vector(1.0, 0.0, 0.0),
    Vector(0.0, 1.0, 0.0),
    Vector(0.0, 0.0, 1.0),
)

scalar_factors = (0.0, 1.2345, -0.98765)


class VectorTest(unittest.TestCase):

    # auxliliar functions to test for equivalence
    def assertEquiv(self, x, y):
        try:
            self.assertTrue(equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Scalar {} is not equivalent to {}.".format(x, y)
            ) from ae

    def assertVectorEquiv(self, x, y):
        try:
            self.assertTrue(vec_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Vector {} is not equivalent to {}.".format(x, y)
            ) from ae

    def test_item(self):
        cs = (1.0, 2.0, 3.0)
        v = Vector(*cs)
        for x, i in zip(iter(v), range(3)):
            self.assertEquiv(x, v[i])
        for x, y in zip(iter(v), cs):
            self.assertEquiv(x, y)

    def test_math_linear(self):
        v1 = Vector(1.1, 2.2, 3.3)
        v2 = Vector(4.4, 5.5, 6.6)
        v3 = Vector(5.5, 7.7, 9.9)
        v4 = Vector(1.0, 1.0, 1.0)
        v5 = Vector(3.3, 3.3, 3.3)

        self.assertVectorEquiv(v1 + v2, v3)
        self.assertVectorEquiv(v4 * 3.3, v5)
        self.assertVectorEquiv(v2 - v1, v5)
        self.assertVectorEquiv(v5 / 3.3, v4)

    def test_math_length(self):
        v0 = Vector(0.0, 0.0, 0.0)
        v1 = Vector(1.0, 2.0, -3.0)

        self.assertEquiv(v0.length(), 0.0)
        self.assertEquiv(v1.length() ** 2, 14.0)

    def test_math_normalized(self):
        n = Vector(1.0, 1.0, 1.0) / math.sqrt(3)
        w = Vector(7.0, 7.0, 7.0)
        self.assertVectorEquiv(w.normalized(), n)

    def test_math_dot_orthonormality(self):
        for v in orth_norm_basis:
            for w in orth_norm_basis:
                if v is w:
                    self.assertEquiv(v.dot(w), 1.0)
                else:
                    self.assertEquiv(v.dot(w), 0.0)

    def test_math_dot_linearity_left(self):
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertEquiv(
                                ((v * a) + (w * b)).dot(u),
                                v.dot(u) * a + w.dot(u) * b,
                            )

    def test_math_dot_linearity_right(self):
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertEquiv(
                                u.dot((v * a) + (w * b)),
                                u.dot(v) * a + u.dot(w) * b,
                            )

    def test_math_cross_orthonormality(self):
        for v in orth_norm_basis:
            for w in orth_norm_basis:
                self.assertVectorEquiv(v.cross(w), -w.cross(v))

    def test_math_cross_linearity_left(self):
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertVectorEquiv(
                                ((v * a) + (w * b)).cross(u),
                                v.cross(u) * a + w.cross(u) * b,
                            )

    def test_math_cross_linearity_right(self):
        for u in orth_norm_basis:
            for v in orth_norm_basis:
                for w in orth_norm_basis:
                    for a in scalar_factors:
                        for b in scalar_factors:
                            self.assertVectorEquiv(
                                u.cross((v * a) + (w * b)),
                                u.cross(v) * a + u.cross(w) * b,
                            )


if __name__ == "__main__":
    unittest.main()
