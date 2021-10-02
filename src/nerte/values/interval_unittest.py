# pylint: disable=R0801
# pylint: disable=C0114
# pylint: disable=C0115

import unittest

import math

from nerte.base_test_case import BaseTestCase

from nerte.values.interval import Interval


class IntervalConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.invalid_values = (math.nan,)

    def test_constructor(self) -> None:
        """Tests the constructor."""
        # must not be sorted by min, max
        Interval(-2.0, 3.0)
        Interval(3.0, -2.0)
        Interval(0.0, 1.0)
        # inf is allowed
        Interval(-math.inf, math.inf)
        Interval(math.inf, -math.inf)
        Interval(-math.inf, 3.0)
        Interval(3.0, math.inf)
        # nan ins not allowed
        for val in self.invalid_values:
            with self.assertRaises(ValueError):
                Interval(val, 0.0)
            with self.assertRaises(ValueError):
                Interval(0.0, val)
            with self.assertRaises(ValueError):
                Interval(val, val)
        # interval must be open and finite
        with self.assertRaises(ValueError):
            Interval(4.0, 4.0)
        with self.assertRaises(ValueError):
            Interval(math.inf, math.inf)


class IntervalPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain_params = (
            (1.1, 3.3),
            (math.inf, 4.4),  # intentionally in reversed order!
            (-5.5, 7.7),
        )
        self.domains = tuple(Interval(*ps) for ps in self.domain_params)
        self.included_values = (
            (2.0, 2.2, 3.0),
            (5.5, 11.11, 1e8),
            (-3.3, 2.2, 5.5),
        )
        self.excluded_values = (
            (-2.2, -1.1),
            (-5.5, -math.inf),
            (-7.7, -math.inf, math.inf),
        )

    def test_attributes(self) -> None:
        """Tests domain parameters getters."""
        for domain, params in zip(self.domains, self.domain_params):
            self.assertAlmostEqual(domain.as_tuple()[0], params[0])
            self.assertAlmostEqual(domain.as_tuple()[1], params[1])
            self.assertAlmostEqual(domain.start(), params[0])
            self.assertAlmostEqual(domain.stop(), params[1])
            self.assertAlmostEqual(domain.min(), min(params))
            self.assertAlmostEqual(domain.max(), max(params))

    def test_contains(self) -> None:
        """Tests domain contains."""
        for domain, params, incl_vals, excl_vals in zip(
            self.domains,
            self.domain_params,
            self.included_values,
            self.excluded_values,
        ):
            # boundaries are included
            for val in params:
                self.assertTrue(val in domain)
            for val in incl_vals:
                self.assertTrue(val in domain)
            for val in excl_vals:
                self.assertFalse(val in domain)


if __name__ == "__main__":
    unittest.main()
