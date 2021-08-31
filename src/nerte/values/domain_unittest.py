# pylint: disable=R0801
# pylint: disable=C0114
# pylint: disable=C0115

import unittest

import math

from nerte.values.domain import Domain1D


class Domain1DConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.invalid_values = (math.nan,)

    def test_domain1d_constructor(self) -> None:
        """Tests the constructor."""
        # must not be sorted by min, max
        Domain1D(-2.0, 3.0)
        Domain1D(3.0, -2.0)
        Domain1D(0.0, 1.0)
        # inf is allowed
        Domain1D(-math.inf, math.inf)
        Domain1D(math.inf, -math.inf)
        Domain1D(-math.inf, 3.0)
        Domain1D(3.0, math.inf)
        # nan ins not allowed
        for val in self.invalid_values:
            with self.assertRaises(ValueError):
                Domain1D(val, 0.0)
            with self.assertRaises(ValueError):
                Domain1D(0.0, val)
            with self.assertRaises(ValueError):
                Domain1D(val, val)
        # interval must be open and finite
        with self.assertRaises(ValueError):
            Domain1D(4.0, 4.0)
        with self.assertRaises(ValueError):
            Domain1D(math.inf, math.inf)


class Domain1DPropertiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.domain_params = (
            (1.1, 3.3),
            (math.inf, 4.4),  # intentionally in reversed order!
            (-5.5, 7.7),
        )
        self.domains = tuple(Domain1D(*ps) for ps in self.domain_params)
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

    def test_domain1d_attributes(self) -> None:
        """Tests domain parameters getters."""
        for domain, params in zip(self.domains, self.domain_params):
            self.assertTrue(domain.as_tuple() == params)

    def test_domain1d_contains(self) -> None:
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
                print(val, domain)
                self.assertFalse(val in domain)


if __name__ == "__main__":
    unittest.main()
