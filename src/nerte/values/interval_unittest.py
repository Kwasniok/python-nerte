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


class FiniteIntervalPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.parameters = (2.0, -1.0)
        self.interval = Interval(*self.parameters)
        self.included_values = (-1.0, 0.0, 2.0)
        self.excluded_values = (-1.1, 2.2 - math.inf, math.inf, math.nan)

    def test_as_tuple(self) -> None:
        """Tests as tuple conversion."""
        self.assertAlmostEqual(self.interval.as_tuple()[0], self.parameters[0])
        self.assertAlmostEqual(self.interval.as_tuple()[1], self.parameters[1])

    def test_getters(self) -> None:
        """Tests getters."""
        self.assertAlmostEqual(self.interval.start(), self.parameters[0])
        self.assertAlmostEqual(self.interval.stop(), self.parameters[1])
        self.assertAlmostEqual(self.interval.min(), min(self.parameters))
        self.assertAlmostEqual(self.interval.max(), max(self.parameters))

    def test_contains(self) -> None:
        """Tests in."""
        for val in self.included_values:
            print(val)
            self.assertTrue(val in self.interval)

    def test_contains_not(self) -> None:
        """Tests not in."""
        for val in self.excluded_values:
            self.assertFalse(val in self.interval)


class SemifiniteIntervalPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.parameters = (2.0, -math.inf)
        self.interval = Interval(*self.parameters)
        self.included_values = (-1.1, -1.0, 0.0, 2.0)
        self.excluded_values = (2.2, -math.inf, math.inf, math.nan)

    def test_as_tuple(self) -> None:
        """Tests as tuple conversion."""
        self.assertAlmostEqual(self.interval.as_tuple()[0], self.parameters[0])
        self.assertAlmostEqual(self.interval.as_tuple()[1], self.parameters[1])

    def test_getters(self) -> None:
        """Tests getters."""
        self.assertAlmostEqual(self.interval.start(), self.parameters[0])
        self.assertAlmostEqual(self.interval.stop(), self.parameters[1])
        self.assertAlmostEqual(self.interval.min(), min(self.parameters))
        self.assertAlmostEqual(self.interval.max(), max(self.parameters))

    def test_contains(self) -> None:
        """Tests in."""
        for val in self.included_values:
            print(val)
            self.assertTrue(val in self.interval)

    def test_contains_not(self) -> None:
        """Tests not in."""
        for val in self.excluded_values:
            self.assertFalse(val in self.interval)


class InfiniteIntervalPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.parameters = (math.inf, -math.inf)
        self.interval = Interval(*self.parameters)
        self.included_values = (-1.1, -1.0, 0.0, 2.0, 2.2)
        self.excluded_values = (-math.inf, math.inf, math.nan)

    def test_as_tuple(self) -> None:
        """Tests as tuple conversion."""
        self.assertAlmostEqual(self.interval.as_tuple()[0], self.parameters[0])
        self.assertAlmostEqual(self.interval.as_tuple()[1], self.parameters[1])

    def test_getters(self) -> None:
        """Tests getters."""
        self.assertAlmostEqual(self.interval.start(), self.parameters[0])
        self.assertAlmostEqual(self.interval.stop(), self.parameters[1])
        self.assertAlmostEqual(self.interval.min(), min(self.parameters))
        self.assertAlmostEqual(self.interval.max(), max(self.parameters))

    def test_contains(self) -> None:
        """Tests in."""
        for val in self.included_values:
            print(val)
            self.assertTrue(val in self.interval)

    def test_contains_not(self) -> None:
        """Tests not in."""
        for val in self.excluded_values:
            self.assertFalse(val in self.interval)


if __name__ == "__main__":
    unittest.main()
