# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115


import unittest

from typing import Callable, TypeVar, Optional

import math
import numpy as np

from nerte.algorithm.runge_kutta import runge_kutta_4_delta

T = TypeVar("T")


# apply function n times
def _iterate(f: Callable[[T], T], n: int, x0: T) -> T:
    x = x0
    for _ in range(n):
        x = f(x)
    return x


# True, iff two floats are equivalent
def _equiv(
    x: float,
    y: float,
    rel_tol: Optional[float] = None,
    abs_tol: Optional[float] = None,
) -> bool:
    if rel_tol is None:
        if abs_tol is None:
            return math.isclose(x, y)
        return math.isclose(x, y, abs_tol=abs_tol)
    if abs_tol is None:
        return math.isclose(x, y, rel_tol=rel_tol)
    return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)


class RungeKuttaTestCase(unittest.TestCase):
    def assertEquiv(
        self,
        x: float,
        y: float,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None,
    ) -> None:
        """
        Asserts the equivalence of two floats.
        Note: This replaces assertTrue(x == y) for float.
        """
        try:
            self.assertTrue(_equiv(x, y, rel_tol, abs_tol))
        except AssertionError as ae:
            raise AssertionError(
                "Scalar {} is not equivalent to {}.".format(x, y)
            ) from ae


class RungeKutta4DeltaFreeTest(RungeKuttaTestCase):
    def setUp(self) -> None:
        self.x0 = np.array((1.0, 1.0))
        # force-free propagation (in carthesian coordinates)
        self.f = lambda x: np.array((x[1], 0.0))
        self.time_step_size = 0.1
        self.total_time = 1.0
        self.x1 = np.array((2.0, 1.0))

    def test_runge_kutta_4_delta_free(self) -> None:
        """Test the implementation of the Runge-Kutta 4 algorithm."""
        steps = int(self.total_time // self.time_step_size) + 1
        x1 = _iterate(
            lambda x: x + runge_kutta_4_delta(self.f, x, self.time_step_size),
            steps,
            self.x0,
        )
        self.assertEquiv(x1[0], self.x1[0])
        self.assertEquiv(x1[1], self.x1[1])


class RungeKutta4DeltaConstantForceTest(RungeKuttaTestCase):
    def setUp(self) -> None:
        self.x0 = np.array((0.0, 0.0))
        # constant force propagation (in carthesian coordinates)
        self.f = lambda x: np.array((x[1], 1.0))
        self.time_step_size = 0.1
        self.total_time = 1.0
        self.x1 = np.array((0.5, 1.0))

    def test_runge_kutta_4_delta_const_acceleration(self) -> None:
        """Test the implementation of the Runge-Kutta 4 algorithm."""
        steps = int(self.total_time // self.time_step_size) + 1
        x1 = _iterate(
            lambda x: x + runge_kutta_4_delta(self.f, x, self.time_step_size),
            steps,
            self.x0,
        )
        self.assertEquiv(x1[0], self.x1[0])
        self.assertEquiv(x1[1], self.x1[1])


class RungeKutta4DeltaHamonicOscillatorTest(RungeKuttaTestCase):
    def setUp(self) -> None:
        self.x0 = np.array((1.0, 0.0))
        # harmonic oscillator (in carthesian coordinates)
        self.f = lambda x: np.array((x[1], -x[0]))
        self.time_step_size = 0.001
        self.total_time = 2 * math.pi
        self.x1 = np.array((1.0, 0.0))

    def test_runge_kutta_4_delta_const_acceleration(self) -> None:
        """Test the implementation of the Runge-Kutta 4 algorithm."""
        steps = int(self.total_time // self.time_step_size) + 1
        x1 = _iterate(
            lambda x: x + runge_kutta_4_delta(self.f, x, self.time_step_size),
            steps,
            self.x0,
        )
        self.assertEquiv(x1[0], self.x1[0], rel_tol=1e-5)
        self.assertEquiv(x1[1], self.x1[1], abs_tol=1e-3)
