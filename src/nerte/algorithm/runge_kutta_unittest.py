# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115


import unittest

from typing import Callable, TypeVar

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


class RungeKutta4DeltaFreeTest(unittest.TestCase):
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
        self.assertAlmostEqual(x1[0], self.x1[0])
        self.assertAlmostEqual(x1[1], self.x1[1])


class RungeKutta4DeltaConstantForceTest(unittest.TestCase):
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
        self.assertAlmostEqual(x1[0], self.x1[0])
        self.assertAlmostEqual(x1[1], self.x1[1])


class RungeKutta4DeltaHamonicOscillatorTest(unittest.TestCase):
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
        self.assertAlmostEqual(x1[0], self.x1[0], places=5)
        self.assertAlmostEqual(x1[1], self.x1[1], places=2)
