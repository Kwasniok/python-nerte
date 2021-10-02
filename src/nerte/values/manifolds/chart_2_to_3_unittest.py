# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Type
import itertools

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.manifolds.chart_2_to_3 import Chart2DTo3D


def _dummy_chart_2d_3d_class() -> Type[Chart2DTo3D]:
    class DummyChart2DTo3D(Chart2DTo3D):
        def __init__(self) -> None:
            interval = Interval(-1, 1)
            self.domain = (interval, interval)

        def is_in_domain(self, coords: Coordinates2D) -> bool:
            return coords[0] in self.domain[0] and coords[1] in self.domain[1]

        def out_of_domain_reason(self, coords: Coordinates2D) -> str:
            return (
                f"Coordinate {coords} not inside Cartesian product of"
                f" intervals {self.domain[0]}x{self.domain[1]}."
            )

        def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
            return Coordinates3D((coords[0], coords[1], 0.0))

        def internal_hook_tangential_space(
            self, coords: Coordinates2D
        ) -> tuple[AbstractVector, AbstractVector]:
            return (
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 1.0, 0.0)),
            )

        def internal_hook_surface_normal(
            self, coords: Coordinates2D
        ) -> AbstractVector:
            return AbstractVector((0.0, 0.0, 1.0))

    return DummyChart2DTo3D


class Chart2DTo3DImplementationTest(BaseTestCase):
    def test_implementation(self) -> None:
        """Tests chart interface implementation."""
        # pylint: disable=R0201
        _dummy_chart_2d_3d_class()


class Chart2DTo3DDomainTest(BaseTestCase):
    def setUp(self) -> None:
        self.DummyChart2DTo3D = _dummy_chart_2d_3d_class()
        self.coord_inside_domain = (-1.0, 0.0, 1.0)
        self.coord_outside_domain = (-2.0, 2.0)

    def test_domain(self) -> None:
        """Test domain properties."""
        man = self.DummyChart2DTo3D()
        i = self.coord_inside_domain
        o = self.coord_outside_domain
        for xs, ys in itertools.product((i, o), (i, o)):
            if xs == ys == i:
                for x, y in zip(xs, ys):
                    man.embed(Coordinates2D((x, y)))
                    man.tangential_space(Coordinates2D((x, y)))
            else:
                for x, y in zip(xs, ys):
                    with self.assertRaises(OutOfDomainError):
                        man.embed(Coordinates2D((x, y)))
                    with self.assertRaises(OutOfDomainError):
                        man.tangential_space(Coordinates2D((x, y)))


if __name__ == "__main__":
    unittest.main()
