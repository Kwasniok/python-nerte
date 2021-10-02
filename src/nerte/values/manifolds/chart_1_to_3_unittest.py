# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Type

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates1D, Coordinates3D
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector
from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.manifolds.chart_1_to_3 import Chart1DTo3D


def _dummy_chart_1d_3d_class() -> Type[Chart1DTo3D]:
    class DummyChart1DTo3D(Chart1DTo3D):
        def __init__(self) -> None:
            self.domain = Interval(-1, 1)

        def is_in_domain(self, coords: Coordinates1D) -> bool:
            return coords[0] in self.domain

        def out_of_domain_reason(self, coords: Coordinates1D) -> str:
            return f"Coordinate {coords[0]} not inside interval {self.domain}."

        def internal_hook_embed(self, coords: Coordinates1D) -> Coordinates3D:
            return Coordinates3D((coords[0], 0.0, 0.0))

        def internal_hook_tangential_space(
            self, coords: Coordinates1D
        ) -> AbstractVector:
            return AbstractVector((1.0, 0.0, 0.0))

    return DummyChart1DTo3D


class Chart1DTo3DImplementationTest(BaseTestCase):
    def test_implementation(self) -> None:
        """Tests chart interface implementation."""
        # pylint: disable=R0201
        _dummy_chart_1d_3d_class()


class Chart1DTo3DDomainTest(BaseTestCase):
    def setUp(self) -> None:
        self.DummyChart1DTo3D = _dummy_chart_1d_3d_class()
        self.coord_inside_domain = (-1.0, 0.0, 1.0)
        self.coord_outside_domain = (-2.0, 2.0)

    def test_domain(self) -> None:
        """Test domain properties."""
        man = self.DummyChart1DTo3D()
        for x in self.coord_inside_domain:
            man.embed(Coordinates1D((x,)))
            man.tangential_space(Coordinates1D((x,)))
        for x in self.coord_outside_domain:
            with self.assertRaises(OutOfDomainError):
                man.embed(Coordinates1D((x,)))
            with self.assertRaises(OutOfDomainError):
                man.tangential_space(Coordinates1D((x,)))


if __name__ == "__main__":
    unittest.main()
