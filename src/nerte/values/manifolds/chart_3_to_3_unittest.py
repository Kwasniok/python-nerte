# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from typing import Type
import itertools

from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector, AbstractMatrix, Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.util.convert import coordinates_as_vector
from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.manifolds.chart_3_to_3 import Chart3DTo3D


def _dummy_chart_3d_3d_class() -> Type[Chart3DTo3D]:
    class DummyChart3DTo3D(Chart3DTo3D):
        def __init__(self) -> None:
            interval = Interval(-1, 1)
            self.domain = (interval, interval, interval)

        def is_in_domain(self, coords: Coordinates3D) -> bool:
            return (
                coords[0] in self.domain[0]
                and coords[1] in self.domain[1]
                and coords[2] in self.domain[2]
            )

        def out_of_domain_reason(self, coords: Coordinates3D) -> str:
            return (
                f"Coordinate {coords} not inside Cartesian product of"
                f" intervals {self.domain[0]}x{self.domain[1]}x{self.domain[2]}."
            )

        def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
            return coords

        def internal_hook_tangential_space(
            self, coords: Coordinates3D
        ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
            return (
                AbstractVector((1.0, 0.0, 0.0)),
                AbstractVector((0.0, 1.0, 0.0)),
                AbstractVector((0.0, 0.0, 1.0)),
            )

        def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
            return Metric(
                AbstractMatrix(
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 1.0, 0.0)),
                    AbstractVector((0.0, 0.0, 1.0)),
                )
            )

        def internal_hook_geodesics_equation(
            self, tangent: TangentialVector
        ) -> TangentialVectorDelta:
            return TangentialVectorDelta(
                tangent.vector, AbstractVector((0.0, 0.0, 0.0))
            )

        def internal_hook_initial_geodesic_tangent_from_coords(
            self, start: Coordinates3D, target: Coordinates3D
        ) -> TangentialVector:
            return TangentialVector(
                start,
                coordinates_as_vector(target) - coordinates_as_vector(start),
            )

    return DummyChart3DTo3D


class Chart3DTo3DImplementationTest(BaseTestCase):
    def test_implementation(self) -> None:
        """Tests chart interface implementation."""
        # pylint: disable=R0201
        _dummy_chart_3d_3d_class()


class Chart3DTo3DDomainTest(BaseTestCase):
    def setUp(self) -> None:
        self.DummyChart3DTo3D = _dummy_chart_3d_3d_class()
        self.coord_inside_domain = (-1.0, 0.0, 1.0)
        self.coord_outside_domain = (-2.0, 2.0, 2.0)
        self.vec = AbstractVector((1.0, 0.0, 0.0))

    def test_implementation(self) -> None:
        """Tests domain properties."""
        man = self.DummyChart3DTo3D()
        i = self.coord_inside_domain
        o = self.coord_outside_domain
        for xs, ys, zs in itertools.product((i, o), (i, o), (i, o)):
            if xs == ys == zs == i:
                for x, y, z in zip(xs, ys, zs):
                    coords = Coordinates3D((x, y, z))
                    coords_in = Coordinates3D(i)
                    tangent = TangentialVector(coords, self.vec)
                    man.embed(coords)
                    man.tangential_space(coords)
                    man.metric(coords)
                    man.scalar_product(coords, self.vec, self.vec)
                    man.length(tangent)
                    man.normalized(tangent)
                    man.geodesics_equation(tangent)
                    man.initial_geodesic_tangent_from_coords(coords_in, coords)
                    man.initial_geodesic_tangent_from_coords(coords, coords_in)
            else:
                for x, y, z in zip(xs, ys, zs):
                    coords = Coordinates3D((x, y, z))
                    coords_in = Coordinates3D(i)
                    tangent = TangentialVector(coords, self.vec)
                    with self.assertRaises(OutOfDomainError):
                        man.embed(coords)
                    with self.assertRaises(OutOfDomainError):
                        man.tangential_space(coords)
                    with self.assertRaises(OutOfDomainError):
                        man.metric(coords)
                    with self.assertRaises(OutOfDomainError):
                        man.scalar_product(coords, self.vec, self.vec)
                    with self.assertRaises(OutOfDomainError):
                        man.length(tangent)
                    with self.assertRaises(OutOfDomainError):
                        man.normalized(tangent)
                    with self.assertRaises(OutOfDomainError):
                        man.geodesics_equation(
                            TangentialVector(coords, self.vec)
                        )
                    with self.assertRaises(OutOfDomainError):
                        man.initial_geodesic_tangent_from_coords(
                            coords_in, coords
                        )
                    with self.assertRaises(OutOfDomainError):
                        man.initial_geodesic_tangent_from_coords(
                            coords, coords_in
                        )


if __name__ == "__main__":
    unittest.main()
