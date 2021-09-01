# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector, cross
from nerte.values.manifold import Manifold2D, OutOfDomainError, Plane
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)

# equivalence of floating point representations with finite precision
𝜀 = 1e-8
# True, iff two floats agree up to the (absolute) precision 𝜀
def _equiv(x: float, y: float) -> bool:
    return abs(x - y) < 𝜀


# True, iff two coordinates component-wise agree up to the (absolute) precision 𝜀
def _coords_equiv(x: Coordinates3D, y: Coordinates3D) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


# True, iff two vectors component-wise agree up to the (absolute) precision 𝜀
def _vec_equiv(x: AbstractVector, y: AbstractVector) -> bool:
    return _equiv(x[0], y[0]) and _equiv(x[1], y[1]) and _equiv(x[2], y[2])


class ManifoldUnittest(unittest.TestCase):
    def assertCoordinates3DEquiv(self, x: Coordinates3D, y: Coordinates3D) -> None:
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_coords_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Coordinates3D {} are not equivalent to {}.".format(x, y)
            ) from ae

    def assertVectorEquiv(self, x: AbstractVector, y: AbstractVector) -> None:
        """
        Asserts ths equivalence of two vectors.
        Note: This replaces assertTrue(x == y) for vectors.
        """
        try:
            self.assertTrue(_vec_equiv(x, y))
        except AssertionError as ae:
            raise AssertionError(
                "Vector {} is not equivalent to {}.".format(x, y)
            ) from ae


class Manifold2DImplementationTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 1.0)

    def test_implementation(self) -> None:
        """Tests manifold interface implementation."""
        # x-y plane
        class DummyManifold2D(Manifold2D):
            def __init__(
                self,
                x0_domain: Domain1D,
                x1_domain: Domain1D,
            ):
                Manifold2D.__init__(
                    self, x0_domain=x0_domain, x1_domain=x1_domain
                )

            def coordinates(self, coords: Coordinates2D) -> Coordinates3D:
                return Coordinates3D((coords[0], coords[1], 0.0))

            def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
                return AbstractVector(0.0, 0.0, 1.0)

            def tangential_space(
                self, coords: Coordinates2D
            ) -> tuple[AbstractVector, AbstractVector]:
                return (
                    AbstractVector(1.0, 0.0, 0.0),
                    AbstractVector(0.0, 1.0, 0.0),
                )

        man = DummyManifold2D(self.domain, self.domain)
        for x, y in ((i, j) for i in range(-10, 11) for j in range(-10, 11)):
            man.coordinates(Coordinates2D((x, y)))

        self.assertTrue(man.x0_domain() is self.domain)
        self.assertTrue(man.x1_domain() is self.domain)


class PlaneConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector(0.0, 0.0, 0.0)
        self.v1 = AbstractVector(1.0, 0.0, 0.0)
        self.v2 = AbstractVector(0.0, 1.0, 0.0)
        self.offset = AbstractVector(0.0, 0.0, 0.0)

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        Plane(b0=self.v1, b1=self.v2)
        Plane(b0=self.v1, b1=self.v2, offset=self.offset)
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v1)
        with self.assertRaises(ValueError):
            Plane(self.v1, self.v0)
        with self.assertRaises(ValueError):
            Plane(self.v0, self.v0)


class PlaneDomainTest(ManifoldUnittest):
    def setUp(self) -> None:
        v1 = AbstractVector(1.0, 0.0, 0.0)
        v2 = AbstractVector(0.0, 1.0, 0.0)
        self.finite_plane = Plane(
            v1, v2, x0_domain=Domain1D(-1.0, 2.0), x1_domain=Domain1D(3.0, -4.0)
        )
        self.infinite_plane = Plane(v1, v2)
        self.coords = (
            Coordinates2D((-2.0, -2.0)),
            Coordinates2D((3.0, -2.0)),
            Coordinates2D((1.0, -5.0)),
            Coordinates2D((1.0, 4.0)),
        )

    def test_plane_coordinates_domain(self) -> None:
        """Tests plane coordinates."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.coordinates(coords)
        for coords in self.coords:
            self.infinite_plane.coordinates(coords)

    def test_plane_surface_normal_domain(self) -> None:
        """Tests plane's surface normal."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.surface_normal(coords)
        for coords in self.coords:
            self.infinite_plane.surface_normal(coords)

    def test_plane_tangential_space_domain(self) -> None:
        """Tests plane's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.tangential_space(coords)
        for coords in self.coords:
            self.infinite_plane.tangential_space(coords)


class PlanePropertiesTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.v1 = AbstractVector(1.0, 0.0, 0.0)
        self.v2 = AbstractVector(0.0, 1.0, 0.0)
        self.n = cross(self.v1, self.v2)
        self.offsets = (
            AbstractVector(0.0, 0.0, 0.0),
            AbstractVector(1.1, 2.2, 3.3),
        )
        self.planes = tuple(
            Plane(self.v1, self.v2, offset=o) for o in self.offsets
        )
        c2d_0 = Coordinates2D((0.0, 0.0))
        c2d_1 = Coordinates2D((1.0, 0.0))
        c2d_2 = Coordinates2D((0.0, 1.0))
        c2d_3 = Coordinates2D((2.0, -3.0))
        c3d_0 = Coordinates3D((0.0, 0.0, 0.0))
        c3d_1 = Coordinates3D((1.0, 0.0, 0.0))
        c3d_2 = Coordinates3D((0.0, 1.0, 0.0))
        c3d_3 = Coordinates3D((2.0, -3.0, 0.0))
        self.coords_2d = (c2d_0, c2d_1, c2d_2, c2d_3)
        self.coords_3d = (c3d_0, c3d_1, c3d_2, c3d_3)

    def test_plane_coordinates(self) -> None:
        """Tests plane coordinates."""
        for plane, offset in zip(self.planes, self.offsets):
            for c2d, c3d in zip(self.coords_2d, self.coords_3d):
                self.assertCoordinates3DEquiv(
                    plane.coordinates(c2d),
                    vector_as_coordinates(coordinates_as_vector(c3d) + offset),
                )

    def test_plane_surface_normal(self) -> None:
        """Tests plane's surface normal."""
        for plane in self.planes:
            for c2d in self.coords_2d:
                self.assertVectorEquiv(plane.surface_normal(c2d), self.n)

    def test_plane_tangential_space(self) -> None:
        """Tests plane's tangential space."""
        for plane in self.planes:
            for c2d in self.coords_2d:
                b0, b1 = plane.tangential_space(c2d)
                self.assertVectorEquiv(b0, self.v1)
                self.assertVectorEquiv(b1, self.v2)


if __name__ == "__main__":
    unittest.main()
