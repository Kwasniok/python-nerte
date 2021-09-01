# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import itertools

from nerte.values.coordinates import Coordinates1D, Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector, cross
from nerte.values.manifold import (
    Manifold1D,
    Manifold2D,
    Manifold3D,
    OutOfDomainError,
    Line,
    Plane,
    Parallelepiped,
)
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
    def assertCoordinates3DEquiv(
        self, x: Coordinates3D, y: Coordinates3D
    ) -> None:
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


class Manifold1DImplementationTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 1.0)
        self.coord_inside_domain = (-1.0, 0.0, 1.0)
        self.coord_outside_domain = (-2.0, 2.0)

    def test_implementation(self) -> None:
        """Tests manifold interface implementation."""
        # x-y plane
        class DummyManifold1D(Manifold1D):
            def __init__(
                self,
                domain: Domain1D,
            ):
                Manifold1D.__init__(self, domain)

            def embed(self, coords: Coordinates1D) -> Coordinates3D:
                self.in_domain_assertion(coords)
                return Coordinates3D((coords, 0.0, 0.0))

            def tangential_space(self, coords: Coordinates1D) -> AbstractVector:
                self.in_domain_assertion(coords)
                return AbstractVector((1.0, 0.0, 0.0))

        man = DummyManifold1D(self.domain)
        for x in self.coord_inside_domain:
            man.embed(Coordinates1D(x))
            man.tangential_space(Coordinates1D(x))
        for x in self.coord_outside_domain:
            with self.assertRaises(OutOfDomainError):
                man.embed(Coordinates1D(x))
            with self.assertRaises(OutOfDomainError):
                man.tangential_space(Coordinates1D(x))

        self.assertTrue(man.domain is self.domain)


class Manifold2DImplementationTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 1.0)
        self.coord_inside_domain = (-1.0, 0.0, 1.0)
        self.coord_outside_domain = (-2.0, 2.0)

    def test_implementation(self) -> None:
        """Tests manifold interface implementation."""
        # x-y plane
        class DummyManifold2D(Manifold2D):
            def __init__(
                self,
                x0_domain: Domain1D,
                x1_domain: Domain1D,
            ):
                Manifold2D.__init__(self, (x0_domain, x1_domain))

            def embed(self, coords: Coordinates2D) -> Coordinates3D:
                self.in_domain_assertion(coords)
                return Coordinates3D((coords[0], coords[1], 0.0))

            def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
                self.in_domain_assertion(coords)
                return AbstractVector((0.0, 0.0, 1.0))

            def tangential_space(
                self, coords: Coordinates2D
            ) -> tuple[AbstractVector, AbstractVector]:
                self.in_domain_assertion(coords)
                return (
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 1.0, 0.0)),
                )

        man = DummyManifold2D(self.domain, self.domain)
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

        self.assertTrue(len(man.domain) == 2)
        self.assertTrue(man.domain[0] is self.domain)
        self.assertTrue(man.domain[1] is self.domain)


class Manifold3DImplementationTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 1.0)
        self.coord_inside_domain = (-1.0, 0.0, 1.0)
        self.coord_outside_domain = (-2.0, 2.0)

    def test_implementation(self) -> None:
        """Tests manifold interface implementation."""
        # x-y plane
        class DummyManifold3D(Manifold3D):
            def __init__(
                self,
                x0_domain: Domain1D,
                x1_domain: Domain1D,
                x2_domain: Domain1D,
            ):
                Manifold3D.__init__(self, (x0_domain, x1_domain, x2_domain))

            def embed(self, coords: Coordinates3D) -> Coordinates3D:
                self.in_domain_assertion(coords)
                return Coordinates3D(coords)

            def tangential_space(
                self, coords: Coordinates3D
            ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
                self.in_domain_assertion(coords)
                return (
                    AbstractVector((1.0, 0.0, 0.0)),
                    AbstractVector((0.0, 1.0, 0.0)),
                    AbstractVector((0.0, 0.0, 1.0)),
                )

        man = DummyManifold3D(self.domain, self.domain, self.domain)
        i = self.coord_inside_domain
        o = self.coord_outside_domain
        for xs, ys, zs in itertools.product((i, o), (i, o), (i, o)):
            if xs == ys == zs == i:
                for x, y, z in zip(xs, ys, zs):
                    man.embed(Coordinates3D((x, y, z)))
                    man.tangential_space(Coordinates3D((x, y, z)))
            else:
                for x, y, z in zip(xs, ys, zs):
                    with self.assertRaises(OutOfDomainError):
                        man.embed(Coordinates3D((x, y, z)))
                    with self.assertRaises(OutOfDomainError):
                        man.tangential_space(Coordinates3D((x, y, z)))

        self.assertTrue(len(man.domain) == 3)
        self.assertTrue(man.domain[0] is self.domain)
        self.assertTrue(man.domain[1] is self.domain)
        self.assertTrue(man.domain[2] is self.domain)


class LineConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_plane_constructor(self) -> None:
        """Tests plane constroctor."""
        Line(direction=self.v1)
        Line(direction=self.v1, offset=self.offset)
        with self.assertRaises(ValueError):
            Line(self.v0)


class LineDomainTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.finite_line = Line(self.v1, Domain1D(-1.0, 2.0))
        self.infinite_line = Line(self.v1)
        self.coords = (Coordinates1D(-2.0), Coordinates1D(3.0))

    def test_line_embed_domain(self) -> None:
        """Tests line coordinates."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_line.embed(coords)
        for coords in self.coords:
            self.infinite_line.embed(coords)

    def test_line_tangential_space_domain(self) -> None:
        """Tests line's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_line.tangential_space(coords)
        for coords in self.coords:
            self.infinite_line.tangential_space(coords)


class LinePropertiesTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.v = AbstractVector((1.0, 2.0, 3.0))
        self.offsets = (
            AbstractVector((0.0, 0.0, 0.0)),
            AbstractVector((1.1, 2.2, 3.3)),
        )
        self.lines = tuple(Line(self.v, offset=o) for o in self.offsets)
        c1d_0 = Coordinates1D(0.0)
        c1d_1 = Coordinates1D(-1.0)
        c1d_3 = Coordinates1D(2.0)
        c3d_0 = Coordinates3D((0.0, 0.0, 0.0))
        c3d_1 = Coordinates3D((-1.0, -2.0, -3.0))
        c3d_3 = Coordinates3D((2.0, 4.0, 6.0))
        self.coords_1d = (c1d_0, c1d_1, c1d_3)
        self.coords_3d = (c3d_0, c3d_1, c3d_3)

    def test_line_embed(self) -> None:
        """Tests line's embedding."""
        for line, offset in zip(self.lines, self.offsets):
            for c1d, c3d in zip(self.coords_1d, self.coords_3d):
                self.assertCoordinates3DEquiv(
                    line.embed(c1d),
                    vector_as_coordinates(coordinates_as_vector(c3d) + offset),
                )

    def test_linee_tangential_space(self) -> None:
        """Tests line's tangential space."""
        for line in self.lines:
            for c1d in self.coords_1d:
                b = line.tangential_space(c1d)
                self.assertVectorEquiv(b, self.v)


class PlaneConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

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
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
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

    def test_plane_embed_domain(self) -> None:
        """Tests plane's embedding."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_plane.embed(coords)
        for coords in self.coords:
            self.infinite_plane.embed(coords)

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
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.n = cross(self.v1, self.v2)
        self.offsets = (
            AbstractVector((0.0, 0.0, 0.0)),
            AbstractVector((1.1, 2.2, 3.3)),
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

    def test_plane_embed(self) -> None:
        """Tests plane coordinates."""
        for plane, offset in zip(self.planes, self.offsets):
            for c2d, c3d in zip(self.coords_2d, self.coords_3d):
                self.assertCoordinates3DEquiv(
                    plane.embed(c2d),
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


class ParallelepipedConstructorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.domain = Domain1D(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((1.0, 0.0, 0.0))
        self.v3 = AbstractVector((1.0, 0.0, 0.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_parallelepiped_constructor(self) -> None:
        """Tests parallelepiped constroctor."""
        Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v3)
        Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v3, offset=self.offset)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v0, b1=self.v2, b2=self.v3)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v1, b1=self.v0, b2=self.v3)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v0)


class ParallelepipedDomainTest(ManifoldUnittest):
    def setUp(self) -> None:
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        v3 = AbstractVector((0.0, 0.0, 1.0))
        self.finite_paraep = Parallelepiped(
            v1,
            v2,
            v3,
            x0_domain=Domain1D(-1.0, 2.0),
            x1_domain=Domain1D(3.0, -4.0),
            x2_domain=Domain1D(-5.0, 6.0),
        )
        self.infinite_paraep = Parallelepiped(v1, v2, v3)
        self.coords = (
            Coordinates3D((-2.0, -2.0, 3.0)),
            Coordinates3D((3.0, -2.0, -3.0)),
            Coordinates3D((1.0, -5.0, 3.0)),
            Coordinates3D((1.0, 4.0, -3.0)),
            Coordinates3D((1.0, 2.0, 7.0)),
        )

    def test_parallelepiped_embed_domain(self) -> None:
        """Tests parallelepiped's embedding."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_paraep.embed(coords)
        for coords in self.coords:
            self.infinite_paraep.embed(coords)

    def test_parallelepiped_tangential_space_domain(self) -> None:
        """Tests parallelepiped's tangential space."""
        for coords in self.coords:
            with self.assertRaises(OutOfDomainError):
                self.finite_paraep.tangential_space(coords)
        for coords in self.coords:
            self.infinite_paraep.tangential_space(coords)


class ParallelepipedPropertiesTest(ManifoldUnittest):
    def setUp(self) -> None:
        self.v1 = AbstractVector((2.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 3.0, 0.0))
        self.v3 = AbstractVector((0.0, 0.0, 5.0))
        self.offsets = (
            AbstractVector((0.0, 0.0, 0.0)),
            AbstractVector((1.1, 2.2, 3.3)),
        )
        self.paraeps = tuple(
            Parallelepiped(self.v1, self.v2, self.v3, offset=o)
            for o in self.offsets
        )
        c_pre_0 = Coordinates3D((0.0, 0.0, 0.0))
        c_pre_1 = Coordinates3D((7.0, 11.0, 13.0))
        c_post_0 = Coordinates3D((0.0, 0.0, 0.0))
        c_post_1 = Coordinates3D((14.0, 33.0, 65.0))
        self.coords_pre = (c_pre_0, c_pre_1)
        self.coords_post = (c_post_0, c_post_1)

    def test_parallelepiped_embed(self) -> None:
        """Tests parallelepiped coordinates."""
        for paraep, offset in zip(self.paraeps, self.offsets):
            for c_pre, c_post in zip(self.coords_pre, self.coords_post):
                self.assertCoordinates3DEquiv(
                    paraep.embed(c_pre),
                    vector_as_coordinates(
                        coordinates_as_vector(c_post) + offset
                    ),
                )

    def test_parallelepiped_tangential_space(self) -> None:
        """Tests parallelepiped's tangential space."""
        for paraep in self.paraeps:
            for c_pre in self.coords_pre:
                b0, b1, b2 = paraep.tangential_space(c_pre)
                self.assertVectorEquiv(b0, self.v1)
                self.assertVectorEquiv(b1, self.v2)
                self.assertVectorEquiv(b2, self.v3)


if __name__ == "__main__":
    unittest.main()
