# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

import itertools

from nerte.values.coordinates_unittest import CoordinatesTestCaseMixin
from nerte.values.linalg_unittest import LinAlgTestCaseMixin

from nerte.values.coordinates import Coordinates1D, Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector
from nerte.values.manifold import (
    Manifold1D,
    Manifold2D,
    Manifold3D,
    OutOfDomainError,
)


class ManifoldTestCaseMixin(CoordinatesTestCaseMixin, LinAlgTestCaseMixin):
    pass


class Manifold1DImplementationTest(unittest.TestCase, ManifoldTestCaseMixin):
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
                domain: tuple[Domain1D],
            ):
                Manifold1D.__init__(self, domain)

            def embed(self, coords: Coordinates1D) -> Coordinates3D:
                self.in_domain_assertion(coords)
                return Coordinates3D((coords[0], 0.0, 0.0))

            def tangential_space(self, coords: Coordinates1D) -> AbstractVector:
                self.in_domain_assertion(coords)
                return AbstractVector((1.0, 0.0, 0.0))

        man = DummyManifold1D((self.domain,))
        for x in self.coord_inside_domain:
            man.embed(Coordinates1D((x,)))
            man.tangential_space(Coordinates1D((x,)))
        for x in self.coord_outside_domain:
            with self.assertRaises(OutOfDomainError):
                man.embed(Coordinates1D((x,)))
            with self.assertRaises(OutOfDomainError):
                man.tangential_space(Coordinates1D((x,)))

        self.assertTrue(man.domain[0] is self.domain)


class Manifold2DImplementationTest(unittest.TestCase, ManifoldTestCaseMixin):
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


class Manifold3DImplementationTest(unittest.TestCase, ManifoldTestCaseMixin):
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
                return coords

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


if __name__ == "__main__":
    unittest.main()
