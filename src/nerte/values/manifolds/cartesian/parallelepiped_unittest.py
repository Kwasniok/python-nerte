# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest


from nerte.base_test_case import BaseTestCase

from nerte.values.coordinates import Coordinates3D
from nerte.values.coordinates_unittest import coordinates_3d_equiv
from nerte.values.interval import Interval
from nerte.values.linalg import AbstractVector
from nerte.values.linalg_unittest import vec_equiv
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)
from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.manifolds.cartesian.parallelepiped import Parallelepiped


class ParallelepipedConstructorTest(BaseTestCase):
    def setUp(self) -> None:
        self.domain = Interval(-1.0, 4.0)
        self.v0 = AbstractVector((0.0, 0.0, 0.0))
        self.v1 = AbstractVector((1.0, 0.0, 0.0))
        self.v2 = AbstractVector((0.0, 1.0, 0.0))
        self.v3 = AbstractVector((0.0, 0.0, 1.0))
        self.offset = AbstractVector((0.0, 0.0, 0.0))

    def test_parallelepiped_constructor(self) -> None:
        """Tests parallelepiped constroctor."""
        Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v3)
        Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v3, offset=self.offset)
        # no zero vector allowed
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v0, b1=self.v2, b2=self.v3)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v1, b1=self.v0, b2=self.v3)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v0)
        # no linear dependency allowed
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v1, b1=self.v1, b2=self.v2)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v1, b1=self.v2, b2=self.v2)
        with self.assertRaises(ValueError):
            Parallelepiped(b0=self.v3, b1=self.v2, b2=self.v3)


class ParallelepipedDomainTest(BaseTestCase):
    def setUp(self) -> None:
        v1 = AbstractVector((1.0, 0.0, 0.0))
        v2 = AbstractVector((0.0, 1.0, 0.0))
        v3 = AbstractVector((0.0, 0.0, 1.0))
        self.finite_paraep = Parallelepiped(
            v1,
            v2,
            v3,
            x0_domain=Interval(-1.0, 2.0),
            x1_domain=Interval(3.0, -4.0),
            x2_domain=Interval(-5.0, 6.0),
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


class ParallelepipedPropertiesTest(BaseTestCase):
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
                self.assertPredicate2(
                    coordinates_3d_equiv,
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
                self.assertPredicate2(vec_equiv, b0, self.v1)
                self.assertPredicate2(vec_equiv, b1, self.v2)
                self.assertPredicate2(vec_equiv, b2, self.v3)


if __name__ == "__main__":
    unittest.main()
