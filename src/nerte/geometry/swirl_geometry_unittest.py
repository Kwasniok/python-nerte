# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
from itertools import permutations

from nerte.values.coordinates import Coordinates
from nerte.values.linalg import AbstractVector
from nerte.values.ray import Ray
from nerte.values.face import Face
from nerte.geometry.swirl_geometry import SwirlGeometry


class SwirlGeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        # face with all permuations of its coordinates
        # NOTE: Results are invariant under coordinate permutation!
        p1 = Coordinates(1.0, 0.0, 0.0)
        p2 = Coordinates(0.0, 1.0, 0.0)
        p3 = Coordinates(0.0, 0.0, 1.0)
        self.faces = list(Face(*ps) for ps in permutations((p1, p2, p3)))
        # geometry (no bend == euclidean)
        self.geo = SwirlGeometry(
            max_steps=10, max_ray_length=10.0, bend_factor=0.0
        )
        # rays pointing 'forwards' towards faces and parallel to face normal
        s0 = Coordinates(0.0, 0.0, 0.0)
        s1 = Coordinates(0.3, 0.0, 0.0)  # one third of p1
        s2 = Coordinates(0.0, 0.3, 0.0)  # one third of p2
        s3 = Coordinates(0.0, 0.0, 0.3)  # one third of p3
        ss = (s0, s1, s2, s3)
        v = AbstractVector(1.0, 1.0, 1.0)
        self.intersecting_rays = list(Ray(start=s, direction=v) for s in ss)

    def test_dummy_geometry_intersects(self) -> None:
        """
        Tests if rays intersect as expected.
        Each ray points 'forwards' towards the face and is parallel to face's
        normal.
        """
        for r in self.intersecting_rays:
            for f in self.faces:
                self.assertTrue(self.geo.intersects(r, f))


if __name__ == "__main__":
    unittest.main()
