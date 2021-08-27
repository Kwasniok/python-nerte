# pylint: disable=R0801

import unittest
from nerte.coordinates import Coordinates
from nerte.vector import Vector
from nerte.camera import Camera


class CameraTest(unittest.TestCase):
    def test(self):
        location = Coordinates(1.1, 2.2, 3.3)
        direction = Vector(0.0, 0.0, 1.0)
        width_vector = Vector(1.0, 0.0, 0.0)
        height_vector = Vector(0.0, 1.0, 0.0)
        dim = 100
        Camera(
            location=location,
            direction=direction,
            canvas_dimensions=(dim, dim),
            detector_manifold=(width_vector, height_vector),
        )


if __name__ == "__main__":
    unittest.main()
