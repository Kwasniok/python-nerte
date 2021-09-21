# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest

from nerte.base_test_case import BaseTestCase

from nerte.values.color import Color, Colors


class ColorTest(BaseTestCase):
    def test_color(self) -> None:
        """Tests color rgb attribute."""
        c = Color(1, 2, 3)
        self.assertTrue(c.rgb == (1, 2, 3))


class ColorsTest(BaseTestCase):
    def test_colors(self) -> None:
        """Tests color constants."""
        self.assertTrue(Colors.BLACK.rgb == (0, 0, 0))
        self.assertTrue(Colors.GRAY.rgb == (128, 128, 128))
        self.assertTrue(Colors.WHITE.rgb == (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
