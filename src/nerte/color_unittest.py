# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
import itertools

from nerte.color import Color, Colors, RandomColorGenerator


class ColorTest(unittest.TestCase):
    def test_color(self):
        """Tests color rgb attribute."""
        c = Color(1, 2, 3)
        self.assertTrue(c.rgb == (1, 2, 3))


class ColorsTest(unittest.TestCase):
    def test_colors(self):
        """Tests color constants."""
        self.assertTrue(Colors.BLACK.rgb == (0, 0, 0))
        self.assertTrue(Colors.GRAY.rgb == (128, 128, 128))
        self.assertTrue(Colors.WHITE.rgb == (255, 255, 255))


class RandomColorGeneratorTest(unittest.TestCase):
    def test_colors_generated_consistently(self):
        """Tests (pseudo-)random color generator for consistency."""
        # RandomColorGenerator with identical seeds must behave identical

        # default seed
        for color1, color2 in itertools.islice(
            zip(
                RandomColorGenerator(),
                RandomColorGenerator(),
            ),
            10,
        ):
            self.assertTrue(color1.rgb == color2.rgb)

        # explicit seed
        seeds = (0, 1234)
        for seed in seeds:
            for color1, color2 in itertools.islice(
                zip(
                    RandomColorGenerator(seed=seed),
                    RandomColorGenerator(seed=seed),
                ),
                10,
            ):
                self.assertTrue(color1.rgb == color2.rgb)


if __name__ == "__main__":
    unittest.main()
