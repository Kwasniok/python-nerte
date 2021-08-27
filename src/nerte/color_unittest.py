# pylint: disable=R0801

import unittest
import itertools

from nerte.color import Color, GRAY, RandomColorDispenser


class ColorTest(unittest.TestCase):
    def test(self):
        c = Color(1, 2, 3)
        self.assertTrue(c.rgb == (1, 2, 3))

    def test_colors(self):
        self.assertTrue(GRAY.rgb == (128, 128, 128))

    def test_colors_dispensed_consistently(self):
        # RandomColorDispenser with identical seeds must behave identical

        # default seed
        for color1, color2 in itertools.islice(
            zip(
                RandomColorDispenser(),
                RandomColorDispenser(),
            ),
            10,
        ):
            self.assertTrue(color1.rgb == color2.rgb)

        # explicit seed
        seeds = (0, 1234)
        for seed in seeds:
            for color1, color2 in itertools.islice(
                zip(
                    RandomColorDispenser(seed=seed),
                    RandomColorDispenser(seed=seed),
                ),
                10,
            ):
                self.assertTrue(color1.rgb == color2.rgb)


if __name__ == "__main__":
    unittest.main()
