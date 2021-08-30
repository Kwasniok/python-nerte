# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144

import unittest
import itertools

from nerte.util.random_color_generator import RandomColorGenerator


class RandomColorGeneratorTest(unittest.TestCase):
    def test_colors_generated_consistently(self) -> None:
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
