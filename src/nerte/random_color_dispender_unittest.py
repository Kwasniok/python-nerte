import unittest
import itertools

from nerte.random_color_dispender import RandomColorDispenser


class RandomColorDispenserTest(unittest.TestCase):
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
            self.assertTrue(color1 == color2)

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
                self.assertTrue(color1 == color2)


if __name__ == "__main__":
    unittest.main()
