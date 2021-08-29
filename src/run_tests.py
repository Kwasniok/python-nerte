"""Script to find and execute all unittests"""

import unittest


def main():
    """Executes all unittests."""
    verbosity = 1
    # find all tests and execute them
    loader = unittest.defaultTestLoader
    test_suite = loader.discover(start_dir=".", pattern="*_unittest.py")
    test_runner = unittest.runner.TextTestRunner(verbosity=verbosity)
    test_runner.run(test_suite)


if __name__ == "__main__":
    main()
