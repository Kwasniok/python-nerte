"""Script to find and execute all unittests."""

import sys
import os
import unittest

# handle warings as errors
# https://docs.python.org/3/library/warnings.html
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("error")
    # enable for subprocesses as well
    os.environ["PYTHONWARNINGS"] = "error"


def do_tests() -> unittest.TestResult:
    """
    Executes all unittests.
    Returns test results.
    """
    # find all tests and execute them
    loader = unittest.defaultTestLoader
    test_suite = loader.discover(start_dir=".", pattern="*_unittest.py")
    test_runner = unittest.runner.TextTestRunner(verbosity=1)
    return test_runner.run(test_suite)


def do_exit(results: unittest.TestResult) -> None:
    """
    Exit the script with an exit code equals to the total amount of unsuccessful
    tests.
    """
    exit_code = (
        len(results.errors) + len(results.failures) + len(results.skipped)
    )
    sys.exit(exit_code)


def main() -> None:
    """
    Executes all unittests and exits with an code equals to the total amount of
    unsuccessful tests.
    """
    results = do_tests()
    do_exit(results)


if __name__ == "__main__":
    main()
