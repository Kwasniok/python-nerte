#! /usr/env/python

"""
This script performs the validation of the code base which must be performed before each commit.
"""


from typing import Optional

import os
import sys
import argparse
import subprocess


class ChangeDirecory:
    """
    Guarded change of working directory.

    To be used with the 'with'-syntax.
    Changes directory on enter.
    Restores old working directory on exit.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._old_path: Optional[str] = None

    def __enter__(self) -> None:
        self._old_path = os.getcwd()
        os.chdir(self._path)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        if self._old_path is None:
            raise RuntimeError("Could not recover old working directory.")
        os.chdir(self._old_path)


# src must exist
if not os.path.isdir("src"):
    print("Code base invalid. No directory 'src' found.")
    sys.exit(1)

# src must be readable
if not os.access("src", os.R_OK):
    print("Code base invalid. Cannot read directory 'src'.")
    sys.exit(1)


def run_mypy() -> None:
    """Runs mypy on src."""
    print("runing mypy ...")
    with ChangeDirecory("."):
        res = subprocess.run(
            "python -m mypy src",
            shell=True,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if res.returncode != 0:
            print("Code base invalid. Mypy failed.")
            sys.exit(1)


def run_pylint(disable: Optional[list[str]] = None) -> None:
    """Runs pylint on src."""
    print("running pylint ...")
    with ChangeDirecory("."):
        if disable is None:
            dis = ""
        else:
            dis = ",".join(disable)
        res = subprocess.run(
            f"pylint --disable={dis} src",
            shell=True,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if res.returncode != 0:
            print("Code base invalid. Pylint failed.")
            sys.exit(1)


def run_tests() -> None:
    """Runs (unit)tests."""
    print("running tests ...")
    with ChangeDirecory("src"):
        res = subprocess.run(
            "python run_tests.py",
            shell=True,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if res.returncode != 0:
            print("Code base invalid. Tests failed.")
            sys.exit(1)


def run_all() -> None:
    """Full run of checks."""
    run_mypy()
    run_pylint()
    run_tests()
    print("Code base is valid!")
    sys.exit(0)  # must be 0


def run_light() -> None:
    """Light run of checks. Allways fails. Good as pre-check."""
    print("WARNING: Running in light mode is not recommended!")
    run_mypy()
    run_tests()
    # do pylint last as it often fails in pre-checks
    run_pylint(disable=["R0801", "W0511"])
    print(
        "WARNING: Results were obtained in light mode."
        " Therefore the code base may not be valid!"
    )
    sys.exit(2)  # must not be 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate code base for committing."
    )
    parser.add_argument(
        "-l",
        "--light",
        action="store_true",
        help="run in light mode (e.g. ignores fixme)"
        "\nNot recommended for final"
        " validation but usefull for intermediate checks when testing ideas."
        " Any result will formally count as a failure.",
    )
    args = parser.parse_args()

    if args.light:
        run_light()
    else:
        run_all()
