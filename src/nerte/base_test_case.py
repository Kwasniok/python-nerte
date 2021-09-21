"""Module for unit test frame work."""

import unittest


from typing import TypeVar, Callable

T = TypeVar("T")  # pylint: disable=C0103
U = TypeVar("U")  # pylint: disable=C0103


class BaseTestCase(unittest.TestCase):
    # pylint: disable=C0103,R0201
    """Base test case for all unit tests to derive from."""

    def assertPredicate1(self, pred: Callable[[T], bool], x: T) -> None:
        """
        Asserts the given unary prediacte is True for x. The Predicate is a
        function of x returning a bool.
        """
        if not pred(x):
            raise AssertionError(
                f"Prediacte {pred.__name__} not fullfilled for {x}."
            )

    def assertPredicate2(
        self, pred: Callable[[T, U], bool], x: T, y: U
    ) -> None:
        """
        Asserts the given binary prediacte is True for x and y. The Predicate is
        a function of x and y returning a bool.
        """
        if not pred(x, y):
            raise AssertionError(
                f"Prediacte `{pred.__name__}` not fullfilled for {x} and {y}."
            )
