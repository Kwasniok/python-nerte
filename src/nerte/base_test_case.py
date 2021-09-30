"""Module for unit test frame work."""

import unittest


from typing import TypeVar, Callable, Optional

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


def float_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[float, float], bool]:
    """
    Returns a function which true iff both floats are considered almost equal.
    """

    # pylint: disable=C0103,W0621
    def float_almost_equal(x: float, y: float) -> bool:
        if delta is None:
            if places is None:
                return round(x - y, 7) == 0
            return round(x - y, places) == 0
        if places is not None:
            raise ValueError(
                f"Cannot determine scalar almost equal if both"
                f" places={places} and delta={delta} is given."
                f" Select only one of them!"
            )
        return abs(x - y) <= delta

    return float_almost_equal


def float_triple_almost_equal(
    places: Optional[int] = None, delta: Optional[float] = None
) -> Callable[[tuple[float, float, float], tuple[float, float, float]], bool]:
    """
    Returns a function which true iff both triples of floats are considered
    almost equal.
    """

    # pylint: disable=C0103,W0621
    def float_triple_almost_equal(
        x: tuple[float, float, float], y: tuple[float, float, float]
    ) -> bool:
        pred = float_almost_equal(places=places, delta=delta)
        return pred(x[0], y[0]) and pred(x[1], y[1]) and pred(x[2], y[2])

    return float_triple_almost_equal
