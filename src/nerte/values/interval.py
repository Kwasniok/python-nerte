"""
Module for representations of intervals (e.g. for charts of manifolds).
Intervals must be convertible to an open interval.
"""

import math


class Interval:

    """
    Representation of a one-dimensional range betwenn two real numbers.
    Note: Must be convertible to an open interval - i.e. of non-zero length.
    Note: Used to denote domains of charts of manifolds.
    """

    def __init__(self, start: float, stop: float) -> None:
        """:raises: ValueError for invalid domain parameters."""

        # NOTE: DON'T use: start == math.nan
        if math.isnan(start) or math.isnan(stop):
            raise ValueError(
                f"Cannot define one-dimensional interval with {start} or {stop}."
                + " Values cannot be NaN."
            )
        if start == stop:
            raise ValueError(
                f"Cannot define one-dimensional interval with {start} or {stop}."
                + " Values cannot be identical, since the domain must be open"
            )
        self._start = start
        self._stop = stop
        # optimization for faster __contains__
        self._min = min(start, stop)
        self._max = max(start, stop)

    def __repr__(self) -> str:
        return f"Interval({self._start}, {self._stop})"

    def __str__(self) -> str:
        return f"({self._start}, {self._stop})"

    def __contains__(self, val: float) -> bool:
        return self._min <= val <= self._max

    def as_tuple(self) -> tuple[float, float]:
        """Returns start and stop parameter as a tuple."""
        return self._start, self._stop

    def start(self) -> float:
        """Returns the  start parameter."""
        return self._start

    def stop(self) -> float:
        """Returns the stop parameter."""
        return self._stop

    def min(self) -> float:
        """Returns the minimum of start and stop parameter."""
        return self._min

    def max(self) -> float:
        """Returns the maximum of start and stop parameter."""
        return self._max
