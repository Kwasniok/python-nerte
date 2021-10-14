"""module for two-dimensional domains."""


from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates2D
from nerte.values.interval import Interval
from nerte.values.domains.base import OutOfDomainError


class Domain2D(ABC):
    """
    Description of a two-dimensional domain: An (open) set of coordinates,
    suitable as a domain of a chart.

    The object behaves like a characteristic function of the set.
    """

    @abstractmethod
    def are_inside(self, coords: Coordinates2D) -> bool:
        """
        Returns True, iff the coordinates are a valid representation of a point.
        """

    @abstractmethod
    def not_inside_reason(self, coords: Coordinates2D) -> str:
        """
        Returns a string describing the domain for for expressive error messages.
        """

    def assert_inside(self, coords: Coordinates2D) -> None:
        """
        Raises OutOfDomainError, iff coordinates are outside of the domain with
        a descriptive message.
        """
        if not self.are_inside(coords):
            raise OutOfDomainError(self.not_inside_reason(coords))


class R2Domain(Domain2D):
    """
    Domain describing the set R^2.

    Note: Can be used for mocking.
    """

    def are_inside(self, coords: Coordinates2D) -> bool:
        # pylint: disable=C0103
        x0, x1 = coords
        return math.isfinite(x0) and math.isfinite(x1)

    def not_inside_reason(self, coords: Coordinates2D) -> str:
        return "invalid coords reason"


R2 = R2Domain()


class Empty(Domain2D):
    """
    Domain describing the empty set.

    Note: Can be used for mocking.
    """

    def are_inside(self, coords: Coordinates2D) -> bool:
        return False

    def not_inside_reason(self, coords: Coordinates2D) -> str:
        return f"Coordinates {coords} are not inside the empty set."


EMPTY2D = Empty()


class CartesianProduct2D(Domain2D):
    """
    Domain describing the cartesian product of two intervals.

    Note: Can be used for mocking.
    """

    def __init__(self, interval0: Interval, interval1: Interval) -> None:
        self.intervals = (interval0, interval1)

    def __repr__(self) -> str:
        return f"{self.intervals[0]}x{self.intervals[1]}"

    def are_inside(self, coords: Coordinates2D) -> bool:
        return coords[0] in self.intervals[0] and coords[1] in self.intervals[1]

    def not_inside_reason(self, coords: Coordinates2D) -> str:
        return (
            f"Coordinates {coords} are not inside the"
            f" cartesian product {self.intervals[0]}x{self.intervals[1]}."
        )
