"""module for three-dimensional domains."""


from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.interval import Interval
from nerte.values.domains.base import OutOfDomainError


class Domain3D(ABC):
    """
    Description of a three-dimensional domain: An (open) set of coordinates,
    suitable as a domain of a chart.

    The object behaves like a characteristic function of the set.
    """

    @abstractmethod
    def are_inside(self, coords: Coordinates3D) -> bool:
        """
        Returns True, iff the coordinates are a valid representation of a point.
        """

    @abstractmethod
    def not_inside_reason(self, coords: Coordinates3D) -> str:
        """
        Returns a string describing the domain for for expressive error messages.
        """

    def assert_inside(self, coords: Coordinates3D) -> None:
        """
        Raises OutOfDomainError, iff coordinates are outside of the domain with
        a descriptive message.
        """
        if not self.are_inside(coords):
            raise OutOfDomainError(self.not_inside_reason(coords))


class R3Domain(Domain3D):
    """
    Domain describing the set R^3.

    Note: Can be used for mocking.
    """

    def are_inside(self, coords: Coordinates3D) -> bool:
        # pylint: disable=C0103
        x0, x1, x2 = coords
        return math.isfinite(x0) and math.isfinite(x1) and math.isfinite(x2)

    def not_inside_reason(self, coords: Coordinates3D) -> str:
        return "invalid coords reason"


R3 = R3Domain()


class Empty(Domain3D):
    """
    Domain describing the empty set.

    Note: Can be used for mocking.
    """

    def are_inside(self, coords: Coordinates3D) -> bool:
        return False

    def not_inside_reason(self, coords: Coordinates3D) -> str:
        return f"Coordinates {coords} are not inside the empty set."


EMPTY3D = Empty()


class CartesianProduct3D(Domain3D):
    """
    Domain describing the cartesian product of two intervals.

    Note: Can be used for mocking.
    """

    def __init__(
        self, interval0: Interval, interval1: Interval, interval2: Interval
    ) -> None:
        self.intervals = (interval0, interval1, interval2)

    def are_inside(self, coords: Coordinates3D) -> bool:
        return (
            coords[0] in self.intervals[0]
            and coords[1] in self.intervals[1]
            and coords[2] in self.intervals[2]
        )

    def not_inside_reason(self, coords: Coordinates3D) -> str:
        return (
            f"Coordinates {coords} are not inside the cartesian product"
            f" {self.intervals[0]}x{self.intervals[1]}x{self.intervals[2]}."
        )
