"""module for one-dimensional domains."""


from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates1D
from nerte.values.interval import Interval
from nerte.values.domains.base import OutOfDomainError


class Domain1D(ABC):
    """
    Description of a one-dimensional domain: An (open) set of coordinates,
    suitable as a domain of a chart.

    The object behaves like a characteristic function of the set.
    """

    @abstractmethod
    def are_inside(self, coords: Coordinates1D) -> bool:
        """
        Returns True, iff the coordinates are a valid representation of a point.
        """

    @abstractmethod
    def not_inside_reason(self, coords: Coordinates1D) -> str:
        """
        Returns a string describing the domain for for expressive error messages.
        """

    def assert_inside(self, coords: Coordinates1D) -> None:
        """
        Raises OutOfDomainError, iff coordinates are outside of the domain with
        a descriptive message.
        """
        if not self.are_inside(coords):
            raise OutOfDomainError(self.not_inside_reason(coords))


class R1Domain(Domain1D):
    """
    Domain describing the set R^1.

    Note: Can be used for mocking.
    """

    def are_inside(self, coords: Coordinates1D) -> bool:
        # pylint: disable=C0103
        (x0,) = coords
        return math.isfinite(x0)

    def not_inside_reason(self, coords: Coordinates1D) -> str:
        return "invalid coords reason"


R1 = R1Domain()


class Empty(Domain1D):
    """
    Domain describing the empty set.

    Note: Can be used for mocking.
    """

    def are_inside(self, coords: Coordinates1D) -> bool:
        return False

    def not_inside_reason(self, coords: Coordinates1D) -> str:
        return f"Coordinates {coords} are not inside the empty set."


EMPTY1D = Empty()

# TODO: test
class CartesianProduct1D(Domain1D):
    """
    Domain describing the cartesian product of one interval.

    Note: Can be used for mocking.
    """

    def __init__(self, interval: Interval) -> None:
        self.interval = interval

    def are_inside(self, coords: Coordinates1D) -> bool:
        return coords[0] in self.interval

    def not_inside_reason(self, coords: Coordinates1D) -> str:
        return (
            f"Coordinate {coords[0]} is not inside the"
            f" interval {self.interval}."
        )
