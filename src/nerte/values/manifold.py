"""Module for representations of (embedded) manifolds."""

from typing import Optional

from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates2D, Coordinates
from nerte.values.linalg import AbstractVector, is_zero_vector, cross
from nerte.values.util.convert import vector_as_coordinates


class OutOfDomainError(ValueError):
    # pylint: disable=W0107
    """Raised when a manifold parameter is outside of the domain."""

    pass


def _range_test(min_max: tuple[float, float]) -> None:
    if min_max[0] == min_max[1]:
        raise ValueError(
            f"Cannot define domain of manifold with range {min_max}. Range has zero length."
        )


class Manifold2D(ABC):
    # pylint: disable=W0107
    """Representation of a two-dimensional manifold in three dimensions."""

    def __init__(
        self, x0_range: tuple[float, float], x1_range: tuple[float, float]
    ) -> None:
        _range_test(x0_range)
        _range_test(x1_range)
        self._x0_range = x0_range
        self._x1_range = x1_range

    def is_in_domain(self, coords: Coordinates2D) -> bool:
        """
        Returns True, iff two-dimensional coordinates are in the map's domain.
        """
        return (min(*self._x0_range) < coords[0] < max(*self._x0_range)) and (
            min(*self._x1_range) < coords[1] < max(*self._x1_range)
        )

    def in_domain_assertion(self, coords: Coordinates2D) -> None:
        """
        Checks if two-dimensional coordinate is in domain of the map.

        Note: To be used in all methods which raise OutOfDomainError.
        """
        if not self.is_in_domain(coords):
            raise OutOfDomainError(
                f"Coordinates {coords} are out of bounds of the manifold."
                + f" Ranges are {self._x0_range} and {self._x1_range}."
            )

    @abstractmethod
    def coordinates(self, coords: Coordinates2D) -> Coordinates:
        """
        Returns the embedded three-dimensional coordinates of the manifold.

        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass

    @abstractmethod
    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        """
        Returns the local surcafe normal of the three-dimensional embedding.

        Note: The normal might not be normalized to length one.
        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass

    @abstractmethod
    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        """
        Returns a basis of the local tangential vector space of the
        three-dimensional embedding.

        Note: The basis might not be normalized.
        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass


class Plane(Manifold2D):
    """Representation of a two-dimensional plane in three dimensions."""

    def __init__(
        self,
        b0: AbstractVector,
        b1: AbstractVector,
        x0_range: Optional[tuple[float, float]] = None,
        x1_range: Optional[tuple[float, float]] = None,
        offset: Optional[AbstractVector] = None,
    ):
        # pylint: disable=R0913
        if is_zero_vector(b0) or is_zero_vector(b1):
            raise ValueError("Basis vector cannot be zero vector..")

        if x0_range is None:
            x0_range = (-math.inf, math.inf)
        if x1_range is None:
            x1_range = (-math.inf, math.inf)

        Manifold2D.__init__(self, x0_range, x1_range)

        self._b0 = b0
        self._b1 = b1
        self._n = cross(b0, b1)
        if offset is None:
            self._offset = AbstractVector(0.0, 0.0, 0.0)
        else:
            self._offset = offset

    def coordinates(self, coords: Coordinates2D) -> Coordinates:
        self.in_domain_assertion(coords)
        return vector_as_coordinates(
            self._b0 * coords[0] + self._b1 * coords[1] + self._offset
        )

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        self.in_domain_assertion(coords)
        return self._n

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        self.in_domain_assertion(coords)
        return (self._b0, self._b1)
