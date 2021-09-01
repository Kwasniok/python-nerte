"""Module for representations of (embedded) manifolds."""

from typing import Optional

from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector, is_zero_vector, cross
from nerte.values.util.convert import vector_as_coordinates


class OutOfDomainError(ValueError):
    # pylint: disable=W0107
    """Raised when a manifold parameter is outside of the domain."""

    pass


class Manifold2D(ABC):
    # pylint: disable=W0107
    """Representation of a two-dimensional manifold in three dimensions."""

    def __init__(self, x0_domain: Domain1D, x1_domain: Domain1D) -> None:
        self._x0_domain = x0_domain
        self._x1_domain = x1_domain

    def x0_domain(self) -> Domain1D:
        """Return the domain of the x0 parameter."""
        return self._x0_domain

    def x1_domain(self) -> Domain1D:
        """Return the domain of the x1 parameter."""
        return self._x1_domain

    def is_in_domain(self, coords: Coordinates2D) -> bool:
        """
        Returns True, iff two-dimensional coordinates are in the map's domain.
        """
        return coords[0] in self._x0_domain and coords[1] in self._x1_domain

    def in_domain_assertion(self, coords: Coordinates2D) -> None:
        """
        Checks if two-dimensional coordinate is in domain of the map.

        Note: To be used in all methods which raise OutOfDomainError.
        """
        if not self.is_in_domain(coords):
            raise OutOfDomainError(
                f"Coordinates3D {coords} are out of bounds of the manifold."
                + f" Ranges are {self._x0_domain} and {self._x1_domain}."
            )

    # TODO: improve name
    @abstractmethod
    def coordinates(self, coords: Coordinates2D) -> Coordinates3D:
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
        x0_domain: Optional[Domain1D] = None,
        x1_domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
        # pylint: disable=R0913
        if is_zero_vector(b0) or is_zero_vector(b1):
            raise ValueError("Basis vector cannot be zero vector..")

        if x0_domain is None:
            x0_domain = Domain1D(-math.inf, math.inf)
        if x1_domain is None:
            x1_domain = Domain1D(-math.inf, math.inf)
        Manifold2D.__init__(self, x0_domain, x1_domain)

        self._b0 = b0
        self._b1 = b1
        self._n = cross(b0, b1)
        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def coordinates(self, coords: Coordinates2D) -> Coordinates3D:
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
