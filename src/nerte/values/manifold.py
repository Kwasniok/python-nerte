"""Module for representations of (embedded) manifolds."""

from typing import Optional

from abc import ABC, abstractmethod

import math

from nerte.values.coordinates import Coordinates1D, Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    is_zero_vector,
    cross,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates


class OutOfDomainError(ValueError):
    # pylint: disable=W0107
    """Raised when a manifold parameter is outside of the domain."""

    pass


# TODO: Implement when type families are supported in python.

# class ManifoldTypeFamily[N:int]():
#     # pylint: disable=W0107
#     """
#     Pseudo-type family for generalized N-dimensional maifolds embedded in three
#     dimensions.
#
#     NOTE: N must be at least 1.
#
#     INFO: Type families are not supported in python 3.9.
#     INFO: Using a base class instead would violate the Liskov substiutional
#           principle and is NOT  advised!
#           (The subclasses would specialize the function signatures.)
#     """
#
#     def __init__(self, domain: tuple[Domain1D, ..., N-times]) -> None:
#         self.domain = domain
#
#     def is_in_domain(self, coords: tuple[float, ..., N-times]) -> bool:
#         """
#         Returns True, iff the domain coordinates are valid.
#         """
#         for x_i, domain_i in zip(coords, self.domain):
#             if x_i not in domain_i:
#                 return False
#         return True
#
#     def in_domain_assertion(self, coords: tuple[float, ..., N-times]) -> None:
#         """
#         Checks if domain coordinates are valid and trhows an OutOfDomainError
#         if not.
#
#         Note: To be used in all methods which raise OutOfDomainError.
#
#         :raises: OutOfDomainError
#         """
#
#         if not self.is_in_domain(coords):
#             raise OutOfDomainError(
#                 f"Coordinates3D {coords} are out of bounds of the manifold."
#                 + f" The domain is {self.domain}."
#             )
#
#     @abstractmethod
#     def embed(
#         self, coords: tuple[float, ..., N-times]
#     ) -> tuple[float, float, float]:
#         """
#         Returns the domain coordinates mapped into the three-dimensional space.
#
#         :raises: OutOfDomainError if the given coordinates are outside of the
#                  domain.
#         """
#         pass
#
#     @abstractmethod
#     def tangential_space(
#         self, coords: tuple[float, ...]
#     ) -> tuple[AbstractVector, ..., , N-times]:
#         """
#         Returns a basis of the local tangential vector space of the
#         three-dimensional embedding.
#
#         Note: The basis might not be normalized.
#
#         :raises: OutOfDomainError if the given coordinates are outside of the
#                  domain.
#         """
#         pass


class Manifold1D(ABC):
    """Representation of a one-dimensional manifold in three dimensions."""

    def __init__(self, domain: Domain1D) -> None:
        self.domain = domain

    def is_in_domain(self, coords: Coordinates1D) -> bool:
        """
        Returns True, iff the domain coordinates are valid.
        """
        return coords in self.domain

    def in_domain_assertion(self, coords: Coordinates1D) -> None:
        """
        Checks if domain coordinates are valid and trhows an OutOfDomainError
        if not.

        Note: To be used in all methods which raise OutOfDomainError.

        :raises: OutOfDomainError
        """

        if not self.is_in_domain(coords):
            raise OutOfDomainError(
                f"Coordinates3D {coords} are out of bounds of the manifold."
                + f" The domain is {self.domain}."
            )

    @abstractmethod
    def embed(self, coords: Coordinates1D) -> Coordinates3D:
        """
        Returns the embedded three-dimensional coordinates of the manifold.

        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass

    @abstractmethod
    def tangential_space(self, coords: Coordinates1D) -> AbstractVector:
        """
        Returns a basis of the local tangential vector space of the
        three-dimensional embedding.

        Note: The basis might not be normalized.
        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass


class Manifold2D(ABC):
    """Representation of a two-dimensional manifold in three dimensions."""

    def __init__(self, domain: tuple[Domain1D, Domain1D]) -> None:
        self.domain = domain

    def is_in_domain(self, coords: tuple[float, float]) -> bool:
        """
        Returns True, iff the domain coordinates are valid.
        """
        return coords[0] in self.domain[0] and coords[1] in self.domain[1]

    def in_domain_assertion(self, coords: Coordinates2D) -> None:
        """
        Checks if domain coordinates are valid and trhows an OutOfDomainError
        if not.

        Note: To be used in all methods which raise OutOfDomainError.

        :raises: OutOfDomainError
        """

        if not self.is_in_domain(coords):
            raise OutOfDomainError(
                f"Coordinates3D {coords} are out of bounds of the manifold."
                + f" The domain is {self.domain}."
            )

    @abstractmethod
    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        """
        Returns the embedded three-dimensional coordinates of the manifold.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass

    @abstractmethod
    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        """
        Returns the local surcafe normal of the three-dimensional embedding.

        Note: The normal might not be normalized to length one.

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

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass


class Manifold3D(ABC):
    """Representation of a three-dimensional manifold in three dimensions."""

    def __init__(self, domain: tuple[Domain1D, Domain1D, Domain1D]) -> None:
        self.domain = domain

    def is_in_domain(self, coords: Coordinates3D) -> bool:
        """
        Returns True, iff the domain coordinates are valid.
        """
        return (
            coords[0] in self.domain[0]
            and coords[1] in self.domain[1]
            and coords[2] in self.domain[2]
        )

    def in_domain_assertion(self, coords: Coordinates3D) -> None:
        """
        Checks if domain coordinates are valid and trhows an OutOfDomainError
        if not.

        Note: To be used in all methods which raise OutOfDomainError.

        :raises: OutOfDomainError
        """

        if not self.is_in_domain(coords):
            raise OutOfDomainError(
                f"Coordinates3D {coords} are out of bounds of the manifold."
                + f" The domain is {self.domain}."
            )

    @abstractmethod
    def embed(self, coords: Coordinates3D) -> Coordinates3D:
        """
        Returns the embedded three-dimensional coordinates of the manifold.

        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass

    @abstractmethod
    def tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        """
        Returns a basis of the local tangential vector space of the
        three-dimensional embedding.

        Note: The basis might not be normalized.
        Note: Deriving classes must callback this method.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        pass


class Line(Manifold1D):
    """Representation of a one-dimensional line embedded in three dimensions."""

    def __init__(
        self,
        direction: AbstractVector,
        domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
        # pylint: disable=R0913
        if is_zero_vector(direction):
            raise ValueError("Directional vector cannot be zero vector..")

        if domain is None:
            domain = Domain1D(-math.inf, math.inf)
        Manifold1D.__init__(self, domain)

        self._direction = direction

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def embed(self, coords: Coordinates1D) -> Coordinates3D:
        self.in_domain_assertion(coords)
        return vector_as_coordinates(self._direction * coords + self._offset)

    def tangential_space(self, coords: Coordinates1D) -> AbstractVector:
        self.in_domain_assertion(coords)
        return self._direction


class Plane(Manifold2D):
    """Representation of a two-dimensional plane embedded in three dimensions."""

    def __init__(  # pylint: disable=R0913
        self,
        b0: AbstractVector,
        b1: AbstractVector,
        x0_domain: Optional[Domain1D] = None,
        x1_domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
        if are_linear_dependent((b0, b1)):
            raise ValueError(
                f"Cannot construct plane. Basis vectors must be linear"
                f" independent (not b0={b0} and b1={b1})."
            )

        if x0_domain is None:
            x0_domain = Domain1D(-math.inf, math.inf)
        if x1_domain is None:
            x1_domain = Domain1D(-math.inf, math.inf)
        Manifold2D.__init__(self, (x0_domain, x1_domain))

        self._b0 = b0
        self._b1 = b1

        self._n = cross(b0, b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
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


class Parallelepiped(Manifold3D):
    """
    Representation of a three-dimensional paralellepiped embedded in three dimensions.
    """

    def __init__(  # pylint: disable=R0913
        self,
        b0: AbstractVector,
        b1: AbstractVector,
        b2: AbstractVector,
        x0_domain: Optional[Domain1D] = None,
        x1_domain: Optional[Domain1D] = None,
        x2_domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
        if are_linear_dependent((b0, b1, b2)):
            raise ValueError(
                f"Cannot construct parallelepiped. Basis vectors must be linear"
                f" independent (not b0={b0}, b1={b1}, b2={b2})."
            )

        if x0_domain is None:
            x0_domain = Domain1D(-math.inf, math.inf)
        if x1_domain is None:
            x1_domain = Domain1D(-math.inf, math.inf)
        if x2_domain is None:
            x2_domain = Domain1D(-math.inf, math.inf)
        Manifold3D.__init__(self, (x0_domain, x1_domain, x2_domain))

        self._b0 = b0
        self._b1 = b1
        self._b2 = b2

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def embed(self, coords: Coordinates3D) -> Coordinates3D:
        self.in_domain_assertion(coords)
        return vector_as_coordinates(
            self._b0 * coords[0]
            + self._b1 * coords[1]
            + self._b2 * coords[2]
            + self._offset
        )

    def tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        self.in_domain_assertion(coords)
        return (self._b0, self._b1, self._b2)
