"""Module for representations of (embedded) manifolds."""

from abc import ABC, abstractmethod


from nerte.values.coordinates import Coordinates1D, Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import AbstractVector


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
