"""Module for charts from 2D to 3D charts."""

from abc import ABC, abstractmethod

from nerte.values.manifolds.base import OutOfDomainError
from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.linalg import AbstractVector


class Chart2DTo3D(ABC):
    """Representation of a two-dimensional manifold in three dimensions."""

    @abstractmethod
    def is_in_domain(self, coords: Coordinates2D) -> bool:
        """
        Returns True, iff the domain coordinates are valid.
        """
        # pylint: disable=W0107
        pass

    @abstractmethod
    def out_of_domain_reason(self, coords: Coordinates2D) -> str:
        """
        Returns a string stating why the coordinates are not inside the domain.

        Note: This method is a hook which is called by in_domain_assertion to
            give more expressive errors. Whether the coordiantes are actually
            inside the domain or not may not be checked and the string may
            or may not be generic or specific for the coordinates.
        """
        # pylint: disable=W0107
        pass

    def in_domain_assertion(self, coords: Coordinates2D) -> None:
        """
        Checks if domain coordinates are valid and trhows an OutOfDomainError
        if not.

        Note: To be used in all methods which test whether coordinates are
        inside the domain.

        :raises: OutOfDomainError
        """
        if not self.is_in_domain(coords):
            raise OutOfDomainError(self.out_of_domain_reason(coords))

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        """
        Returns the embedded three-dimensional coordinates of the manifold.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.in_domain_assertion(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot embed coordinates {coords}."
                f" Coordinates out of domain. "
                + self.out_of_domain_reason(coords)
            ) from ex
        return self.internal_hook_embed(coords)

    @abstractmethod
    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        """
        Hook for embed method.
        Returns the embedded three-dimensional coordinates of the manifold.
        IMPORTANT: It is trusted that self.in_domain_assertion(coords) is True.
        """
        # pylint: disable=W0107
        pass

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
        try:
            self.in_domain_assertion(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create basis of tangential space at coordinates {coords}."
                f" Coordinates out of domain. "
                + self.out_of_domain_reason(coords)
            ) from ex
        return self.internal_hook_tangential_space(coords)

    @abstractmethod
    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        """
        Hook for tangential_space method.
        Returns a basis of the tangetial space.
        IMPORTANT: It is trusted that self.in_domain_assertion(coords) is True.
        """
        # pylint: disable=W0107
        pass

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        """
        Returns the local surcafe normal of the three-dimensional embedding.

        Note: The normal might not be normalized to length one.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.in_domain_assertion(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create surface normal at coordinates {coords}."
                f" Coordinates out of domain. "
                + self.out_of_domain_reason(coords)
            ) from ex
        return self.internal_hook_surface_normal(coords)

    @abstractmethod
    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        """
        Hook for surface_normal method.
        Returns a basis of the tangetial space.
        IMPORTANT: It is trusted that self.in_domain_assertion(coords) is True.
        """
        # pylint: disable=W0107
        pass
