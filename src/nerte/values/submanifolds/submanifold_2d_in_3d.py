"""Module for two-dimensional submanifolds in three-dimensional manifolds."""

from abc import ABC, abstractmethod

from nerte.values.domains import OutOfDomainError, Domain2D
from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    UNIT_VECTOR0,
    UNIT_VECTOR1,
    UNIT_VECTOR2,
)


class Submanifold2DIn3D(ABC):
    """
    Representation of a two-dimensional submanifold in a three-dimensional
    manifold as a chart.
    """

    def __init__(self, domain: Domain2D) -> None:
        self.domain = domain

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        """
        Returns the embedded three-dimensional coordinates of the manifold.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot embed coordinates {coords}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_embed(coords)

    @abstractmethod
    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        """
        Hook for embed method.
        Returns the embedded three-dimensional coordinates of the manifold.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
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
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create basis of tangential space at coordinates {coords}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_tangential_space(coords)

    @abstractmethod
    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        """
        Hook for tangential_space method.
        Returns a basis of the tangetial space.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
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
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create surface normal at coordinates {coords}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_surface_normal(coords)

    @abstractmethod
    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        """
        Hook for surface_normal method.
        Returns a basis of the tangetial space.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass


class CanonicalImmersion2DIn3D(Submanifold2DIn3D):
    """
    Chart embedding a subset of R^2 into R^3 via the canonical immersion
        f(x_0, x_1) = (x_0, x_1, 0)

    Note: Can be used for mocking.
    """

    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        # pylint: disable=C0103
        x0, x1 = coords
        return Coordinates3D((x0, x1, 0))

    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        return (UNIT_VECTOR0, UNIT_VECTOR1)

    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        return UNIT_VECTOR2
