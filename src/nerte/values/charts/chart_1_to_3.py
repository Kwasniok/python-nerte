"""Module for charts from 1D to 3D charts."""

from abc import ABC, abstractmethod


from nerte.values.domains import OutOfDomainError, Domain1D
from nerte.values.coordinates import Coordinates1D, Coordinates3D
from nerte.values.linalg import AbstractVector, UNIT_VECTOR0


class Chart1DTo3D(ABC):
    """Representation of a one-dimensional manifold in three dimensions."""

    def __init__(self, domain: Domain1D) -> None:
        self.domain = domain

    def embed(self, coords: Coordinates1D) -> Coordinates3D:
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
    def internal_hook_embed(self, coords: Coordinates1D) -> Coordinates3D:
        """
        Hook for embed method.
        Returns the embedded three-dimensional coordinates of the manifold.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass

    def tangential_space(self, coords: Coordinates1D) -> AbstractVector:
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
        self, coords: Coordinates1D
    ) -> AbstractVector:
        """
        Hook for tangential_space method.
        Returns a basis of the tangetial space.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass


class CanonicalImmersionChart1DTo3D(Chart1DTo3D):
    """
    Chart embedding a subset of R into R^3 via the canonical immersion
        f(x_0) = (x_0, 0, 0)

    Note: Can be used for mocking.
    """

    def internal_hook_embed(self, coords: Coordinates1D) -> Coordinates3D:
        # pylint: disable=C0103
        (x0,) = coords
        return Coordinates3D((x0, 0, 0))

    def internal_hook_tangential_space(
        self, coords: Coordinates1D
    ) -> AbstractVector:
        return UNIT_VECTOR0
