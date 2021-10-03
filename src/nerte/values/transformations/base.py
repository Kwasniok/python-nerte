"""Base module for transformations between charts."""

from abc import ABC, abstractmethod

from nerte.values.coordinates import Coordinates3D
from nerte.values.tangential_vector import TangentialVector
from nerte.values.domains import OutOfDomainError, Domain3D


class Transformation3D(ABC):
    """
    Reviewepresentation of a transformations between charts.

    A transformation 'translates' the representation of a manifold as one chart
    to that by another chart.
    """

    def __init__(self, domain: Domain3D) -> None:
        self.domain = domain

    def transform_coords(self, coords: Coordinates3D) -> Coordinates3D:
        """
        Returns the transformed coordinates.

        :raises: OutOfDomainError iff coordinates are outside the domain
            NOTE: Membership of the codomain is NOT checked!
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot transform coordinates={coords}. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_transform_coords(coords)

    @abstractmethod
    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        """
        Hook for transform_coords method.
        Returns the transformed coordinates.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """

    def transform_vector(self, vector: TangentialVector) -> TangentialVector:
        """
        Returns the transformed (tangential) vector.

        :raises: OutOfDomainError iff coordinates are outside the domain
            NOTE: Membership of the codomain is NOT checked!
        """
        try:
            self.domain.assert_inside(vector.point)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot transform vector={vector}."
                + self.domain.not_inside_reason(vector.point)
            ) from ex
        return self.internal_hook_transform_vector(vector)

    @abstractmethod
    def internal_hook_transform_vector(
        self, vector: TangentialVector
    ) -> TangentialVector:
        """
        Hook for transform_vector method.
        Returns the transformed (tangential) vector.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """


class Identity(Transformation3D):
    """Identity transformation for three-dimensional domains."""

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return coords

    def internal_hook_transform_vector(
        self, vector: TangentialVector
    ) -> TangentialVector:
        return vector
