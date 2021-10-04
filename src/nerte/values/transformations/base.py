"""Base module for transformations between charts."""

from abc import ABC, abstractmethod

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import mat_vec_mult, AbstractMatrix, IDENTITY_MATRIX
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

    def jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        """
        Returns the Jacobian matrix of the transformation.

        :raises: OutOfDomainError iff coordinates are outside the domain
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create Jacobian for coordinates={coords}. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_jacobian(coords)

    @abstractmethod
    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        """
        Hook for jacobian method.
        Returns the Jacobian matrix of the transformation.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """

    def transform_tangent(self, tangent: TangentialVector) -> TangentialVector:
        """
        Returns the transformed tangential vector.

        :raises: OutOfDomainError iff coordinates are outside the domain
            NOTE: Membership of the codomain is NOT checked!
        """
        try:
            self.domain.assert_inside(tangent.point)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot transform tanential vector={tangent}."
                + self.domain.not_inside_reason(tangent.point)
            ) from ex

        point = self.transform_coords(tangent.point)
        jacobian = self.jacobian(tangent.point)
        vector = mat_vec_mult(jacobian, tangent.vector)
        return TangentialVector(point, vector)


class Identity(Transformation3D):
    """Identity transformation for three-dimensional domains."""

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return coords

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return IDENTITY_MATRIX
