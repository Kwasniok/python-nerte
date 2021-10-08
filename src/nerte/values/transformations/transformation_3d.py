"""Module for transformations between two three-dimensional charts."""

from abc import ABC, abstractmethod

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractMatrix,
    Rank3Tensor,
    mat_vec_mult,
    IDENTITY_MATRIX,
    ZERO_RANK3TENSOR,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.domains import OutOfDomainError, Domain3D


class Transformation3D(ABC):
    """
    Reviewepresentation of a transformations between open sets.

    A transformation 'translates' the representation of a manifold as one chart
    to that by another chart.
    """

    def __init__(self, domain: Domain3D, codomain: Domain3D) -> None:
        self.domain = domain
        self.codomain = codomain

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

    def inverse_transform_coords(self, coords: Coordinates3D) -> Coordinates3D:
        """
        Returns the inversely transformed coordinates.

        :raises: OutOfDomainError iff coordinates are outside the domain
            NOTE: Membership of the codomain is NOT checked!
        """
        try:
            self.codomain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot (inversely) transform coordinates={coords}. "
                + self.codomain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_inverse_transform_coords(coords)

    @abstractmethod
    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        """
        Hook for inverse_transform_coords method.
        Returns the inversely transformed coordinates.
        IMPORTANT: It is trusted that self.codomain.assert_inside(coords) is
        True.
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

    def inverse_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        """
        Returns the inverse Jacobian matrix of the transformation.

        :raises: OutOfDomainError iff coordinates are outside the domain
        """
        try:
            self.codomain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create (inverse) Jacobian for coordinates={coords}. "
                + self.codomain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_inverse_jacobian(coords)

    @abstractmethod
    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        """
        Hook for inverse_jacobian method.
        Returns the inverse Jacobian matrix of the transformation.
        IMPORTANT: It is trusted that self.codomain.assert_inside(coords) is
        True.
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
                f"Cannot transform tanential vector={tangent}. "
                + self.domain.not_inside_reason(tangent.point)
            ) from ex
        return self.internal_hook_transform_tangent(tangent)

    def internal_hook_transform_tangent(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        """
        Hook for transform_tangent method.
        Returns the transformed tangential vector.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is
        True.
        """
        point = self.internal_hook_transform_coords(tangent.point)
        jacobian = self.internal_hook_jacobian(tangent.point)
        vector = mat_vec_mult(jacobian, tangent.vector)
        return TangentialVector(point, vector)

    def inverse_transform_tangent(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        """
        Returns the inversely transformed tangential vector.

        :raises: OutOfDomainError iff coordinates are outside the domain
            NOTE: Membership of the domain is NOT checked!
        """
        try:
            self.codomain.assert_inside(tangent.point)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot (inversely) transform tanential vector={tangent}. "
                + self.codomain.not_inside_reason(tangent.point)
            ) from ex
        return self.internal_hook_inverse_transform_tangent(tangent)

    def internal_hook_inverse_transform_tangent(
        self, tangent: TangentialVector
    ) -> TangentialVector:
        """
        Hook for inverse_transform_tangent method.
        Returns the inversely transformed tangential vector.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is
        True.
        """
        point = self.internal_hook_inverse_transform_coords(tangent.point)
        jacobian = self.internal_hook_inverse_jacobian(tangent.point)
        vector = mat_vec_mult(jacobian, tangent.vector)
        return TangentialVector(point, vector)

    def hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        """
        Returns the Hesse tansor conatining all second derivatives.

        :raises: OutOfDomainError iff coordinates are outside the domain
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create second level Jacobian for"
                f" coordinates={coords}. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_hesse_tensor(coords)

    @abstractmethod
    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        """
        Hook for hesse_tensor method.
        Returns the Hesse tansor conatining all second derivatives.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """

    def inverse_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        """
        Returns the Hesse tansor of the inverse transformation
        conatining all second derivatives.

        :raises: OutOfDomainError iff coordinates are outside the domain
        """
        try:
            self.codomain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create (inverse) second level Jacobian for"
                f" coordinates={coords}. "
                + self.codomain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_inverse_hesse_tensor(coords)

    @abstractmethod
    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        """
        Hook for inverse_hesse_tensor method.
        Returns the Hesse tansor of the inverse transformation
        conatining all second derivatives.
        IMPORTANT: It is trusted that self.codomain.assert_inside(coords) is
        True.
        """


class Identity(Transformation3D):
    """Identity transformation for three-dimensional domains."""

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return coords

    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return coords

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return IDENTITY_MATRIX

    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        return IDENTITY_MATRIX

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        return ZERO_RANK3TENSOR

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        return ZERO_RANK3TENSOR
