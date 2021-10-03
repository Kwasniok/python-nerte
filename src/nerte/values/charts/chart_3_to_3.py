"""Module for charts from 3D to 3D charts."""

from abc import ABC, abstractmethod


import math

from nerte.values.domains import OutOfDomainError, Domain3D
from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    STANDARD_BASIS,
    Metric,
    IDENTITY_METRIC,
    dot,
    is_zero_vector,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.util.convert import coordinates_as_vector


class Chart3DTo3D(ABC):
    """Representation of a three-dimensional manifold in three dimensions."""

    def __init__(self, domain: Domain3D) -> None:
        self.domain = domain

    def embed(self, coords: Coordinates3D) -> Coordinates3D:
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
    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        """
        Hook for embed method.
        Returns the embedded three-dimensional coordinates of the manifold.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass

    def tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
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
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        """
        Hook for tangential_space method.
        Returns a basis of the tangetial space.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass

    def metric(self, coords: Coordinates3D) -> Metric:

        """
        Returns the (local) metric for the given coordinates.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create metric at coordinates {coords}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_metric(coords)

    @abstractmethod
    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        """
        Hook for metric method.
        Returns the (local) metric for the given coordinates.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass

    def scalar_product(
        self, coords: Coordinates3D, vec1: AbstractVector, vec2: AbstractVector
    ) -> float:
        """
        Returns the result of the scalar product of both vectors at the given
        coordinates.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot calculate scalar product at coordiantes {coords}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return dot(vec1, vec2, metric=self.metric(coords))

    def length(self, tangent: TangentialVector) -> float:
        """
        Returns the length of the tangential vector.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.domain.assert_inside(tangent.point)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot calculate length of tangential vector {tangent}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(tangent.point)
            ) from ex
        return math.sqrt(
            self.scalar_product(tangent.point, tangent.vector, tangent.vector)
        )

    def normalized(self, tangent: TangentialVector) -> TangentialVector:
        """
        Returns the nrmalized tangential vector (length is normlaized to 1 and
        direction is kept).

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
                 ValueError if the given vector is the zero vector.
        """
        try:
            self.domain.assert_inside(tangent.point)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot normalize tangential vector {tangent}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(tangent.point)
            ) from ex
        if is_zero_vector(tangent.vector):
            raise ValueError(
                f"Cannot normalize tangential vector {tangent}."
                f" Vector must be non-zero."
            )
        length = self.length(tangent)
        return TangentialVector(tangent.point, tangent.vector / length)

    def geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        """
        Returns the local change in coordiantes and their velocities.

        Let x(𝜆) be a geodesic.
        For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2).

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.domain.assert_inside(tangent.point)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create basis of tangential space at coordinates"
                f" {tangent.point}. Coordinates out of domain. "
                + self.domain.not_inside_reason(tangent.point)
            ) from ex
        return self.internal_hook_geodesics_equation(tangent)

    @abstractmethod
    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        """
        Hook for geodesics_equation method.
        Returns the local change in coordiantes and their velocities.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass

    def initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        """
        Returns a tangent at coordinates start. A geodesic cast from this
        tangent connects start with target.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
                 ValueError if no geodesic connecting both coordinates exists.
        """
        try:
            self.domain.assert_inside(start)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create initial tangent of geodesic for start"
                f" coordiantes {start}. Coordinates out of domain. "
                + self.domain.not_inside_reason(start)
            ) from ex
        try:
            self.domain.assert_inside(target)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create initial tangent of geodesic for target"
                f" coordiantes {target}. Coordinates out of domain. "
                + self.domain.not_inside_reason(target)
            ) from ex
        return self.internal_hook_initial_geodesic_tangent_from_coords(
            start, target
        )

    @abstractmethod
    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        """
        Hook for geodesics_equation method.
        Returns the local change in coordiantes and their velocities.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        # pylint: disable=W0107
        pass


class IdentityChart3D(Chart3DTo3D):
    """
    Chart embedding a subset of R^3 into itself via the identity transformation.

    Note: Can be used for mocking.
    """

    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        return coords

    def internal_hook_tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        return STANDARD_BASIS

    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        return IDENTITY_METRIC

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        return TangentialVectorDelta(tangent.vector, ZERO_VECTOR)

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        return TangentialVector(
            start,
            coordinates_as_vector(target) - coordinates_as_vector(start),
        )
