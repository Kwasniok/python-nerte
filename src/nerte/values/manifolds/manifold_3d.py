"""Module for abstract three-dimensional (differentiable) manifolds."""

from abc import ABC, abstractmethod


import math

from nerte.values.domains import OutOfDomainError, Domain3D
from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    Metric,
    Rank3Tensor,
    mat_vec_mult,
    tensor_3_vec_contract,
    dot,
    is_zero_vector,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta


class Manifold3D(ABC):
    """
    Representation of an abstract three-dimensional (differentiable) manifold.
    """

    def __init__(self, domain: Domain3D) -> None:
        self.domain = domain

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

    def christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        """
        Returns the (local) Christoffel symbols of the second kind.

        :raises: OutOfDomainError if the given coordinates are outside of the
                 domain.
        """
        try:
            self.domain.assert_inside(coords)
        except OutOfDomainError as ex:
            raise OutOfDomainError(
                f"Cannot create Christoffel symbols of second kind at"
                f" coordinates {coords}."
                f" Coordinates out of domain. "
                + self.domain.not_inside_reason(coords)
            ) from ex
        return self.internal_hook_christoffel_2(coords)

    @abstractmethod
    def internal_hook_christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        """
        Hook for christoffel_2 method.
        Returns the (local) Christoffel symbols of the second kind.
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

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        """
        Hook for geodesics_equation method.
        Returns the local change in coordiantes and their velocities.
        IMPORTANT: It is trusted that self.domain.assert_inside(coords) is True.
        """
        christoffel_2 = self.internal_hook_christoffel_2(tangent.point)
        acceleration = -mat_vec_mult(
            tensor_3_vec_contract(christoffel_2, tangent.vector, 2),
            tangent.vector,
        )
        return TangentialVectorDelta(tangent.vector, acceleration)

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
