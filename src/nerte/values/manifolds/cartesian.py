"""Module for representing manifolds in cartesian coordinates."""

from typing import Optional

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
from nerte.values.linalg import AbstractMatrix, Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.manifold import Manifold1D, Manifold2D, Manifold3D


def cartesian_metric(coords: Coordinates3D) -> Metric:
    # pylint: disable=w0613
    """Returns the local metric in cartesian coordinates."""
    return Metric(
        AbstractMatrix(
            AbstractVector((1, 0, 0)),
            AbstractVector((0, 1, 0)),
            AbstractVector((0, 0, 1)),
        )
    )


def cartesian_geodesic_equation(
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    cartesian coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d^2x/d𝜆^2).
    """
    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector((0.0, 0.0, 0.0)),
    )


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
        Manifold1D.__init__(self, (domain,))

        self._direction = direction

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def embed(self, coords: Coordinates1D) -> Coordinates3D:
        self.in_domain_assertion(coords)
        return vector_as_coordinates(self._direction * coords[0] + self._offset)

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
