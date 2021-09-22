"""Module for representing manifolds in carthesian swirl coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.domain import Domain1D
from nerte.values.linalg import (
    AbstractVector,
    mat_vec_mult,
    cross,
    are_linear_dependent,
)
from nerte.values.util.convert import vector_as_coordinates
from nerte.values.linalg import AbstractMatrix, Metric
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.manifold import Manifold2D


def carthesian_swirl_metric(swirl: float, coords: Coordinates3D) -> Metric:
    # pylint: disable=C0103
    """Returns the local metric in carthesian swirl coordinates."""
    a = swirl
    x, y, z = coords
    r = math.sqrt(x ** 2 + y ** 2)
    if not 0 < r < math.inf:
        raise ValueError(
            f"Cannot generate matric for carthesian swirl={swirl} coordinates"
            f" at (x, y, z)={coords}."
            f" Coordinate values must be restricted to "
            f" 0 < r = sqrt(x ** 2 + y ** 2)."
        )

    # frequent factors
    s = x ** 2 - y ** 2
    axz = a * x * z
    ayz = a * y * z
    axyrz = a * x * y * r * z

    return Metric(
        AbstractMatrix(
            AbstractVector(
                (
                    1 + axz * ((-2 * y) / r + axz),
                    (a * z * (s + axyrz)) / r,
                    a * (-(y * r) + axz * (r ** 2)),
                )
            ),
            AbstractVector(
                (
                    (a * z * (s + axyrz)) / r,
                    1 + ayz * ((2 * x) / r + ayz),
                    a * (x * r + ayz * (r ** 2)),
                )
            ),
            AbstractVector(
                (
                    a * (-(y * r) + axz * (r ** 2)),
                    a * (x * r + ayz * (r ** 2)),
                    1 + a ** 2 * (r ** 2) ** 2,
                )
            ),
        )
    )


def carthesian_swirl_geodesic_equation(
    swirl: float,
    tangent: TangentialVector,
) -> TangentialVectorDelta:
    # pylint: disable=C0103,R0914
    """
    Returns a tangential vector delta which encodes the geodesic equation of
    carthesian swirl coordinates.

    Let x(𝜆) be a geodesic.
    For tangent (x, dx/d𝜆) it returns (dx/d𝜆, d**2x/d𝜆**2).
    """
    a = swirl
    x, y, z = tangent.point
    vx, vy, vz = tangent.vector[0], tangent.vector[1], tangent.vector[2]

    # radial factors
    r = math.sqrt(x ** 2 + y ** 2)
    if not 0 < r < math.inf:
        raise ValueError(
            f"Cannot generate geodesic equation for carthesian swirl={swirl}"
            f" coordinates at (x, y, z)={tangent.point}."
            f" Coordinate values must be restricted to "
            f" 0 < r = sqrt(x ** 2 + y ** 2)."
        )
    R = 1 / r

    # frequent factors
    uyz = vy * vz
    uxz = vx * vz

    # polynomial factors for vx
    faR_x = 2 * uyz * x ** 2 + 2 * uxz * x * y + 4 * uyz * y ** 2
    faR3_x = (
        2 * vx * vy * x ** 3 * z
        + 3 * vy ** 2 * x ** 2 * y * z
        + (vx ** 2 + 2 * vy ** 2) * y ** 3 * z
    )
    fa2r2_x = vz ** 2 * x
    fa2R2_x = (
        2 * uxz * x ** 4 * z
        + 4 * uyz * x ** 3 * y * z
        + 4 * uyz * x * y ** 3 * z
        - 2 * uxz * y ** 4 * z
    )
    fa2R4_x = (
        vx ** 2 * x ** 5 * z ** 2
        + 4 * vx * vy * x ** 4 * y * z ** 2
        + (-(vx ** 2) + 3 * vy ** 2) * x ** 3 * y ** 2 * z ** 2
        + 2 * vx * vy * x ** 2 * y ** 3 * z ** 2
        + (-2 * vx ** 2 + 3 * vy ** 2) * x * y ** 4 * z ** 2
        - 2 * vx * vy * y ** 5 * z ** 2
    )
    fa3r3_x = vz ** 2 * y * z
    fa3R_x = (
        2 * uxz * x ** 3 * y * z ** 2
        + 2 * uyz * x ** 2 * y ** 2 * z ** 2
        + 2 * uxz * x * y ** 3 * z ** 2
        + 2 * uyz * y ** 4 * z ** 2
    )
    fa3R3_x = (
        vx ** 2 * x ** 4 * y * z ** 3
        + 2 * vx * vy * x ** 3 * y ** 2 * z ** 3
        + (vx ** 2 + vy ** 2) * x ** 2 * y ** 3 * z ** 3
        + 2 * vx * vy * x * y ** 4 * z ** 3
        + vy ** 2 * y ** 5 * z ** 3
    )

    # polynomial factors for vy
    faR_y = -4 * uxz * x ** 2 - 2 * uyz * x * y - 2 * uxz * y ** 2
    faR3_y = (
        (-2 * vx ** 2 - vy ** 2) * x ** 3 * z
        - 3 * vx ** 2 * x * y ** 2 * z
        - 2 * vx * vy * y ** 3 * z
    )
    fa2r2_y = vz ** 2 * y
    fa2R2_y = (
        -2 * uyz * x ** 4 * z
        + 4 * uxz * x ** 3 * y * z
        + 4 * uxz * x * y ** 3 * z
        + 2 * uyz * y ** 4 * z
    )
    fa2R4_y = (
        -2 * vx * vy * x ** 5 * z ** 2
        + (3 * vx ** 2 - 2 * vy ** 2) * x ** 4 * y * z ** 2
        + 2 * vx * vy * x ** 3 * y ** 2 * z ** 2
        + (3 * vx ** 2 - vy ** 2) * x ** 2 * y ** 3 * z ** 2
        + 4 * vx * vy * x * y ** 4 * z ** 2
        + vy ** 2 * y ** 5 * z ** 2
    )
    fa3r3_y = -(vz ** 2) * x * z
    fa3R_y = (
        -2 * uxz * x ** 4 * z ** 2
        - 2 * uyz * x ** 3 * y * z ** 2
        - 2 * uxz * x ** 2 * y ** 2 * z ** 2
        - 2 * uyz * x * y ** 3 * z ** 2
    )
    fa3R3_y = (
        -(vx ** 2 * x ** 5 * z ** 3)
        - 2 * vx * vy * x ** 4 * y * z ** 3
        + (-(vx ** 2) - vy ** 2) * x ** 3 * y ** 2 * z ** 3
        - 2 * vx * vy * x ** 2 * y ** 3 * z ** 3
        - vy ** 2 * x * y ** 4 * z ** 3
    )

    return TangentialVectorDelta(
        tangent.vector,
        AbstractVector(
            (
                a * (R * faR_x + R ** 3 * faR3_x)
                + a ** 2
                * (r ** 2 * fa2r2_x + R ** 2 * fa2R2_x + R ** 4 * fa2R4_x)
                + a ** 3 * (r ** 3 * fa3r3_x + R * fa3R_x + R ** 3 * fa3R3_x),
                a * (R * faR_y + R ** 3 * faR3_y)
                + a ** 2
                * (r ** 2 * fa2r2_y + R ** 2 * fa2R2_y + R ** 4 * fa2R4_y)
                + a ** 3 * (r ** 3 * fa3r3_y + R * fa3R_y + R ** 3 * fa3R3_y),
                0,
            )
        ),
    )


def carthesian_to_carthesian_swirl_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns carthesian swirl coordinates obtained from carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian coordinates={coords} to carthesian swirl"
            f" coordinates. All values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    if not 0 < r < math.inf:
        raise ValueError(
            f"Cannot convert carthesian coordinates={coords} to carthesian swirl"
            f" coordinates. All carthesian swirl coordinates are restricted by"
            f" 0 < r but r={r}."
        )
    phi = math.atan2(y, x)
    return Coordinates3D(
        (r * math.cos(a * r * z + phi), r * math.sin(a * r * z + phi), z)
    )


def carthesian_swirl_to_carthesian_coords(
    swirl: float,
    coords: Coordinates3D,
) -> Coordinates3D:
    """
    Returns carthesian coordinates obtained from carthesian swirl coordinates.

    :param coords: carthesian swirl coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    """
    return carthesian_to_carthesian_swirl_coords(-swirl, coords)


def carthesian_to_carthesian_swirl_vector(
    swirl: float, coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of carthesian swirl coordinate
    from a vector in tangential vector space in carthesian coordinates.

    :param coords: carthesian coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    :param vec: vector in tangential vector space of the carthesian coordinates
                (x, y, z) such that vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian vector={vec} @ coordinates"
            f" (x,y,z)={coords} to carthesian swirl={swirl} vector."
            f" All carthesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    if not 0 < r < math.inf:
        raise ValueError(
            f"Cannot convert carthesian vector={vec} @ coordinates"
            f" (x,y,z)={coords} to carthesian swirl={swirl} vector."
            f" All carthesian coordinates are restricted by"
            f" 0 < r."
            f" Here r={r}."
        )
    # frequent values
    phi = math.atan2(y, x)
    arz = a * r * z
    alpha = phi - arz  # deswirl
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector(
            (
                ((x + y * arz) * cos_alpha + y * sin_alpha) / r,
                -(y * cos_alpha + (x + y * arz) * sin_alpha) / r,
                a * y * r,
            )
        ),
        AbstractVector(
            (
                ((y - x * arz) * cos_alpha - x * sin_alpha) / r,
                (x * cos_alpha + (y - x * arz) * sin_alpha) / r,
                -(a * x * r),
            )
        ),
        AbstractVector((0, 0, 1)),
    )

    return mat_vec_mult(jacobian, vec)


def carthesian_swirl_to_carthesian_vector(
    swirl: float, coords: Coordinates3D, vec: AbstractVector
) -> AbstractVector:
    """
    Returns vector in tangential vector space of carthesian coordinates from
    a vector in tangential vector space in carthesian swirl  coordinates.

    :param coords: carthesian swirl coordinates (x, y, z)
                   where -inf < x < inf and -inf < y < inf and -inf < z < inf
                   and 0 < r = sqrt(x^2 + y^2)
    :param vec: vector in tangential vector space of the carthesian swirl
                coordinates (x, y, z) such that
                vec = e_x * x + e_y * y + e_z * z
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = coords
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian swirl={swirl} vector={vec} @ coordinates"
            f" (x,y,z)={coords} to carthesian vector."
            f" All carthesian swirl coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    if not 0 < r < math.inf:
        raise ValueError(
            f"Cannot convert carthesian swirl={swirl} vector={vec} @ coordinates"
            f" (x,y,z)={coords} to carthesian vector."
            f" All carthesian swirl coordinates are restricted by"
            f" 0 < r."
            f" Here r={r}."
        )
    # frequent values
    alpha = math.atan2(y, x)  # already swirled
    arz = a * r * z
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector(
            (
                (x * cos_alpha + (y - x * arz) * sin_alpha) / r,
                (y * cos_alpha - (x + y * arz) * sin_alpha) / r,
                -(a * (r ** 2) * sin_alpha),
            )
        ),
        AbstractVector(
            (
                ((-y + x * arz) * cos_alpha + x * sin_alpha) / r,
                ((x + y * arz) * cos_alpha + y * sin_alpha) / r,
                a * (r ** 2) * cos_alpha,
            )
        ),
        AbstractVector((0, 0, 1)),
    )
    return mat_vec_mult(jacobian, vec)


def carthesian_to_carthesian_swirl_tangential_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from carthesian to cylindircal
    coordinates.
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = tangential_vector.point
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian tangential vector={tangential_vector}"
            f" to carthesian swirl={swirl} tangential vector."
            f" All carthesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    if 0 < r < math.inf:
        raise ValueError(
            f"Cannot convert carthesian tangential vector={tangential_vector}"
            f" to carthesian swirl={swirl} tangential vector."
            f" All carthesian coordinates are restricted by"
            f" 0 < r."
            f" Here r={r}."
        )
    # frequent values
    phi = math.atan2(y, x)
    arz = a * r * z
    alpha = phi + arz  # swirl
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector(
            (
                ((x + y * arz) * cos_alpha + y * sin_alpha) / r,
                -(y * cos_alpha + (x + y * arz) * sin_alpha) / r,
                a * y * r,
            )
        ),
        AbstractVector(
            (
                ((y - x * arz) * cos_alpha - x * sin_alpha) / r,
                (x * cos_alpha + (y - x * arz) * sin_alpha) / r,
                -(a * x * r),
            )
        ),
        AbstractVector((0, 0, 1)),
    )

    return TangentialVector(
        point=Coordinates3D((r * cos_alpha, r * sin_alpha, z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )


def carthesian_swirl_to_carthesian_tangential_vector(
    swirl: float,
    tangential_vector: TangentialVector,
) -> TangentialVector:
    """
    Returns tangential vector transformed from carthesian swirl to carthesian
    coordinates.
    """
    # pylint:disable=C0103
    a = swirl
    x, y, z = tangential_vector.point
    if (
        not -math.inf < x < math.inf
        or not -math.inf < y < math.inf
        or not -math.inf < z < math.inf
    ):
        raise ValueError(
            f"Cannot convert carthesian swirl={swirl} tangential vector={tangential_vector}"
            f" to carthesian tangential vector."
            f" All carthesian coordinate values must be finte."
        )
    r = math.sqrt(x ** 2 + y ** 2)
    if 0 < r < math.inf:
        raise ValueError(
            f"Cannot convert carthesian swirl={swirl} tangential vector={tangential_vector}"
            f" to carthesian tangential vector."
            f" All carthesian coordinates are restricted by"
            f" 0 < r."
            f" Here r={r}."
        )
    # frequent values
    alpha = math.atan2(y, x)  # already swirled
    arz = a * r * z
    phi = alpha - arz  # deswirl
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    # jacobian
    jacobian = AbstractMatrix(
        AbstractVector(
            (
                (x * cos_alpha + (y - x * arz) * sin_alpha) / r,
                (y * cos_alpha - (x + y * arz) * sin_alpha) / r,
                -(a * (r ** 2) * sin_alpha),
            )
        ),
        AbstractVector(
            (
                ((-y + x * arz) * cos_alpha + x * sin_alpha) / r,
                ((x + y * arz) * cos_alpha + y * sin_alpha) / r,
                a * (r ** 2) * cos_alpha,
            )
        ),
        AbstractVector((0, 0, 1)),
    )
    return TangentialVector(
        point=Coordinates3D((r * math.cos(phi), r * math.sin(phi), z)),
        vector=mat_vec_mult(jacobian, tangential_vector.vector),
    )


class Plane(Manifold2D):
    """
    Representation of a two-dimensional plane embedded in carthesian swirl
    coordinates.
    """

    def __init__(  # pylint: disable=R0913
        self,
        swirl: float,
        b0: AbstractVector,
        b1: AbstractVector,
        x0_domain: Optional[Domain1D] = None,
        x1_domain: Optional[Domain1D] = None,
        offset: Optional[AbstractVector] = None,
    ):
        if not -math.inf < swirl < math.inf:
            raise ValueError(
                f"Cannot construct plane. Swirl={swirl} must be finite."
            )

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

        self._swirl = swirl
        self._b0 = b0
        self._b1 = b1
        self._n = cross(b0, b1)
        self._carthesian_basis_vectors = (self._b0, self._b1)

        if offset is None:
            self._offset = AbstractVector((0.0, 0.0, 0.0))
        else:
            self._offset = offset

    def _embed_in_carthesian_coordinates(
        self, coords: Coordinates2D
    ) -> Coordinates3D:
        self.in_domain_assertion(coords)
        point = self._b0 * coords[0] + self._b1 * coords[1] + self._offset
        return vector_as_coordinates(point)

    def embed(self, coords: Coordinates2D) -> Coordinates3D:
        coords3d = self._embed_in_carthesian_coordinates(coords)
        return carthesian_to_carthesian_swirl_coords(self._swirl, coords3d)

    def surface_normal(self, coords: Coordinates2D) -> AbstractVector:
        coords3d = self._embed_in_carthesian_coordinates(coords)
        return carthesian_to_carthesian_swirl_vector(
            self._swirl, coords3d, self._n
        )

    def tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        coords3d = self._embed_in_carthesian_coordinates(coords)
        return (
            carthesian_to_carthesian_swirl_vector(
                self._swirl, coords3d, self._b0
            ),
            carthesian_to_carthesian_swirl_vector(
                self._swirl, coords3d, self._b1
            ),
        )
