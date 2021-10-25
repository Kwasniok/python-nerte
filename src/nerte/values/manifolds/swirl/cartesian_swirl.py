"""Base module for representing manifolds in cartesian coordinates."""

from typing import Optional

import math

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    ZERO_MATRIX,
    Metric,
    Rank3Tensor,
    mat_vec_mult,
    dot,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.domains import Domain3D
from nerte.values.domains.cartesian_swirl import CartesianSwirlDomain
from nerte.values.manifolds.manifold_3d import Manifold3D


class CartesianSwirl(Manifold3D):
    """
    Represenation of the abstract swirl manifold with respect to cartesian
    coordinates.

    Note: This representation is the standard representation.
    """

    def __init__(self, swirl: float, domain: Optional[Domain3D] = None) -> None:

        if not math.isfinite(swirl):
            raise ValueError(
                f"Cannot construct cartesian swirl manifold with swirl={swirl}."
                f" Value must be finite."
            )

        if domain is None:
            domain = CartesianSwirlDomain(swirl)

        Manifold3D.__init__(self, domain)

        self.swirl = swirl

    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        # pylint: disable=C0103
        a = self.swirl
        u, v, z = coords
        # frequent factors
        r = math.sqrt(u ** 2 + v ** 2)
        s = u ** 2 - v ** 2
        auz = a * u * z
        avz = a * v * z
        auvrz = a * u * v * r * z
        return Metric(
            AbstractMatrix(
                AbstractVector(
                    (
                        1 + auz * ((-2 * v) / r + auz),
                        (a * z * (s + auvrz)) / r,
                        a * (-(v * r) + auz * (r ** 2)),
                    )
                ),
                AbstractVector(
                    (
                        (a * z * (s + auvrz)) / r,
                        1 + avz * ((2 * u) / r + avz),
                        a * (u * r + avz * (r ** 2)),
                    )
                ),
                AbstractVector(
                    (
                        a * (-(v * r) + auz * (r ** 2)),
                        a * (u * r + avz * (r ** 2)),
                        1 + a ** 2 * (r ** 2) ** 2,
                    )
                ),
            )
        )

    def internal_hook_christoffel_2(self, coords: Coordinates3D) -> Rank3Tensor:
        # pylint: disable=C0103
        a = self.swirl
        u, v, z = coords
        # frequent factors
        r = math.sqrt(u ** 2 + v ** 2)
        arz = a * r * z
        arz2 = arz ** 2
        ar = a * r
        az = a * z
        alpha = math.atan2(v, u)
        c_alpha = math.cos(alpha)
        s_alpha = math.sin(alpha)
        c_2alpha = math.cos(2 * alpha)
        s_2alpha = math.sin(2 * alpha)
        c_3alpha = math.cos(3 * alpha)
        s_3alpha = math.sin(3 * alpha)
        fac1 = (
            az
            * (
                (3 + arz2) * c_alpha
                + c_3alpha
                - arz * (arz * c_3alpha + s_alpha - 3 * s_3alpha)
            )
        ) / 4
        fac2 = (ar * (s_2alpha + arz * (2 * c_2alpha + arz * s_2alpha))) / 2
        fac3 = (
            ar * (-3 - arz2 + (1 + arz2) * c_2alpha - 2 * arz * s_2alpha)
        ) / 2
        fac4 = (
            az
            * (
                2 * s_alpha ** 3
                + arz * c_alpha * (-1 + 3 * c_2alpha + arz * s_2alpha)
            )
        ) / 2
        fac5 = (
            ar * (3 + arz2 + (1 + arz2) * c_2alpha - 2 * arz * s_2alpha)
        ) / 2
        fac6 = (
            az
            * (
                -(arz * (c_alpha + 3 * c_3alpha))
                - 2 * (1 + arz2 + (-1 + arz2) * c_2alpha) * s_alpha
            )
            / 4
        )
        fac7 = (
            az
            * s_alpha
            * (-5 - arz2 + (-1 + arz2) * c_2alpha - 3 * arz * s_2alpha)
        ) / 2
        fac8 = -(a ** 2 * r ** 3 * (c_alpha + arz * s_alpha))
        fac9 = (
            az
            * c_alpha
            * (5 + arz2 + (-1 + arz2) * c_2alpha - 3 * arz * s_2alpha)
        ) / 2
        fac10 = a ** 2 * r ** 3 * (arz * c_alpha - s_alpha)
        return Rank3Tensor(
            AbstractMatrix(
                AbstractVector((fac6, -fac1, -fac2)),
                AbstractVector((-fac1, fac7, fac3)),
                AbstractVector((-fac2, fac3, fac8)),
            ),
            AbstractMatrix(
                AbstractVector((fac9, fac4, fac5)),
                AbstractVector((fac4, fac1, fac2)),
                AbstractVector((fac5, fac2, fac10)),
            ),
            ZERO_MATRIX,
        )

    def _metric_inverted(self, coords: Coordinates3D) -> AbstractMatrix:
        # pylint: disable=C0103
        a = self.swirl
        u, v, z = coords
        # frequent factors
        r = math.sqrt(u ** 2 + v ** 2)
        s = u ** 2 - v ** 2
        aur = a * u * r
        avr = a * v * r
        u2v2z2 = u ** 2 + v ** 2 + z ** 2
        return AbstractMatrix(
            AbstractVector(
                (
                    1 + a * v * ((2 * u * z) / r + a * v * (r ** 2 + z ** 2)),
                    a * ((-s * z) / r - a * u * v * u2v2z2),
                    avr,
                )
            ),
            AbstractVector(
                (
                    a * ((-s * z) / r - a * u * v * u2v2z2),
                    1 + a * u * ((-2 * v * z) / r + a * u * u2v2z2),
                    -aur,
                )
            ),
            AbstractVector((avr, -aur, 1)),
        )

    def _christoffel_1(
        self, coords: Coordinates3D
    ) -> tuple[AbstractMatrix, AbstractMatrix, AbstractMatrix]:
        # pylint: disable=C0103
        a = self.swirl
        u, v, z = coords
        # frequent factors
        r = math.sqrt(u ** 2 + v ** 2)
        arz = a * r * z
        alpha = math.atan2(v, u)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        cos_2alpha = math.cos(2 * alpha)
        sin_2alpha = math.sin(2 * alpha)
        cos_3alpha = math.cos(3 * alpha)
        sin_3alpha = math.sin(3 * alpha)
        # POSSIBLE-OPTIMIZATION: symmetric matrices
        return (
            AbstractMatrix(
                AbstractVector(
                    (
                        a * z * (arz * cos_alpha - sin_alpha ** 3),
                        -a * z * cos_alpha ** 3,
                        a * r * cos_alpha * (arz * cos_alpha - sin_alpha),
                    )
                ),
                AbstractVector(
                    (
                        -a * z * cos_alpha ** 3,
                        -0.25
                        * a
                        * z
                        * (-4 * arz * cos_alpha + 9 * sin_alpha + sin_3alpha),
                        0.5 * a * r * (-3 + cos_2alpha + arz * sin_2alpha),
                    )
                ),
                AbstractVector(
                    (
                        a * r * cos_alpha * (arz * cos_alpha - sin_alpha),
                        0.5 * a * r * (-3 + cos_2alpha + arz * sin_2alpha),
                        -(a ** 2) * r ** 3 * cos_alpha,
                    )
                ),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        1
                        / 4
                        * a
                        * z
                        * (9 * cos_alpha - cos_3alpha + 4 * arz * sin_alpha),
                        a * z * sin_alpha ** 3,
                        1 / 2 * a * r * (3 + cos_2alpha + arz * sin_2alpha),
                    )
                ),
                AbstractVector(
                    (
                        a * z * sin_alpha ** 3,
                        a * z * (cos_alpha ** 3 + arz * sin_alpha),
                        a * r * sin_alpha * (cos_alpha + arz * sin_alpha),
                    )
                ),
                AbstractVector(
                    (
                        1 / 2 * a * r * (3 + cos_2alpha + arz * sin_2alpha),
                        a * r * sin_alpha * (cos_alpha + arz * sin_alpha),
                        -(a ** 2) * r ** 3 * sin_alpha,
                    )
                ),
            ),
            AbstractMatrix(
                AbstractVector(
                    (
                        1 / 2 * a ** 2 * r ** 2 * z * (3 + cos_2alpha),
                        a ** 2 * r ** 2 * z * cos_alpha * sin_alpha,
                        2 * a ** 2 * r ** 3 * cos_alpha,
                    )
                ),
                AbstractVector(
                    (
                        a ** 2 * r ** 2 * z * cos_alpha * sin_alpha,
                        -(1 / 2) * a ** 2 * r ** 2 * z * (-3 + cos_2alpha),
                        2 * a ** 2 * r ** 3 * sin_alpha,
                    )
                ),
                AbstractVector(
                    (
                        2 * a ** 2 * r ** 3 * cos_alpha,
                        2 * a ** 2 * r ** 3 * sin_alpha,
                        0,
                    )
                ),
            ),
        )

    def _christoffel_2(
        self, coords: Coordinates3D
    ) -> tuple[AbstractMatrix, AbstractMatrix, AbstractMatrix]:
        chris_1 = self._christoffel_1(coords)
        metric_inv = self._metric_inverted(coords)
        return (
            chris_1[0] * metric_inv[0][0]
            + chris_1[1] * metric_inv[0][1]
            + chris_1[2] * metric_inv[0][2],
            chris_1[0] * metric_inv[1][0]
            + chris_1[1] * metric_inv[1][1]
            + chris_1[2] * metric_inv[1][2],
            chris_1[0] * metric_inv[2][0]
            + chris_1[1] * metric_inv[2][1]
            + chris_1[2] * metric_inv[2][2],
        )

    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        chris_2 = self._christoffel_2(tangent.point)
        return TangentialVectorDelta(
            tangent.vector,
            AbstractVector(
                (
                    -dot(
                        tangent.vector, mat_vec_mult(chris_2[0], tangent.vector)
                    ),
                    -dot(
                        tangent.vector, mat_vec_mult(chris_2[1], tangent.vector)
                    ),
                    -dot(
                        tangent.vector, mat_vec_mult(chris_2[2], tangent.vector)
                    ),
                )
            ),
        )

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        # pylint: disable=C0103,R0914
        a = self.swirl
        u0, v0, z0 = start
        u1, v1, z1 = target
        # frequent factors
        r0 = math.sqrt(u0 ** 2 + v0 ** 2)
        r1 = math.sqrt(u1 ** 2 + v1 ** 2)
        alpha0 = math.atan2(v0, u0)
        alpha1 = math.atan2(v1, u1)
        arz0 = a * r0 * z0
        arz1 = a * r1 * z1
        phi0 = alpha0 + arz0
        phi1 = alpha1 + arz1
        vector = AbstractVector(
            (
                -r0 * math.cos(arz0 - phi0)
                + r1 * math.cos(arz0 - phi1)
                - a
                * r0
                * (-2 * r0 * z0 + r0 * z1 + r1 * z0 * math.cos(phi0 - phi1))
                * math.sin(arz0 - phi0),
                a * r0 ** 2 * (2 * z0 - z1) * math.cos(arz0 - phi0)
                - 1 / 2 * arz0 * r1 * math.cos(arz0 - phi1)
                - 1 / 2 * arz0 * r1 * math.cos(arz0 - 2 * phi0 + phi1)
                + r0 * math.sin(arz0 - phi0)
                - r1 * math.sin(arz0 - phi1),
                -z0 + z1,
            )
        )
        return TangentialVector(start, vector)
