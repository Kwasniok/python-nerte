"""Module for codomain transformed charts from 3D to 3D charts."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    AbstractMatrix,
    Metric,
    mat_vec_mult,
    mat_mult,
    dot,
    transposed,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.tangential_vector_delta import TangentialVectorDelta
from nerte.values.transformations import Transformation3D
from nerte.values.charts.chart_3_to_3 import Chart3DTo3D


# TODO: all methods must be revised
# TODO: rename to pushforward
class CodomainTransformedChart3DTo3D(Chart3DTo3D):
    """
    A new chart based on chart f with a domain transformed by transfomation g,
    such that
        g(f(x0, x1, x2)) = (u0, u1, u2)
    where
        f : U -> V
        g : V -> w
    and
        U, V, W ⊆ R^3
    """

    def __init__(
        self, transformation: Transformation3D, chart: Chart3DTo3D
    ) -> None:

        Chart3DTo3D.__init__(self, chart.domain)
        self.chart = chart
        self.transformation = transformation

    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        return self.transformation.transform_coords(
            self.chart.internal_hook_embed(coords)
        )

    # TODO: test with non-id chart
    def internal_hook_tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        # pylint: disable=C0103
        point = self.chart.internal_hook_embed(coords)
        jacobian_transposed = transposed(self.transformation.jacobian(point))
        v0, v1, v2 = self.chart.internal_hook_tangential_space(coords)
        basis_matrix = AbstractMatrix(v0, v1, v2)
        transformed_basis_matrix = mat_mult(basis_matrix, jacobian_transposed)
        return (
            transformed_basis_matrix[0],
            transformed_basis_matrix[1],
            transformed_basis_matrix[2],
        )

    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        point = self.chart.internal_hook_embed(coords)
        metric = self.chart.internal_hook_metric(coords).matrix()
        transformed_point = self.transformation.transform_coords(point)
        jacobian = self.transformation.inverse_jacobian(transformed_point)
        return Metric(
            mat_mult(transposed(jacobian), mat_mult(metric, jacobian))
        )

    # TODO: test for non-linear transformation
    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        jacobian = self.transformation.jacobian(tangent.point)
        delta = self.chart.internal_hook_geodesics_equation(tangent)
        velocity = delta.point_delta
        transformed_velocity = mat_vec_mult(jacobian, velocity)
        acceleration = delta.vector_delta
        hesse = self.transformation.hesse_tensor(tangent.point)
        corr = AbstractVector(
            (
                -dot(
                    transformed_velocity,
                    mat_vec_mult(hesse[0], transformed_velocity),
                ),
                -dot(
                    transformed_velocity,
                    mat_vec_mult(hesse[1], transformed_velocity),
                ),
                -dot(
                    transformed_velocity,
                    mat_vec_mult(hesse[1], transformed_velocity),
                ),
            )
        )
        transformed_acceleration = mat_vec_mult(jacobian, acceleration + corr)
        return TangentialVectorDelta(
            transformed_velocity, transformed_acceleration
        )

    # TODO: revise
    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        tangent = self.chart.internal_hook_initial_geodesic_tangent_from_coords(
            start, target
        )
        return self.transformation.internal_hook_transform_tangent(tangent)
