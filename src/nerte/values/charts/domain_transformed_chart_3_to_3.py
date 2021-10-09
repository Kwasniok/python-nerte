"""Module for domain transformed charts from 3D to 3D charts."""


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
# TODO: rename to pullback
class DomainTransformedChart3DTo3D(Chart3DTo3D):
    """
    A new chart based on chart f with a domain transformed by transfomation g,
    such that
        f(g(x0, x1, x2)) = (u0, u1, u2)
    where
        g : U -> V
        f : V -> w
    and
        U, V, W ⊆ R^3
    """

    def __init__(
        self, chart: Chart3DTo3D, transformation: Transformation3D
    ) -> None:

        Chart3DTo3D.__init__(self, transformation.domain)
        self.transformation = transformation
        self.chart = chart

    def internal_hook_embed(self, coords: Coordinates3D) -> Coordinates3D:
        return self.chart.embed(
            self.transformation.internal_hook_transform_coords(coords)
        )

    # TODO: test with non-id chart
    def internal_hook_tangential_space(
        self, coords: Coordinates3D
    ) -> tuple[AbstractVector, AbstractVector, AbstractVector]:
        # pylint: disable=C0103
        transformed_coords = self.transformation.internal_hook_transform_coords(
            coords
        )
        jacobian = self.transformation.internal_hook_jacobian(coords)
        v0, v1, v2 = self.chart.tangential_space(transformed_coords)
        inner_jacobian = transposed(AbstractMatrix(v0, v1, v2))
        total_jacobian = mat_mult(jacobian, inner_jacobian)
        total_basis = transposed(total_jacobian)
        return (total_basis[0], total_basis[1], total_basis[2])

    # TODO change to matrix
    def internal_hook_metric(self, coords: Coordinates3D) -> Metric:
        transformed_coords = self.transformation.internal_hook_transform_coords(
            coords
        )
        transformed_metric = self.chart.metric(transformed_coords).matrix()
        jacobian = self.transformation.internal_hook_jacobian(coords)
        return Metric(
            mat_mult(
                transposed(jacobian), mat_mult(transformed_metric, jacobian)
            )
        )

    # TODO: test for non-linear transformation
    def internal_hook_geodesics_equation(
        self, tangent: TangentialVector
    ) -> TangentialVectorDelta:
        transformed_point = self.transformation.internal_hook_transform_coords(
            tangent.point
        )
        jacobian = self.transformation.internal_hook_jacobian(tangent.point)
        inverse_jacobian = self.transformation.inverse_jacobian(tangent.point)
        transformed_vector = mat_vec_mult(jacobian, tangent.vector)
        transformed_delta = self.chart.geodesics_equation(
            TangentialVector(transformed_point, transformed_vector)
        )
        transformed_velocity = transformed_delta.point_delta
        velocity = mat_vec_mult(inverse_jacobian, transformed_velocity)
        transformed_acceleration = transformed_delta.vector_delta
        inverse_hesse = self.transformation.inverse_hesse_tensor(tangent.point)
        corr = AbstractVector(
            (
                -dot(velocity, mat_vec_mult(inverse_hesse[0], velocity)),
                -dot(velocity, mat_vec_mult(inverse_hesse[1], velocity)),
                -dot(velocity, mat_vec_mult(inverse_hesse[2], velocity)),
            )
        )
        acceleration = mat_vec_mult(
            inverse_jacobian, transformed_acceleration + corr
        )
        return TangentialVectorDelta(velocity, acceleration)

    def internal_hook_initial_geodesic_tangent_from_coords(
        self, start: Coordinates3D, target: Coordinates3D
    ) -> TangentialVector:
        transformed_start = self.transformation.internal_hook_transform_coords(
            start
        )
        transformed_target = self.transformation.internal_hook_transform_coords(
            target
        )
        inverse_jacobian = self.transformation.inverse_jacobian(start)
        transformed_vector = self.chart.initial_geodesic_tangent_from_coords(
            transformed_start,
            transformed_target,
        ).vector
        return TangentialVector(
            start, mat_vec_mult(inverse_jacobian, transformed_vector)
        )
