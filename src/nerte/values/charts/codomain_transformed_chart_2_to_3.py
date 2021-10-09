"""Module for codomain transformed charts from 2D to 3D charts."""

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    AbstractMatrix,
    mat_mult,
    transposed,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.transformations import Transformation3D
from nerte.values.charts.chart_2_to_3 import Chart2DTo3D

# TODO: rename to pushforward
class CodomainTransformedChart2DTo3D(Chart2DTo3D):
    """
    A new chart based on chart f with a domain transformed by transfomation g,
    such that
        g(f(x0, x1)) = (u0, u1, u2)
    where
        f : U -> V
        g : V -> w
    and
        U ⊆ R^2, V, W ⊆ R^3
    """

    def __init__(
        self, transformation: Transformation3D, chart: Chart2DTo3D
    ) -> None:

        Chart2DTo3D.__init__(self, chart.domain)
        self.chart = chart
        self.transformation = transformation

    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        return self.transformation.transform_coords(
            self.chart.internal_hook_embed(coords)
        )

    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        # pylint: disable=C0103
        point = self.chart.internal_hook_embed(coords)
        jacobian_transposed = transposed(self.transformation.jacobian(point))
        v0, v1 = self.chart.internal_hook_tangential_space(coords)
        basis_matrix = AbstractMatrix(v0, v1, ZERO_VECTOR)
        transformed_basis_matrix = mat_mult(basis_matrix, jacobian_transposed)
        return (transformed_basis_matrix[0], transformed_basis_matrix[1])

    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        point = self.chart.internal_hook_embed(coords)
        normal = self.chart.internal_hook_surface_normal(coords)
        return self.transformation.transform_tangent(
            TangentialVector(point, normal)
        ).vector
