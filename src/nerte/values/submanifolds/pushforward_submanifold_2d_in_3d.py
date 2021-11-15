"""
Module for two-dimensional submanifolds in three-dimensional manifolds where
the representation is pushed forward.

"""

from nerte.values.coordinates import Coordinates2D, Coordinates3D
from nerte.values.linalg import (
    AbstractVector,
    ZERO_VECTOR,
    AbstractMatrix,
    mat_mult,
    transposed,
)
from nerte.values.tangential_vector import TangentialVector
from nerte.values.submanifolds.submanifold_2d_in_3d import Submanifold2DIn3D
from nerte.values.transitions.transition_3d import Transition3D


class PushforwardSubmanifold2DIn3D(Submanifold2DIn3D):
    """
    Representation of a two-dimensional submanifold in a three-dimensional
    manifold which is pushed forwards.

    Let f be the submanifold of the submanifold in the original representation and g
    a transition of representation then it represents h where
        h(x0, x1) = g(f(x0, x1)) = (u0, u1, u2)
    where
        f : U -> V
        g : V -> w
    and
        U ⊆ R^2, V, W ⊆ R^3
    """

    def __init__(
        self, submanifold: Submanifold2DIn3D, transition: Transition3D
    ) -> None:

        Submanifold2DIn3D.__init__(self, submanifold.domain)

        self.submanifold = submanifold
        self.transition = transition

    def internal_hook_embed(self, coords: Coordinates2D) -> Coordinates3D:
        return self.transition.transform_coords(self.submanifold.embed(coords))

    def internal_hook_tangential_space(
        self, coords: Coordinates2D
    ) -> tuple[AbstractVector, AbstractVector]:
        # pylint: disable=C0103
        point = self.submanifold.internal_hook_embed(coords)
        jacobian_transposed = transposed(self.transition.jacobian(point))
        v0, v1 = self.submanifold.internal_hook_tangential_space(coords)
        basis_matrix = AbstractMatrix(v0, v1, ZERO_VECTOR)
        transformed_basis_matrix = mat_mult(basis_matrix, jacobian_transposed)
        return (transformed_basis_matrix[0], transformed_basis_matrix[1])

    def internal_hook_surface_normal(
        self, coords: Coordinates2D
    ) -> AbstractVector:
        point = self.submanifold.internal_hook_embed(coords)
        normal = self.submanifold.internal_hook_surface_normal(coords)
        return self.transition.transform_tangent(
            TangentialVector(point, normal)
        ).vector
