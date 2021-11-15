"""Module for linear transformations."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import (
    AbstractMatrix,
    Rank3Tensor,
    ZERO_RANK3TENSOR,
    mat_vec_mult,
    inverted,
)
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)
from nerte.values.domains import Domain3D
from nerte.values.transitions.transition_3d import Transition3D


class Linear3D(Transition3D):
    """Linear transformation of three-dimensional coordinates."""

    def __init__(
        self, domain: Domain3D, codomain: Domain3D, matrix: AbstractMatrix
    ) -> None:
        if not matrix.is_invertible():
            raise ValueError(
                f"Cannot construct scale transformation with"
                f" matrix={matrix}. Matrix must be invertible"
            )
        Transition3D.__init__(self, domain, codomain)
        self.matrix = matrix
        self.inverse_matrix = inverted(matrix)

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return vector_as_coordinates(
            mat_vec_mult(self.matrix, coordinates_as_vector(coords))
        )

    def internal_hook_inverse_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return vector_as_coordinates(
            mat_vec_mult(self.inverse_matrix, coordinates_as_vector(coords))
        )

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return self.matrix

    def internal_hook_inverse_jacobian(
        self, coords: Coordinates3D
    ) -> AbstractMatrix:
        return self.inverse_matrix

    def internal_hook_hesse_tensor(self, coords: Coordinates3D) -> Rank3Tensor:
        return ZERO_RANK3TENSOR

    def internal_hook_inverse_hesse_tensor(
        self, coords: Coordinates3D
    ) -> Rank3Tensor:
        return ZERO_RANK3TENSOR
