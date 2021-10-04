"""Module for linear transformations."""

from nerte.values.coordinates import Coordinates3D
from nerte.values.linalg import AbstractMatrix, mat_vec_mult
from nerte.values.util.convert import (
    coordinates_as_vector,
    vector_as_coordinates,
)
from nerte.values.domains import Domain3D
from nerte.values.transformations.base import Transformation3D


class Linear(Transformation3D):
    """Linear transformation of three-dimensional coordinates."""

    def __init__(self, domain: Domain3D, matrix: AbstractMatrix) -> None:
        if not matrix.is_invertible():
            raise ValueError(
                f"Cannot construct scale transformation with"
                f" matrix={matrix}. Matrix must be invertible"
            )
        Transformation3D.__init__(self, domain)
        self.matrix = matrix

    def internal_hook_transform_coords(
        self, coords: Coordinates3D
    ) -> Coordinates3D:
        return vector_as_coordinates(
            mat_vec_mult(self.matrix, coordinates_as_vector(coords))
        )

    def internal_hook_jacobian(self, coords: Coordinates3D) -> AbstractMatrix:
        return self.matrix
