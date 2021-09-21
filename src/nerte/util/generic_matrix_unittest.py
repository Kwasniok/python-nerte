# pylint: disable=R0801
# pylint: disable=C0103
# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=C0144
import unittest

from nerte.base_test_case import BaseTestCase

from nerte.util.generic_matrix import GenericMatrix


class GenericMatrixConstructorTest(BaseTestCase):
    def test_constructor(self) -> None:
        """Tests the constructor."""
        GenericMatrix[int]([])
        GenericMatrix[int]([[0]])
        GenericMatrix[int]([[0, 1], [2, 3]])
        with self.assertRaises(ValueError):
            GenericMatrix[int]([[0, 1], []])
            GenericMatrix[int]([[0], [2, 3]])


class GenericMatrixPropertiesTest(BaseTestCase):
    def setUp(self) -> None:
        self.mat0 = GenericMatrix[int]([])
        self.mat1 = GenericMatrix[int]([[0]])
        self.mat2 = GenericMatrix[int]([[0, 1], [2, 3], [4, 5]])

    def test_properties(self) -> None:
        """Tests the properties."""
        self.assertEqual(self.mat0.dimensions(), (0, 0))
        self.assertEqual(self.mat1.dimensions(), (1, 1))
        self.assertEqual(self.mat2.dimensions(), (3, 2))


class GenericMatrixMutationTest(BaseTestCase):
    def setUp(self) -> None:
        n = 2
        self.mat = GenericMatrix[tuple[int, int]](
            [[(0, 0) for _ in range(n)] for _ in range(n)]
        )

    def test_mutations(self) -> None:
        """Tests the mutations."""
        width, height = self.mat.dimensions()
        for i in range(width):
            for j in range(height):
                self.assertEqual(self.mat[i, j], (0, 0))
                self.mat[i, j] = (i, j)
                self.assertEqual(self.mat[i, j], (i, j))


if __name__ == "__main__":
    unittest.main()
