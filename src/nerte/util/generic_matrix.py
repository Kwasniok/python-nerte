"""Module for generic mutable (object) matrix."""

from typing import TypeVar, Generic

T = TypeVar("T")  # pylint: disable=C0103


class GenericMatrix(Generic[T]):
    """Generic mutable (object) matrix."""

    def __init__(self, data: list[list[T]]) -> None:

        dim0 = len(data)
        if dim0 > 0:
            dim1 = len(data[0])
            for i in range(dim0):
                if not len(data[i]) == dim1:
                    raise ValueError(
                        f"Cannot construct object matrix form nested list {data}."
                        f" Matrix dimensions are inconsistent."
                    )
        else:
            dim1 = 0

        self._data = data
        self._dimensions = (dim0, dim1)

    def __getitem__(self, i: tuple[int, int]) -> T:
        return self._data[i[0]][i[1]]

    def __setitem__(self, i: tuple[int, int], value: T) -> None:
        self._data[i[0]][i[1]] = value

    def dimensions(self) -> tuple[int, int]:
        """Returns the dimensions of the matrix as a tuple."""
        return self._dimensions
