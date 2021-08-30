"""Module for representing an object."""

from typing import Optional

from nerte.geometry.face import Face
from nerte.color import Color, Colors


class Object:
    """
    Representation of an object.
    An object consists of triangular faces and a color.
    """

    def __init__(self, color: Optional[Color] = None) -> None:
        self._faces: list[Face] = []
        self.color = Colors.GRAY
        if color is not None:
            self.color = color

    def add_face(self, face: Face) -> None:
        """Adds a face to the object."""
        self._faces.append(face)

    def faces(self) -> list[Face]:
        """Returns an iterable object enumerating all faces."""
        return self._faces
