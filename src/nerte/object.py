"""Module for representing an object."""

from nerte.face import Face
from nerte.color import Color, Colors


class Object:
    """
    Representation of an object.
    An object consists of triangular faces and a color.
    """

    def __init__(self, color: Color = None):
        self._faces = []
        self.color = Colors.GRAY
        if color is not None:
            self.color = color

    def add_face(self, face: Face):
        """Adds a face to the object."""
        self._faces.append(face)

    def faces(self):
        """Returns an iterable object enumerating all faces."""
        return self._faces
