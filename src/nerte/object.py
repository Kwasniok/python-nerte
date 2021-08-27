from nerte.face import Face
from nerte.color import Color, GRAY


class Object:
    def __init__(self, color: Color = None):
        self._faces = []
        self.color = GRAY
        if color is not None:
            self.color = color

    def add_face(self, face: Face):
        self._faces.append(face)

    def faces(self):
        return self._faces
