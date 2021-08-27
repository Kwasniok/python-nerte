from nerte.face import Face


class Object:
    def __init__(self, color=None):
        self._faces = []
        self.color = color

    def add_face(self, face: Face):
        self._faces.append(face)

    def faces(self):
        return self._faces
