from nerte.face import Face


class Object:
    def __init__(self):
        self._faces = []

    def add_face(self, face: Face):
        self._faces.append(face)

    def faces(self):
        return self._faces
