class Coordinates:
    def __init__(self, x1, x2, x3):
        self._x = (x1, x2, x3)

    def __repr__(self):
        return "C" + repr(self._x)

    def __getitem__(self, i: int) -> float:
        return self._x[i]
