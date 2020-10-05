import numpy as np
import pypoman

class InputConstraint:
    def __init__(self):
        pass

class PolytopeInputConstraint(InputConstraint):
    def __init__(self, A, b):
        InputConstraint.__init__(self)
        self.A = A
        self.b = b

    def to_output_constraint(self):
        return PolytopeOutputConstraint(A=self.A)

    def to_linf(self):
        vertices = np.stack(pypoman.duality.compute_polytope_vertices(self.A, self.b))
        ranges = np.dstack([np.min(vertices, axis=0), np.max(vertices, axis=0)])[0]
        return ranges

class LpInputConstraint(InputConstraint):
    def __init__(self, p, range):
        InputConstraint.__init__(self)
        self.range = range
        self.p = p

    def to_output_constraint(self):
        return LpOutputConstraint(p=self.p, range=self.range)

class OutputConstraint:
    def __init__(self):
        pass

class PolytopeOutputConstraint(OutputConstraint):
    def __init__(self, A, b=None):
        OutputConstraint.__init__(self)
        self.A = A
        self.b = b

    def to_input_constraint(self):
        return PolytopeInputConstraint(A=self.A, b=self.b)

class LpOutputConstraint(OutputConstraint):
    def __init__(self, p, range=None):
        OutputConstraint.__init__(self)
        self.range = range
        self.p = p

    def to_input_constraint(self):
        return LpInputConstraint(p=self.p, range=self.range)

