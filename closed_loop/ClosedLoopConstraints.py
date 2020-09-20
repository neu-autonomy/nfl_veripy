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

