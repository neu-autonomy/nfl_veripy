from .ClosedLoopPartitioner import ClosedLoopPartitioner

class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)