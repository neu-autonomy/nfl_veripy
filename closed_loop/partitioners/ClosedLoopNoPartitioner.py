from .ClosedLoopPartitioner import ClosedLoopPartitioner

class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=None):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)