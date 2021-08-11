from .ClosedLoopPartitioner import ClosedLoopPartitioner


class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=None, make_animation=False, show_animation=False):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics, make_animation=make_animation, show_animation=show_animation)
