import nfl_veripy.dynamics as dynamics

from .ClosedLoopPartitioner import ClosedLoopPartitioner


class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        super().__init__(dynamics=dynamics)
