from .ClosedLoopPartitioner import ClosedLoopPartitioner

import nn_closed_loop.dynamics as dynamics
import numpy as np
from typing import Optional

class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics: dynamics.Dynamics, num_partitions: Optional[np.ndarray] = None, make_animation: bool = False, show_animation: bool = False):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics, make_animation=make_animation, show_animation=show_animation)
