import numpy as np
import partition.propagators as propagators
from copy import deepcopy

class ClosedLoopPropagator(propagators.Propagator):
    def __init__(self, input_shape=None, dynamics=None):
        propagators.Propagator.__init__(self, input_shape=input_shape)
        self.dynamics = dynamics

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        raise NotImplementedError

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        output_constraints = []
        output_constraint, _ = self.get_one_step_reachable_set(input_constraint, output_constraint)
        output_constraints.append(deepcopy(output_constraint))
        for i in np.arange(0+self.dynamics.dt, t_max, self.dynamics.dt):
            next_input_constraint = output_constraint.to_input_constraint()
            next_output_constraint = deepcopy(output_constraint)
            output_constraint, _ = self.get_one_step_reachable_set(next_input_constraint, next_output_constraint)
            output_constraints.append(deepcopy(output_constraint))
        return output_constraints, {}
