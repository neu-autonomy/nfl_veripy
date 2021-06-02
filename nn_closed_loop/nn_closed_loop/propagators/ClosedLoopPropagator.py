import numpy as np
import nn_partition.propagators as propagators
from copy import deepcopy
import nn_closed_loop.constraints as constraints


class ClosedLoopPropagator(propagators.Propagator):
    def __init__(self, input_shape=None, dynamics=None):
        propagators.Propagator.__init__(self, input_shape=input_shape)
        self.dynamics = dynamics

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        raise NotImplementedError

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        output_constraints = []
        output_constraint, _ = self.get_one_step_reachable_set(
            input_constraint, output_constraint
        )
        output_constraints.append(deepcopy(output_constraint))
        for i in np.arange(0 + self.dynamics.dt, t_max, self.dynamics.dt):
            next_input_constraint = output_constraint.to_input_constraint()
            next_output_constraint = deepcopy(output_constraint)
            output_constraint, _ = self.get_one_step_reachable_set(
                next_input_constraint, next_output_constraint
            )
            output_constraints.append(deepcopy(output_constraint))
        return output_constraints, {}

    def get_one_step_backprojection_set(self, output_constraint, intput_constraint):
        raise NotImplementedError

    def get_backprojection_set(self, output_constraint, input_constraint, t_max):
        input_constraints = []
        input_constraint, _ = self.get_one_step_backprojection_set(
            output_constraint, input_constraint
        )
        input_constraints.append(deepcopy(input_constraint))
        for i in np.arange(0 + self.dynamics.dt, t_max, self.dynamics.dt):
            next_output_constraint = input_constraint.to_output_constraint()
            next_input_constraint = deepcopy(input_constraint)
            input_constraint, _ = self.get_one_step_backprojection_set(
                next_output_constraint, next_input_constraint
            )
            input_constraints.append(deepcopy(input_constraint))
        return input_constraints, {}

    def output_to_constraint(self, bs, output_constraint):
        raise NotImplementedError
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            output_constraint.range = np.empty((num_states, 2))
            output_constraint.range[:, 0] = -bs[(num_facets // 2):]
            output_constraint.range[:, 1] = bs[:(num_facets // 2)]
        elif isinstance(
            output_constraint, constraints.EllipsoidOutputConstraint
        ):
            output_constraint.center = b
            output_constraint.shape = A
        else:
            raise NotImplementedError
        return output_constraint
