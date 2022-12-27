import numpy as np
import nn_partition.propagators as propagators
from copy import deepcopy
import nn_closed_loop.constraints as constraints


class ClosedLoopPropagator(propagators.Propagator):
    def __init__(self, input_shape=None, dynamics=None, boundary_type="rectangle", num_polytope_facets=None):
        super().__init__(input_shape=input_shape)
        self.dynamics = dynamics
        self.boundary_type = boundary_type
        self.num_polytope_facets = num_polytope_facets

    def get_one_step_reachable_set(self, input_constraint):
        raise NotImplementedError

    def get_reachable_set(self, initial_set, t_max):
        reachable_sets = []
        info = {'per_timestep': []}
        reachable_set, this_info = self.get_one_step_reachable_set(
            initial_set
        )
        reachable_sets.append(deepcopy(reachable_set))
        info['per_timestep'].append(this_info)
        for i in np.arange(0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt):
            next_initial_set = deepcopy(reachable_set)
            reachable_set, this_info = self.get_one_step_reachable_set(
                next_initial_set
            )
            reachable_sets.append(deepcopy(reachable_set))
            info['per_timestep'].append(this_info)

        # [LpConstraint, LpConstraint, ..., LpConstraint] -> LpConstraint(range=(num_timesteps, num_states, 2))
        # [PolytopeConstraint, PolytopeConstraint, ..., PolytopeConstraint] -> PolytopeConstraint(A=(num_timesteps, num_facets, num_states))
        reachable_sets = constraints.list_to_constraint(reachable_sets)

        return reachable_sets, info

    def get_one_step_backprojection_set(self, target_set, overapprox=False,):
        raise NotImplementedError

    def get_backprojection_set(self, target_sets, t_max, num_partitions=None, overapprox=False, refined=False):
        input_constraint_list = []
        tightened_infos_list = []
        if not isinstance(target_sets, list):
            target_set_list = [deepcopy(target_sets)]
        else:
            target_set_list = deepcopy(target_sets)

        for target_set in target_set_list:
            backprojection_sets, tightened_infos = self.get_single_target_backprojection_set(target_set, t_max=t_max, num_partitions=num_partitions, overapprox=overapprox, refined=refined)

            backprojection_set_list.append(deepcopy(backprojection_sets))
            tightened_infos_list.append(deepcopy(tightened_infos))

        return backprojection_set_list, tightened_infos_list

    def get_single_target_backprojection_set(self, target_set, t_max, num_partitions=None, overapprox=False, refined=False):

        backprojection_set, info = self.get_one_step_backprojection_set(
            target_set, num_partitions=num_partitions, overapprox=overapprox, refined=refined
        )

        return backprojection_set, info

    def output_to_constraint(self, bs, constraint):
        raise NotImplementedError
        if isinstance(constraint, constraints.PolytopeOutputConstraint):
            constraint.b = bs
        elif isinstance(constraint, constraints.LpOutputConstraint):
            constraint.range = np.empty((num_states, 2))
            constraint.range[:, 0] = -bs[(num_facets // 2):]
            constraint.range[:, 1] = bs[:(num_facets // 2)]
        elif isinstance(
            constraint, constraints.EllipsoidOutputConstraint
        ):
            constraint.center = b
            constraint.shape = A
        else:
            raise NotImplementedError
        return constraint
