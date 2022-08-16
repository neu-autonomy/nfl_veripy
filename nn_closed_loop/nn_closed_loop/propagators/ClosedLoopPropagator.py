from random import randrange
import numpy as np
import nn_partition.propagators as propagators
from copy import deepcopy
import nn_closed_loop.constraints as constraints
from itertools import product


class ClosedLoopPropagator(propagators.Propagator):
    def __init__(self, input_shape=None, dynamics=None):
        propagators.Propagator.__init__(self, input_shape=input_shape)
        self.dynamics = dynamics

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        raise NotImplementedError

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        output_constraints = []
        info = {'per_timestep': []}
        output_constraint, this_info = self.get_one_step_reachable_set(
            input_constraint, output_constraint
        )
        output_constraints.append(deepcopy(output_constraint))
        info['per_timestep'].append(this_info)
        for i in np.arange(0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt):
            next_input_constraint = deepcopy(output_constraint)
            next_output_constraint = deepcopy(output_constraint)
            output_constraint, this_info = self.get_one_step_reachable_set(
                next_input_constraint, next_output_constraint
            )
            output_constraints.append(deepcopy(output_constraint))
            info['per_timestep'].append(this_info)

        return output_constraints, info

    def get_one_step_backprojection_set(self, output_constraint, intput_constraint, overapprox=False,):
        raise NotImplementedError


    def get_backprojection_set(self, output_constraints, input_constraint, t_max, num_partitions=None, overapprox=False, refined=False, heuristic='guided', all_lps=False, slow_cvxpy=False):
        input_constraint_list = []
        tightened_infos_list = []
        if not isinstance(output_constraints, list):
            output_constraint_list = [deepcopy(output_constraints)]
        else:
            output_constraint_list = deepcopy(output_constraints)

        for output_constraint in output_constraint_list:
            input_constraints, tightened_infos = self.get_single_target_backprojection_set(output_constraint, input_constraint, t_max=t_max, num_partitions=num_partitions, overapprox=overapprox, refined=refined, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy)

            input_constraint_list.append(deepcopy(input_constraints))
            tightened_infos_list.append(deepcopy(tightened_infos))

        return input_constraint_list, tightened_infos_list

    def get_single_target_backprojection_set(self, output_constraint, input_constraint, t_max, num_partitions=None, overapprox=False, refined=False, heuristic='guided', all_lps=False, slow_cvxpy=False):
        element_list = []
        dim = output_constraint.range.shape[0]
        if not type(num_partitions).__module__ == np.__name__:
            num_partitions = np.array([num_partitions for i in range(dim)])
            # import pdb; pdb.set_trace()
        else:
            num_partitions_ = num_partitions
            # num_partitions_ = np.array([3, 3, 3, 1, 1, 1])
            num_partitions_ = np.array([24,24])
        input_shape = output_constraint.range.shape[:-1]
        slope = np.divide(
            (output_constraint.range[..., 1] - output_constraint.range[..., 0]), num_partitions_
        )
        
        for el in product(*[range(int(num)) for num in num_partitions_.flatten()]):
        
            oc_ = np.array(el).reshape(input_shape)
            output_range_ = np.empty_like(output_constraint.range)
            output_range_[..., 0] = output_constraint.range[..., 0] + np.multiply(
                oc_, slope
            )
            output_range_[..., 1] = output_constraint.range[..., 0] + np.multiply(
                oc_ + 1, slope
            )
            oc = constraints.LpConstraint(range=output_range_)

            element_list.append(oc)
        
        input_constraints_ = []
        info_ = {'per_timestep': []}
        
        for oc in [output_constraint]: # element_list:
            part_input_constraints = []
            part_info = []

            input_constraint, this_info = self.get_one_step_backprojection_set(
                oc, input_constraint, num_partitions=num_partitions, overapprox=overapprox, collected_input_constraints=[oc]+deepcopy(part_input_constraints), refined=refined, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy
            )
            part_input_constraints.append(deepcopy(input_constraint))
            part_info.append(deepcopy(this_info))
            # info_['per_timestep'].append(this_info)

            if overapprox:
                for i in np.arange(0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt):
                    next_output_constraint = over_approximate_constraint(deepcopy(input_constraint))
                    next_input_constraint = deepcopy(next_output_constraint)
                    input_constraint, this_info = self.get_one_step_backprojection_set(
                        next_output_constraint, next_input_constraint, num_partitions=num_partitions, overapprox=overapprox, collected_input_constraints=[oc]+deepcopy(part_input_constraints), infos=part_info, refined= refined, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy
                    )
                    part_input_constraints.append(deepcopy(input_constraint))
                    part_info.append(deepcopy(this_info))
                    # info_['per_timestep'].append(this_info)
                    # print(oc.range)
                    # print(input_constraint.range)
            else:
                for i in np.arange(0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt):
                    # TODO: Support N-step backprojection in the under-approximation case
                    raise NotImplementedError
            
            input_constraints_.append(deepcopy(part_input_constraints))
            info_['per_timestep'].append(deepcopy(part_info))

        # output_constraint: describes goal/avoid set at t=t_max
        # input_constraints: [BP_{-1}, ..., BP_{-t_max}]
        #       i.e., [ set of states that will get to goal/avoid set in 1 step,
        #               ...,
        #               set of states that will get to goal/avoid set in t_max steps
        #             ]

        input_constraints = [constraints.LpConstraint() for i in range(int(t_max))]
        info = {'per_timestep': []}

        

        for i, partition in enumerate(input_constraints_):
            for j, timestep in enumerate(partition):
                if input_constraints[j].range is None:
                    input_constraints[j].range = timestep.range
                else:
                    min_range = np.array([np.minimum(input_constraints[j].range[:,0],timestep.range[:,0])]).T
                    max_range = np.array([np.maximum(input_constraints[j].range[:,1],timestep.range[:,1])]).T
                    input_constraints[j].range = np.hstack((min_range, max_range))
                    


        for i in range(int(t_max)):
            info['per_timestep'].append([])

        for i, partition in enumerate(info_['per_timestep']):
            partition_info = info_['per_timestep'][i]
            for j, timestep in enumerate(partition_info):
                info['per_timestep'][j].append(timestep)


        import pdb; pdb.set_trace()

        return input_constraints, info

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


def over_approximate_constraint(constraint):
    # Note: this is a super sketchy implementation that only works under certain cases
    # specifically when all the contraints have the same A matrix

    # TODO: Add an assert
    # TODO: implement a more general version

    # constraint.A = constraint.A[0]
    # constraint.b = [np.max(np.array(constraint.b), axis=0)]

    return constraint
