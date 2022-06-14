from .ClosedLoopPartitioner import ClosedLoopPartitioner
import nn_closed_loop.constraints as constraints
import numpy as np
import pypoman
from itertools import product
from copy import deepcopy
from nn_closed_loop.utils.utils import range_to_polytope, get_crown_matrices


class ClosedLoopUniformPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=16, make_animation=False, show_animation=False):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics, make_animation=make_animation, show_animation=show_animation)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    def get_one_step_reachable_set(
        self,
        input_constraint,
        output_constraint,
        propagator,
        num_partitions=None,
    ):
        reachable_set, info = self.get_reachable_set(
            input_constraint,
            output_constraint,
            propagator,
            t_max=1,
            num_partitions=num_partitions,
        )
        return reachable_set, info

    def get_reachable_set(
        self,
        input_constraint,
        output_constraint,
        propagator,
        t_max,
        num_partitions=None,
    ):

        input_range = input_constraint.to_range()

        info = {}
        num_propagator_calls = 0

        input_shape = input_range.shape[:-1]

        if num_partitions is None:
            num_partitions = np.ones(input_shape, dtype=int)
            if (
                isinstance(self.num_partitions, np.ndarray)
                and input_shape == self.num_partitions.shape
            ):
                num_partitions = self.num_partitions
            elif len(input_shape) > 1:
                num_partitions[0, 0] = self.num_partitions
            else:
                num_partitions *= self.num_partitions
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), num_partitions
        )

        ranges = []

        for element in product(
            *[range(num) for num in num_partitions.flatten()]
        ):
            element_ = np.array(element).reshape(input_shape)
            input_range_this_cell = np.empty_like(input_range)
            input_range_this_cell[..., 0] = input_range[..., 0] + np.multiply(
                element_, slope
            )
            input_range_this_cell[..., 1] = input_range[..., 0] + np.multiply(
                element_ + 1, slope
            )

            input_constraint_this_cell = input_constraint.get_cell(input_range_this_cell)

            output_constraint_this_cell, info_this_cell = propagator.get_reachable_set(
                input_constraint_this_cell, deepcopy(output_constraint), t_max
            )
            num_propagator_calls += t_max
            info['nn_matrices'] = info_this_cell

            reachable_set_this_cell = output_constraint.add_cell(output_constraint_this_cell)
            ranges.append((input_range_this_cell, reachable_set_this_cell))

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        return output_constraint, info

    def get_one_step_backprojection_set(
        self, target_sets, propagator, num_partitions=None, overapprox=False
    ):

        backreachable_set, info = self.get_one_step_backreachable_set(target_sets[-1])
        info['backreachable_set'] = backreachable_set

        if overapprox:
            # Set an empty Constraint that will get filled in
            backprojection_set = constraints.LpConstraint(
                range=np.vstack(
                    (
                        np.inf*np.ones(propagator.dynamics.num_states),
                        -np.inf*np.ones(propagator.dynamics.num_states)
                    )
                ).T,
                p=np.inf
            )
        else:
            backprojection_set = constraints.PolytopeConstraint(A=[], b=[])

        '''
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to backprojection_set
        '''

        # Setup the partitions
        if num_partitions is None:
            num_partitions = np.array([10, 10])

        input_range = backreachable_set.range
        input_shape = input_range.shape[:-1]
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), num_partitions
        )

        # Iterate through each partition
        for element in product(
            *[range(int(num)) for num in num_partitions.flatten()]
        ):
            # Compute this partition's min/max xt values
            element = np.array(element).reshape(input_shape)
            backreachable_set_this_cell = constraints.LpConstraint(
                range=np.empty_like(input_range), p=np.inf
            )
            backreachable_set_this_cell.range[:, 0] = input_range[:, 0] + np.multiply(
                element, slope
            )
            backreachable_set_this_cell.range[:, 1] = input_range[:, 0] + np.multiply(
                element + 1, slope
            )

            backprojection_set_this_cell, this_info = propagator.get_one_step_backprojection_set(
                backreachable_set_this_cell,
                target_sets,
                overapprox=overapprox,
            )

            backprojection_set += backprojection_set_this_cell

            # TODO: Store the detailed partitions in info

        if overapprox:

            # These will be used to further backproject this set in time
            backprojection_set.crown_matrices = get_crown_matrices(
                propagator,
                backprojection_set,
                self.dynamics.num_inputs,
                self.dynamics.sensor_noise
            )

        return backprojection_set, info
