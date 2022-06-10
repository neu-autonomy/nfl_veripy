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

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # only used to compute slope in non-closedloop manner...
            input_polytope_verts = pypoman.duality.compute_polytope_vertices(
                A_inputs, b_inputs
            )
            input_range = np.empty((A_inputs.shape[1], 2))
            input_range[:, 0] = np.min(np.stack(input_polytope_verts), axis=0)
            input_range[:, 1] = np.max(np.stack(input_polytope_verts), axis=0)

        elif isinstance(input_constraint, constraints.LpConstraint):
            input_range = input_constraint.range
        else:
            raise NotImplementedError

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
            input_range_ = np.empty_like(input_range)
            input_range_[..., 0] = input_range[..., 0] + np.multiply(
                element_, slope
            )
            input_range_[..., 1] = input_range[..., 0] + np.multiply(
                element_ + 1, slope
            )

            if isinstance(
                input_constraint, constraints.PolytopeConstraint
            ):
                # This is a disaster hack to partition polytopes
                A_rect, b_rect = range_to_polytope(input_range_)
                rectangle_verts = pypoman.polygon.compute_polygon_hull(
                    A_rect, b_rect
                )
                input_polytope_verts = pypoman.polygon.compute_polygon_hull(
                    A_inputs, b_inputs
                )
                partition_verts = pypoman.intersection.intersect_polygons(
                    input_polytope_verts, rectangle_verts
                )
                (
                    A_inputs_,
                    b_inputs_,
                ) = pypoman.duality.compute_polytope_halfspaces(
                    partition_verts
                )
                input_constraint_ = input_constraint.__class__(
                    A_inputs_, b_inputs_
                )
            elif isinstance(input_constraint, constraints.LpConstraint):
                input_constraint_ = input_constraint.__class__(
                    range=input_range_, p=input_constraint.p
                )
            else:
                raise NotImplementedError

            output_constraint_, info_ = propagator.get_reachable_set(
                input_constraint_, deepcopy(output_constraint), t_max
            )
            num_propagator_calls += t_max
            info['nn_matrices'] = info_

            if isinstance(
                output_constraint, constraints.PolytopeConstraint
            ):
                reachable_set_ = [o.b for o in output_constraint_]
                if output_constraint.b is None:
                    output_constraint.b = np.stack(reachable_set_)

                tmp = np.dstack(
                    [output_constraint.b, np.stack(reachable_set_)]
                )
                output_constraint.b = np.max(tmp, axis=-1)

                ranges.append((input_range_, reachable_set_))
            elif isinstance(output_constraint, constraints.LpConstraint):
                reachable_set_ = [o.range for o in output_constraint_]
                if output_constraint.range is None:
                    output_constraint.range = np.stack(reachable_set_)

                tmp = np.stack(
                    [output_constraint.range, np.stack(reachable_set_)],
                    axis=-1,
                )

                output_constraint.range[..., 0] = np.min(
                    tmp[..., 0, :], axis=-1
                )
                output_constraint.range[..., 1] = np.max(
                    tmp[..., 1, :], axis=-1
                )
                ranges.append((input_range_, np.stack(reachable_set_)))
            else:
                raise NotImplementedError

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        return output_constraint, info

    def get_one_step_backprojection_set(
        self, target_sets, propagator, num_partitions=None, overapprox=False
    ):

        backreachable_set, info = self.get_one_step_backreachable_set(target_sets[-1])

        '''
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to input_constraint
        '''

        # Setup the partitions
        if num_partitions is None:
            num_partitions = np.array([10, 10])

        input_range = backreachable_set.range
        input_shape = input_range.shape[:-1]
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), num_partitions
        )

        # Set an empty Constraint that will get filled in
        xt_max = -np.inf*np.ones(propagator.dynamics.num_states)
        xt_min = np.inf*np.ones(propagator.dynamics.num_states)

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

            if backprojection_set_this_cell is None:
                continue
            else:
                xt_min = np.minimum(backprojection_set_this_cell.range[:, 0], xt_min)
                xt_max = np.maximum(backprojection_set_this_cell.range[:, 1], xt_max)

            # TODO: Store the detailed partitions in info

        if overapprox:

            backprojection_set = constraints.LpConstraint(
                range=np.vstack((xt_min, xt_max)).T,
                p=np.inf
            )

            # These will be used to further backproject this set in time
            backprojection_set.crown_matrices = get_crown_matrices(
                propagator,
                backprojection_set,
                self.dynamics.num_inputs,
                self.dynamics.sensor_noise
            )

        return backprojection_set, info
