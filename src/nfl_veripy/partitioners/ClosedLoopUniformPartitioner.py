from itertools import product
from typing import Optional

import numpy as np

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.utils import get_crown_matrices

from .ClosedLoopPartitioner import ClosedLoopPartitioner


class ClosedLoopUniformPartitioner(ClosedLoopPartitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        super().__init__(dynamics=dynamics)
        self.num_partitions: np.ndarray = np.array([4, 4])
        self.interior_condition: str = "linf"

    # TODO: set num_partitions attr properly using
    # np.array(ast.literal_eval(num_partitions))

    def get_one_step_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        num_partitions: Optional[np.ndarray] = None,
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        reachable_set, info = self.get_reachable_set(
            initial_set,
            propagator,
            t_max=1,
        )
        return reachable_set.get_constraint_at_time_index(0), info

    def get_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        reachable_sets = constraints.create_empty_multi_timestep_constraint(
            propagator.boundary_type, num_facets=propagator.num_polytope_facets
        )

        input_range = initial_set.to_range()

        info = {}
        num_propagator_calls = 0

        input_shape = input_range.shape[:-1]

        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), self.num_partitions
        )

        ranges = []

        for element in product(
            *[range(num) for num in self.num_partitions.flatten()]
        ):
            element_ = np.array(element).reshape(input_shape)
            input_range_this_cell = np.empty_like(input_range)
            input_range_this_cell[..., 0] = input_range[..., 0] + np.multiply(
                element_, slope
            )
            input_range_this_cell[..., 1] = input_range[..., 0] + np.multiply(
                element_ + 1, slope
            )

            initial_set_this_cell = initial_set.get_cell(input_range_this_cell)

            reachable_sets_this_cell, info_this_cell = (
                propagator.get_reachable_set(initial_set_this_cell, t_max)
            )
            num_propagator_calls += t_max
            info["nn_matrices"] = info_this_cell

            reachable_sets.add_cell(reachable_sets_this_cell)
            ranges.append((input_range_this_cell, reachable_sets_this_cell))

        reachable_sets.update_main_constraint_with_cells(overapprox=True)

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(self.num_partitions)

        return reachable_sets, info

    def get_one_step_backprojection_set(
        self,
        target_sets: constraints.MultiTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        overapprox: bool = False,
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        backreachable_set, info = self.get_one_step_backreachable_set(
            target_sets.get_constraint_at_time_index(-1)
        )
        info["backreachable_set"] = backreachable_set

        backprojection_set = constraints.create_empty_constraint(
            boundary_type=propagator.boundary_type,
            num_facets=propagator.num_polytope_facets,
        )

        """
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to backprojection_set
        """

        # Setup the partitions
        input_range = backreachable_set.range
        input_shape = input_range.shape[:-1]
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), self.num_partitions
        )

        # Iterate through each partition
        for element in product(
            *[range(int(num)) for num in self.num_partitions.flatten()]
        ):
            # Compute this partition's min/max xt values
            element = np.array(element).reshape(input_shape)
            backreachable_set_this_cell = constraints.LpConstraint(
                range=np.empty_like(input_range), p=np.inf
            )
            backreachable_set_this_cell.range[:, 0] = input_range[
                :, 0
            ] + np.multiply(element, slope)
            backreachable_set_this_cell.range[:, 1] = input_range[
                :, 0
            ] + np.multiply(element + 1, slope)

            backprojection_set_this_cell, this_info = (
                propagator.get_one_step_backprojection_set(
                    backreachable_set_this_cell,
                    target_sets,
                    overapprox=overapprox,
                )
            )

            backprojection_set.add_cell(backprojection_set_this_cell)

        backprojection_set.update_main_constraint_with_cells(
            overapprox=overapprox
        )

        if overapprox:
            # These will be used to further backproject this set in time
            backprojection_set.crown_matrices = get_crown_matrices(
                propagator,
                backprojection_set,
                self.dynamics.num_inputs,
                self.dynamics.sensor_noise,
            )

        return backprojection_set, info
