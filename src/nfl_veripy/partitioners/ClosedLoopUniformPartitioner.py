import ast
from itertools import product
from typing import Any, Optional

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

    @property
    def num_partitions(self):
        return self._num_partitions

    @num_partitions.setter
    def num_partitions(self, value):
        if type(value) == np.ndarray:
            self._num_partitions = value
        elif type(value) == str:
            self._num_partitions = np.array(ast.literal_eval(value))

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

        info = {}  # type: dict[str, Any]
        num_propagator_calls = 0

        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), self.num_partitions
        )

        ranges = []

        for element in product(
            *[range(num) for num in self.num_partitions.flatten()]
        ):
            element_this_cell = np.array(element)
            input_range_this_cell = np.empty_like(input_range)
            input_range_this_cell[..., 0] = input_range[..., 0] + np.multiply(
                element_this_cell, slope
            )
            input_range_this_cell[..., 1] = input_range[..., 0] + np.multiply(
                element_this_cell + 1, slope
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
        assert backreachable_set.range is not None
        input_range = backreachable_set.range
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), self.num_partitions
        )

        # Iterate through each partition
        for element in product(
            *[range(int(num)) for num in self.num_partitions.flatten()]
        ):
            # Compute this partition's min/max xt values
            element_this_cell = np.array(element)
            ranges = np.empty_like(input_range)
            ranges[:, 0] = input_range[:, 0] + np.multiply(
                element_this_cell, slope
            )
            ranges[:, 1] = input_range[:, 0] + np.multiply(
                element_this_cell + 1, slope
            )
            backreachable_set_this_cell = constraints.LpConstraint(
                range=ranges, p=np.inf
            )

            backprojection_set_this_cell, this_info = (
                propagator.get_one_step_backprojection_set(
                    backreachable_set_this_cell,
                    target_sets,
                    overapprox=overapprox,
                )
            )
            backprojection_set.add_cell(
                backprojection_set_this_cell  # type: ignore
            )

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
