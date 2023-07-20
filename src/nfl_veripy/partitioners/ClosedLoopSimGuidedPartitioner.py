import time
from itertools import product
from typing import Any, Optional, Union

import numpy as np

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.utils import get_crown_matrices, sect

from .ClosedLoopPartitioner import ClosedLoopPartitioner


class ClosedLoopSimGuidedPartitioner(ClosedLoopPartitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        super().__init__(dynamics=dynamics)
        self.interior_condition: str = "linf"

        self.termination_condition_type: str = "num_propagator_calls"
        self.termination_condition_value: int = 200

    def check_termination(  # type: ignore
        self,
        num_propagator_calls: float,
    ) -> bool:
        if self.termination_condition_type == "num_propagator_calls":
            terminate = (
                num_propagator_calls >= self.termination_condition_value
            )
        else:
            raise NotImplementedError
        return terminate

    def get_one_step_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        num_partitions: Union[None, int, list, np.ndarray] = None,
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
        num_partitions: Union[None, int, list, np.ndarray] = None,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        t_start_overall = time.time()
        info = {}  # type: dict[str, Any]
        propagator_computation_time = 0.0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = "max"

        num_propagator_calls = 0
        interior_M = (
            []
        )  # type: list[tuple[constraints.SingleTimestepConstraint, constraints.MultiTimestepConstraint]] # noqa

        # Run N simulations (i.e., randomly sample N pts from input range -->
        # query NN --> get N output pts)
        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        # (Line 5-6)
        reachable_sets_range_sim = self.get_sampled_out_range_guidance(
            initial_set, propagator, t_max, num_samples=1000
        )

        # Get initial output reachable set (Line 3)
        t_start = time.time()

        reachable_sets, info = propagator.get_reachable_set(initial_set, t_max)
        t_end = time.time()
        propagator_computation_time += t_end - t_start
        num_propagator_calls += t_max

        M = [(initial_set, reachable_sets)]  # (Line 4)

        reachable_sets_range, info = self.partition_loop(
            M,
            interior_M,
            reachable_sets_range_sim,
            sect_method,
            num_propagator_calls,
            initial_set,
            reachable_sets.range,
            propagator,
            propagator_computation_time,
            t_start_overall,
            t_max,
            info,
        )

        # info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls

        reachable_sets = constraints.MultiTimestepConstraint(
            constraints=[
                constraints.LpConstraint(range=r) for r in reachable_sets_range
            ]
        )

        # store the cells in the constraints - only for plotting
        reachable_sets.cells = [x[1] for x in M + interior_M]

        return reachable_sets, info

    def partition_loop(
        self,
        M: list[
            tuple[
                constraints.SingleTimestepConstraint,
                constraints.MultiTimestepConstraint,
            ]
        ],
        interior_M: list[
            tuple[
                constraints.SingleTimestepConstraint,
                constraints.MultiTimestepConstraint,
            ]
        ],
        reachable_set_range_sim,
        sect_method: str,
        num_propagator_calls: float,
        initial_set: constraints.SingleTimestepConstraint,
        u_e,
        propagator: propagators.ClosedLoopPropagator,
        propagator_computation_time: float,
        t_start_overall: float,
        t_max: int,
        info: dict,
    ) -> tuple[np.ndarray, dict]:
        # Store these items at each iteration for eventual visualization
        info["iterations"] = {}
        info["iterations"][-1] = {
            "M": M,
            "interior_M": interior_M,
            "num_propagator_calls": num_propagator_calls,
        }

        # Used by UnGuided, SimGuided, GreedySimGuided, etc.
        iteration = 0
        terminate = False
        while len(M) != 0 and not terminate:
            initial_set_this_cell, reachable_set_this_cell = self.grab_from_M(
                M, reachable_set_range_sim
            )  # (Line 9)

            if self.check_if_partition_within_sim_bnds(
                reachable_set_this_cell, reachable_set_range_sim
            ):
                # Line 11
                interior_M.append(
                    (initial_set_this_cell, reachable_set_this_cell)
                )
            else:
                # Line 14
                terminate = self.check_termination(num_propagator_calls)

                if not terminate:
                    # Line 15
                    sected_initial_set_ranges = sect(
                        initial_set_this_cell.range, 2, select=sect_method
                    )
                    # Lines 16-17
                    for sected_initial_set_range in sected_initial_set_ranges:
                        t_start = time.time()

                        initial_set_this_sected_cell = (
                            constraints.LpConstraint(
                                range=sected_initial_set_range
                            )
                        )
                        reachable_set_this_sected_cell, this_info = (
                            propagator.get_reachable_set(
                                initial_set_this_sected_cell, t_max
                            )
                        )
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += t_max

                        M.append(
                            (
                                initial_set_this_sected_cell,
                                reachable_set_this_sected_cell,
                            )
                        )  # Line 18

                else:  # Lines 19-20
                    M.append((initial_set_this_cell, reachable_set_this_cell))

                info["iterations"][iteration] = {
                    "M": M,
                    "interior_M": interior_M,
                    "num_propagator_calls": num_propagator_calls,
                }
            iteration += 1

        # Line 24
        u_e = self.squash_down_to_one_range(
            reachable_set_range_sim, M + interior_M
        )
        # u_e = self.squash_down_to_one_range(reachable_set_range_sim, M)

        ranges = []
        for m in M + interior_M:
            ranges.append((m[0].range, m[1].range))
        info["all_partitions"] = ranges

        return u_e, info

    def squash_down_to_one_range(self, output_range_sim, M):
        # (len(M)+1, t_max, n_states, 2)
        tmp = np.vstack(
            [
                np.array([m[1].range for m in M]),
                np.expand_dims(output_range_sim, axis=0),
            ]
        )
        mins = np.min(tmp[..., 0], axis=0)
        maxs = np.max(tmp[..., 1], axis=0)
        return np.stack([mins, maxs], axis=2)

    def grab_from_M(
        self,
        M: list[
            tuple[
                constraints.SingleTimestepConstraint,
                constraints.MultiTimestepConstraint,
            ]
        ],
        output_range_sim: Optional[np.ndarray],
    ) -> tuple[
        constraints.SingleTimestepConstraint,
        constraints.MultiTimestepConstraint,
    ]:
        input_range, output_range = M.pop(0)
        return input_range, output_range

    def check_if_partition_within_sim_bnds(
        self, reachable_set, reachable_set_range_sim
    ):
        reachable_set_range = reachable_set.range

        # Check if reachable_set_range's linf ball is within
        # reachable_set_range_sim's linf ball *for all timesteps*
        inside = np.all(
            (reachable_set_range_sim[..., 0] - reachable_set_range[..., 0])
            <= 0
        ) and np.all(
            (reachable_set_range_sim[..., 1] - reachable_set_range[..., 1])
            >= 0
        )
        return inside

    def get_one_step_backprojection_set(
        self, target_sets, propagator, num_partitions=None, overapprox=False
    ):
        # Currently this seems to just copy-paste the uniform partitioning code
        raise NotImplementedError
        backreachable_set, info = self.get_one_step_backreachable_set(
            target_sets[-1]
        )
        info["backreachable_set"] = backreachable_set

        if overapprox:
            # Set an empty Constraint that will get filled in
            backprojection_set = constraints.LpConstraint(
                range=np.vstack(
                    (
                        np.inf * np.ones(propagator.dynamics.num_states),
                        -np.inf * np.ones(propagator.dynamics.num_states),
                    )
                ).T,
                p=np.inf,
            )
        else:
            backprojection_set = constraints.PolytopeConstraint(A=[], b=[])

        """
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to backprojection_set
        """

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

            if backprojection_set_this_cell is None:
                continue
            else:
                if overapprox:
                    backprojection_set.range[:, 0] = np.minimum(
                        backprojection_set_this_cell.range[:, 0],
                        backprojection_set.range[:, 0],
                    )
                    backprojection_set.range[:, 1] = np.maximum(
                        backprojection_set_this_cell.range[:, 1],
                        backprojection_set.range[:, 1],
                    )
                else:
                    backprojection_set.A.append(backprojection_set_this_cell.A)
                    backprojection_set.b.append(backprojection_set_this_cell.b)

            # TODO: Store the detailed partitions in info

        if overapprox:
            # These will be used to further backproject this set in time
            backprojection_set.crown_matrices = get_crown_matrices(
                propagator,
                backprojection_set,
                self.dynamics.num_inputs,
                self.dynamics.sensor_noise,
            )

        return backprojection_set, info
