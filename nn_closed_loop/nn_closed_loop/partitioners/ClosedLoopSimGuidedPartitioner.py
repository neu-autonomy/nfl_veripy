from .ClosedLoopPartitioner import ClosedLoopPartitioner
import nn_closed_loop.constraints as constraints
import numpy as np
from copy import deepcopy
import time
from nn_partition.utils.utils import sect


class ClosedLoopSimGuidedPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=16, make_animation=False, show_animation=False):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics, make_animation=make_animation, show_animation=show_animation)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"

        self.termination_condition_type = "num_propagator_calls"
        self.termination_condition_value = 200

        self.reachable_set_color = 'tab:blue'
        self.reachable_set_zorder = 2
        self.initial_set_color = 'tab:red'
        self.initial_set_zorder = 2
        self.sample_zorder = 1

    def check_termination(
        self,
        input_constraint,
        num_propagator_calls,
        u_e,
        output_range_sim,
        M,
        elapsed_time,
    ):
        if self.termination_condition_type == "input_cell_size":
            raise NotImplementedError
        elif self.termination_condition_type == "num_propagator_calls":
            terminate = (
                num_propagator_calls >= self.termination_condition_value
            )
        elif self.termination_condition_type == "pct_improvement":
            raise NotImplementedError
        elif self.termination_condition_type == "pct_error":
            raise NotImplementedError
        elif self.termination_condition_type == "verify":
            raise NotImplementedError
        elif self.termination_condition_type == "time_budget":
            raise NotImplementedError
            # terminate = elapsed_time >= self.termination_condition_value
        else:
            raise NotImplementedError
        return terminate

    def get_one_step_reachable_set(
        self,
        initial_set,
        propagator,
        num_partitions=None,
    ):
        reachable_set, info = self.get_reachable_set(
            initial_set,
            propagator,
            t_max=1,
            num_partitions=num_partitions,
        )
        return reachable_set, info

    def get_reachable_set(
        self,
        initial_set,
        propagator,
        t_max,
        num_partitions=None,
    ):

        t_start_overall = time.time()
        info = {}
        propagator_computation_time = 0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = "max"

        num_propagator_calls = 0
        interior_M = []

        # Run N simulations (i.e., randomly sample N pts from input range -->
        # query NN --> get N output pts)
        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        # (Line 5-6)
        reachable_set_range_sim = self.get_sampled_out_range_guidance(
            initial_set, propagator, t_max, num_samples=1000
        )

        # Get initial output reachable set (Line 3)
        t_start = time.time()

        reachable_set, info = propagator.get_reachable_set(
            initial_set, t_max
        )
        t_end = time.time()
        propagator_computation_time += t_end - t_start
        num_propagator_calls += t_max

        M = [(initial_set, reachable_set)]  # (Line 4)

        if self.make_animation:
            self.setup_visualization(
                initial_set,
                reachable_set,
                propagator,
                show_samples=True,
                inputs_to_highlight=[
                    {"dim": [0], "name": "$x_0$"},
                    {"dim": [1], "name": "$x_1$"},
                ],
                aspect="auto",
                initial_set_color=self.initial_set_color,
                initial_set_zorder=self.initial_set_zorder,
                sample_zorder=self.sample_zorder
            )

        reachable_set_range, info = self.partition_loop(
            M,
            interior_M,
            reachable_set_range_sim,
            sect_method,
            num_propagator_calls,
            initial_set,
            reachable_set.range,
            propagator,
            propagator_computation_time,
            t_start_overall,
            t_max,
        )

        # info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        reachable_set = constraints.LpConstraint(range=reachable_set_range)

        return reachable_set, info

    def partition_loop(
        self,
        M,
        interior_M,
        reachable_set_range_sim,
        sect_method,
        num_propagator_calls,
        initial_set,
        u_e,
        propagator,
        propagator_computation_time,
        t_start_overall,
        t_max,
    ):
        if self.make_animation:
            self.call_visualizer(reachable_set_range_sim, M+interior_M, num_propagator_calls, interior_M, iteration=-1)

        # Used by UnGuided, SimGuided, GreedySimGuided, etc.
        iteration = 0
        terminate = False
        start_time_partition_loop = t_start_overall
        while len(M) != 0 and not terminate:
            initial_set_this_cell, reachable_set_this_cell = self.grab_from_M(M, reachable_set_range_sim)  # (Line 9)

            if self.check_if_partition_within_sim_bnds(
                reachable_set_this_cell, reachable_set_range_sim
            ):
                # Line 11
                interior_M.append((initial_set_this_cell, reachable_set_this_cell))
            else:
                # Line 14
                elapsed_time = time.time() - start_time_partition_loop
                terminate = self.check_termination(
                    initial_set,
                    num_propagator_calls,
                    u_e,
                    reachable_set_range_sim,
                    M + [(initial_set_this_cell, reachable_set_this_cell)] + interior_M,
                    elapsed_time,
                )

                if not terminate:
                    # Line 15
                    sected_initial_set_ranges = sect(initial_set_this_cell.range, 2, select=sect_method)
                    # Lines 16-17
                    for sected_initial_set_range in sected_initial_set_ranges:
                        t_start = time.time()

                        initial_set_this_sected_cell = constraints.LpConstraint(range=sected_initial_set_range)
                        reachable_set_this_sected_cell, info = propagator.get_reachable_set(
                            initial_set_this_sected_cell, t_max
                        )
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += t_max

                        M.append((initial_set_this_sected_cell, reachable_set_this_sected_cell))  # Line 18

                else:  # Lines 19-20
                    M.append((initial_set_this_cell, reachable_set_this_cell))

                if self.make_animation:
                    self.call_visualizer(reachable_set_range_sim, M+interior_M, num_propagator_calls, interior_M, iteration=iteration, dont_tighten_layout=False)
            iteration += 1

        # Line 24
        u_e = self.squash_down_to_one_range(reachable_set_range_sim, M+interior_M)
        # u_e = self.squash_down_to_one_range(reachable_set_range_sim, M)
        t_end_overall = time.time()

        ranges = []
        for m in M+interior_M:
            ranges.append((m[0].range, m[1].range))
        info["all_partitions"] = ranges

        # Stats & Visualization
        # info = self.compile_info(
        #     reachable_set_range_sim,
        #     M,
        #     interior_M,
        #     num_propagator_calls,
        #     t_end_overall,
        #     t_start_overall,
        #     propagator_computation_time,
        #     iteration,
        # )
        if self.make_animation:
            self.compile_animation(iteration, delete_files=False, start_iteration=-1)

        return u_e, info

    def call_visualizer(self, output_range_sim, M, num_propagator_calls, interior_M, iteration, dont_tighten_layout=False):
        u_e = self.squash_down_to_one_range(output_range_sim, M)
        # title = "# Partitions: {}, Error: {}".format(str(len(M)+len(interior_M)), str(round(error, 3)))
        title = "# Propagator Calls: {}".format(
            str(int(num_propagator_calls))
        )
        # title = None

        output_constraint = constraints.LpConstraint(range=u_e)
        self.visualize(M, interior_M, output_constraint, iteration=iteration, title=title, reachable_set_color=self.reachable_set_color, reachable_set_zorder=self.reachable_set_zorder, dont_tighten_layout=dont_tighten_layout)

    def squash_down_to_one_range(self, output_range_sim, M):

        # (len(M)+1, t_max, n_states, 2)
        tmp = np.vstack([np.array([m[1].range for m in M]), np.expand_dims(output_range_sim, axis=0)])
        mins = np.min(tmp[...,0], axis=0)
        maxs = np.max(tmp[...,1], axis=0)
        return np.stack([mins, maxs], axis=2)

    def grab_from_M(self, M, output_range_sim=None):
        return M.pop(0)

    def check_if_partition_within_sim_bnds(
        self, reachable_set, reachable_set_range_sim
    ):
        reachable_set_range = reachable_set.range

        # Check if reachable_set_range's linf ball is within
        # reachable_set_range_sim's linf ball *for all timesteps*
        inside = np.all(
            (reachable_set_range_sim[..., 0] - reachable_set_range[..., 0]) <= 0
        ) and np.all((reachable_set_range_sim[..., 1] - reachable_set_range[..., 1]) >= 0)
        return inside

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

            if backprojection_set_this_cell is None:
                continue
            else:
                if overapprox:
                    backprojection_set.range[:, 0] = np.minimum(backprojection_set_this_cell.range[:, 0], backprojection_set.range[:, 0])
                    backprojection_set.range[:, 1] = np.maximum(backprojection_set_this_cell.range[:, 1], backprojection_set.range[:, 1])
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
                self.dynamics.sensor_noise
            )

        return backprojection_set, info

