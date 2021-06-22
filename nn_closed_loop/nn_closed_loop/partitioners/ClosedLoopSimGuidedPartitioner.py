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
        self.termination_condition_value = 500

    def check_termination(
        self,
        input_range_,
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

    def grab_from_M(self, M, output_range_sim=None):
        return M.pop(0)

    def check_if_partition_within_sim_bnds(
        self, output_range, output_range_sim
    ):
        output_range_ = np.array(output_range)

        # Check if output_range's linf ball is within
        # output_range_sim's linf ball *for all timesteps*
        inside = np.all(
            (output_range_sim[..., 0] - output_range_[..., 0]) <= 0
        ) and np.all((output_range_sim[..., 1] - output_range_[..., 1]) >= 0)
        return inside

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

    def partition_loop(
        self,
        M,
        interior_M,
        output_range_sim,
        sect_method,
        num_propagator_calls,
        input_range,
        u_e,
        propagator,
        propagator_computation_time,
        t_start_overall,
        t_max,
        output_constraint
    ):
        if self.make_animation:
            self.call_visualizer(output_range_sim, M+interior_M, num_propagator_calls, interior_M, iteration=-1)

        # Used by UnGuided, SimGuided, GreedySimGuided, etc.
        iteration = 0
        terminate = False
        start_time_partition_loop = t_start_overall
        while len(M) != 0 and not terminate:
            input_constraint_, reachable_set_ = self.grab_from_M(M, output_range_sim)  # (Line 9)

            if self.check_if_partition_within_sim_bnds(
                reachable_set_, output_range_sim
            ):
                # Line 11
                interior_M.append((input_constraint_, reachable_set_))
            else:
                # Line 14
                elapsed_time = time.time() - start_time_partition_loop
                terminate = self.check_termination(
                    input_range,
                    num_propagator_calls,
                    u_e,
                    output_range_sim,
                    M + [(input_constraint_, reachable_set_)] + interior_M,
                    elapsed_time,
                )

                if not terminate:
                    # Line 15
                    input_ranges_ = sect(input_constraint_.range, 2, select=sect_method)
                    # Lines 16-17
                    for input_range_ in input_ranges_:
                        t_start = time.time()

                        input_constraint_ = constraints.LpConstraint(range=input_range_)
                        output_constraint_, info = propagator.get_reachable_set(
                            input_constraint_, deepcopy(output_constraint), t_max
                        )
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += t_max

                        reachable_set_ = [o.range for o in output_constraint_]
                        M.append((input_constraint_, reachable_set_))  # Line 18

                else:  # Lines 19-20
                    M.append((input_constraint_, reachable_set_))

                if self.make_animation:
                    self.call_visualizer(output_range_sim, M+interior_M, num_propagator_calls, interior_M, iteration=iteration)
            iteration += 1

        # Line 24
        u_e = self.squash_down_to_one_range(output_range_sim, M+interior_M)
        # u_e = self.squash_down_to_one_range(output_range_sim, M)
        t_end_overall = time.time()

        ranges = []
        for m in M+interior_M:
            ranges.append((m[0].range, np.stack(m[1])))
        info["all_partitions"] = ranges

        # Stats & Visualization
        # info = self.compile_info(
        #     output_range_sim,
        #     M,
        #     interior_M,
        #     num_propagator_calls,
        #     t_end_overall,
        #     t_start_overall,
        #     propagator_computation_time,
        #     iteration,
        # )
        if self.make_animation:
            self.compile_animation(iteration, delete_files=True)

        return u_e, info

    def call_visualizer(self, output_range_sim, M, num_propagator_calls, interior_M, iteration):
        u_e = self.squash_down_to_one_range(output_range_sim, M)
        # title = "# Partitions: {}, Error: {}".format(str(len(M)+len(interior_M)), str(round(error, 3)))
        title = "# Propagator Calls: {}".format(
            str(int(num_propagator_calls))
        )
        # title = None

        output_constraint = constraints.LpConstraint(range=u_e)
        self.visualize(M, interior_M, output_constraint, iteration=iteration, title=title)

    def squash_down_to_one_range(self, output_range_sim, M):

        # (len(M)+1, t_max, n_states, 2)
        tmp = np.vstack([np.array([m[-1] for m in M]), np.expand_dims(output_range_sim, axis=0)])
        mins = np.min(tmp[...,0], axis=0)
        maxs = np.max(tmp[...,1], axis=0)
        return np.stack([mins, maxs], axis=2)

    def get_reachable_set(
        self,
        input_constraint,
        output_constraint,
        propagator,
        t_max,
        num_partitions=None,
    ):

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            raise NotImplementedError
            # A_inputs = input_constraint.A
            # b_inputs = input_constraint.b

            # # only used to compute slope in non-closedloop manner...
            # input_polytope_verts = pypoman.duality.compute_polytope_vertices(
            #     A_inputs, b_inputs
            # )
            # input_range = np.empty((A_inputs.shape[1], 2))
            # input_range[:, 0] = np.min(np.stack(input_polytope_verts), axis=0)
            # input_range[:, 1] = np.max(np.stack(input_polytope_verts), axis=0)

        elif isinstance(input_constraint, constraints.LpConstraint):
            input_range = input_constraint.range
        else:
            raise NotImplementedError

        t_start_overall = time.time()
        info = {}
        # input_shape = input_range.shape[:-1]
        propagator_computation_time = 0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = "max"

        num_propagator_calls = 0
        interior_M = []

        # Run N simulations (i.e., randomly sample N pts from input range -->
        # query NN --> get N output pts)
        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        # (Line 5-6)
        output_range_sim = self.get_sampled_out_range(
            input_constraint, propagator, t_max, num_samples=1000
        )

        # Get initial output reachable set (Line 3)
        t_start = time.time()

        output_constraint_, info = propagator.get_reachable_set(
            input_constraint, deepcopy(output_constraint), t_max
        )
        t_end = time.time()
        propagator_computation_time += t_end - t_start
        num_propagator_calls += t_max

        if isinstance(
            output_constraint, constraints.PolytopeConstraint
        ):
            raise NotImplementedError
        elif isinstance(output_constraint, constraints.LpConstraint):
            reachable_set = [o.range for o in output_constraint_]
            M = [(input_constraint, reachable_set)]  # (Line 4)
        else:
            raise NotImplementedError

        u_e = reachable_set.copy()

        if self.make_animation:
            output_constraint_ = constraints.LpConstraint(range=[o.range for o in output_constraint_])
            self.setup_visualization(
                input_constraint,
                output_constraint_,
                propagator,
                show_samples=True,
                outputs_to_highlight=[
                    {"dim": [0], "name": "py"},
                    {"dim": [1], "name": "pz"},
                ],
                inputs_to_highlight=[
                    {"dim": [0], "name": "py"},
                    {"dim": [1], "name": "pz"},
                ],
                aspect="auto",
            )

        u_e, info = self.partition_loop(
            M,
            interior_M,
            output_range_sim,
            sect_method,
            num_propagator_calls,
            input_range,
            u_e,
            propagator,
            propagator_computation_time,
            t_start_overall,
            t_max,
            output_constraint,
        )

        # info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        output_constraint.range = u_e

        return output_constraint, info
