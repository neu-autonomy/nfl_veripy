from .ClosedLoopPartitioner import ClosedLoopPartitioner
import nn_closed_loop.constraints as constraints
import numpy as np
import pypoman
from itertools import product
from copy import deepcopy
from nn_closed_loop.utils.utils import range_to_polytope
import time
from nn_partition.utils.utils import sect


class ClosedLoopSimGuidedPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=16):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

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
            # #  print(input_range_[...,1] - input_range_[...,0])
            # M_numpy = np.dstack([input_range for (input_range, _) in M])

            # terminate = (
            #     np.min(M_numpy[:, 1] - M_numpy[:, 0])
            #     <= self.termination_condition_value
            # )
        elif self.termination_condition_type == "num_propagator_calls":
            terminate = (
                num_propagator_calls >= self.termination_condition_value
            )
        elif self.termination_condition_type == "pct_improvement":
            raise NotImplementedError
            # # This doesnt work very well, because a lot of times
            # # the one-step improvement is zero
            # last_u_e = u_e.copy()
            # if self.interior_condition in ["lower_bnds", "linf"]:
            #     u_e = self.squash_down_to_one_range(output_range_sim, M)
            #     improvement = self.get_error(last_u_e, u_e)
            #     if iteration == 0:
            #         improvement = np.inf
            # elif self.interior_condition == "convex_hull":
            #     # raise NotImplementedError
            #     last_hull = estimated_hull.copy()

            #     estimated_hull = self.squash_down_to_convex_hull(
            #         M, self.sim_convex_hull.points
            #     )
            #     improvement = self.get_error(last_hull, estimated_hull)
            # terminate = improvement <= self.termination_condition_value
        elif self.termination_condition_type == "pct_error":
            raise NotImplementedError
        #     if self.interior_condition in ["lower_bnds", "linf"]:
        #         u_e = self.squash_down_to_one_range(output_range_sim, M)
        #         error = self.get_error(output_range_sim, u_e)
        #     elif self.interior_condition == "convex_hull":
        #         estimated_hull = self.squash_down_to_convex_hull(
        #             M, self.sim_convex_hull.points
        #         )
        #         error = self.get_error(self.sim_convex_hull, estimated_hull)
        #     terminate = error <= self.termination_condition_value
        # #   print(error)
        elif self.termination_condition_type == "verify":
            raise NotImplementedError
            # M_ = M + [(input_range_, output_range_)]
            # ndim = M_[0][1].shape[0]
            # pts = np.empty((len(M_) * (2 ** (ndim)), ndim))
            # i = 0
            # for (input_range, output_range) in M:
            #     for pt in product(*output_range):
            #         pts[i, :] = pt
            #         i += 1
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
        # if self.make_animation:
        #     self.call_visualizer(output_range_sim, M, num_propagator_calls, interior_M, iteration=-1)

        # Used by UnGuided, SimGuided, GreedySimGuided, etc.
        iteration = 0
        terminate = False
        start_time_partition_loop = t_start_overall
        while len(M) != 0 and not terminate:
            # print('------')
            # print("Iteration {}".format(iteration))
            input_constraint_, reachable_set_ = self.grab_from_M(M, output_range_sim)  # (Line 9)
            # print("Grabbed the following from M:")
            # print("input_constraint_.range: {}".format(input_constraint_.range))
            # print("reachable_set_: {}".format(reachable_set_))
            # print('---')

            if self.check_if_partition_within_sim_bnds(
                reachable_set_, output_range_sim
            ):
                print("Was within sim bounds. Add to interior_M.")
                # Line 11
                interior_M.append((input_constraint_, reachable_set_))
            else:
                # print("Was not within sim bounds. Check for termination or sect.")
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
                    # print("Don't terminate. Sect.")
                    # Line 15
                    input_ranges_ = sect(input_constraint_.range, 2, select=sect_method)
                    # print("input_ranges_: {}".format(input_ranges_))
                    # Lines 16-17
                    # print("Looping through...")
                    for input_range_ in input_ranges_:
                        # print("input_range_: {}".format(input_range_))
                        t_start = time.time()

                        input_constraint_ = constraints.LpInputConstraint(range=input_range_)
                        output_constraint_, info = propagator.get_reachable_set(
                            input_constraint_, deepcopy(output_constraint), t_max
                        )
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += t_max

                        reachable_set_ = [o.range for o in output_constraint_]
                        # print("reachable_set_: {}".format(reachable_set_))
                        M.append((input_constraint_, reachable_set_))  # Line 18

                else:  # Lines 19-20
                    # print("Terminate.")
                    M.append((input_constraint_, reachable_set_))

                # print("M: {}".format(M))
                # if self.make_animation:
                #     self.call_visualizer(output_range_sim, M, num_propagator_calls, interior_M, iteration=iteration)
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
        # if self.make_animation:
        #     self.compile_animation(iteration)

        return u_e, info

    def squash_down_to_one_range(self, output_range_sim, M):

        # (len(M)+1, t_max, n_states, 2)
        tmp = np.vstack([np.array([m[-1] for m in M]), np.expand_dims(output_range_sim, axis=0)])
        mins = np.min(tmp[...,0], axis=0)
        maxs = np.max(tmp[...,1], axis=0)
        return np.stack([mins, maxs], axis=2)

        # tmp = np.stack(
        #     [output_constraint.range, np.stack(reachable_set_)],
        #     axis=-1,
        # )

        # output_constraint.range[..., 0] = np.min(
        #     tmp[..., 0, :], axis=-1
        # )
        # output_constraint.range[..., 1] = np.max(
        #     tmp[..., 1, :], axis=-1
        # )
        # ranges.append((input_range_, np.stack(reachable_set_)))


        # u_e = np.empty_like(output_range_sim)
        # if len(M) > 0:
        #     # Squash all of M down to one range
        #     M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
        #     u_e[:, 1] = np.max(M_numpy[:, 1, :], axis=1)
        #     u_e[:, 0] = np.min(M_numpy[:, 0, :], axis=1)

        #     # Combine M (remaining ranges) with u_e (interior ranges)
        #     tmp = np.dstack([output_range_sim, u_e])
        #     u_e[:, 1] = np.max(tmp[:, 1, :], axis=1)
        #     u_e[:, 0] = np.min(tmp[:, 0, :], axis=1)
        # return u_e

    def get_reachable_set(
        self,
        input_constraint,
        output_constraint,
        propagator,
        t_max,
        num_partitions=None,
    ):

        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
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

        elif isinstance(input_constraint, constraints.LpInputConstraint):
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
            output_constraint, constraints.PolytopeOutputConstraint
        ):
            raise NotImplementedError
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            reachable_set = [o.range for o in output_constraint_]
            M = [(input_constraint, reachable_set)]  # (Line 4)
        else:
            raise NotImplementedError

        u_e = reachable_set.copy()

        # if self.make_animation:
        #     self.setup_visualization(
        #         input_range,
        #         output_range,
        #         propagator,
        #         show_input=self.show_input,
        #         show_output=self.show_output,
        #     )

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
