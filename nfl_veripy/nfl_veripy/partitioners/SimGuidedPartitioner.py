from .Partitioner import Partitioner
import numpy as np
import time


class SimGuidedPartitioner(Partitioner):
    def __init__(
        self,
        num_simulations=1000,
        termination_condition_type="input_cell_size",
        termination_condition_value=0.1,
        interior_condition="linf",
        make_animation=False,
        show_animation=False,
        show_input=True,
        show_output=True,
        adaptive_flag=False,
    ):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation
        self.show_input = show_input
        self.show_output = show_output
        self.adaptive_flag = adaptive_flag

    def grab_from_M(self, M, output_range_sim):
        input_range_, output_range_ = M.pop(0)
        return input_range_, output_range_

    def check_if_partition_within_sim_bnds(
        self, output_range, output_range_sim
    ):
        # Check if output_range's linf ball is within
        # output_range_sim's linf ball
        inside = np.all(
            (output_range_sim[..., 0] - output_range[..., 0]) <= 0
        ) and np.all((output_range_sim[..., 1] - output_range[..., 1]) >= 0)
        return inside

    def get_output_range(self, input_range, propagator):
        t_start_overall = time.time()
        propagator_computation_time = 0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = "max"

        num_propagator_calls = 0
        interior_M = []

        # Run N simulations (i.e., randomly sample N pts from input range -->
        # query NN --> get N output pts)
        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        # (Line 5-6)
        output_range_sim, sampled_outputs, sampled_inputs = self.sample(
            input_range, propagator
        )

        if self.adaptive_flag:
            (
                M,
                M_e,
                num_propagator_calls,
                propagator_computation_time,
            ) = self.expand_partition(
                propagator,
                sampled_inputs,
                sampled_outputs,
                output_range_sim,
                input_range,
                num_propagator_calls,
                propagator_computation_time,
            )
            u_e = output_range_sim

            # only for animation... (shouldn't count against algorithm)
            output_range, _ = propagator.get_output_range(input_range)

        else:

            # Get initial output reachable set (Line 3)
            t_start = time.time()
            output_range, _ = propagator.get_output_range(input_range)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1
            M = [(input_range, output_range)]  # (Line 4)

            u_e = output_range.copy()

        if self.termination_condition_type == "verify":
            raise NotImplementedError
            print(
                np.matmul(
                    self.termination_condition_value[0], sampled_outputs.T
                )
            )
            violated = np.all(
                np.matmul(
                    self.termination_condition_value[0], sampled_outputs.T
                )
                > self.termination_condition_value[1]
            )
            if violated:
                return "UNSAT"

        if self.make_animation:
            self.setup_visualization(
                input_range,
                output_range,
                propagator,
                show_input=self.show_input,
                show_output=self.show_output,
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
        )

        return u_e, info
