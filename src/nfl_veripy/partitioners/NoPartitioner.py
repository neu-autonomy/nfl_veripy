from .Partitioner import Partitioner
import numpy as np
import time


class NoPartitioner(Partitioner):
    def __init__(
        self,
        num_simulations=1000,
        termination_condition_type="input_cell_size",
        termination_condition_value=0.1,
        interior_condition="linf",
        make_animation=False,
        show_animation=False,
    ):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation

    def get_output_range(self, input_range, propagator):
        input_shape = input_range.shape[:-1]
        iteration = -1
        sampled_inputs = np.random.uniform(
            input_range[..., 0],
            input_range[..., 1],
            (int(self.num_simulations),) + input_shape,
        )
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull

            self.sim_convex_hull = ConvexHull(sampled_outputs)

        output_range_sim = np.empty(sampled_outputs.shape[1:] + (2,))
        output_range_sim[:, 1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:, 0] = np.min(sampled_outputs, axis=0)

        t_start = time.time()
        output_range, info = propagator.get_output_range(input_range)
        t_end = time.time()

        propagator_computation_time = t_end - t_start
        num_propagator_calls = 1
        info = self.compile_info(
            output_range_sim,
            [(input_range, output_range)],
            [],
            num_propagator_calls,
            t_end,
            t_start,
            propagator_computation_time,
            iteration,
        )

        return output_range, info
