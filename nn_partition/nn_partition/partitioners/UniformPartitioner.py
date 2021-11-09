from .Partitioner import Partitioner
import numpy as np
from itertools import product
import time


class UniformPartitioner(Partitioner):
    def __init__(
        self,
        num_simulations=1000,
        num_partitions=16,
        termination_condition_type="input_cell_size",
        termination_condition_value=0.1,
        interior_condition="linf",
        make_animation=False,
        show_animation=False,
        show_input=True,
        show_output=True,
    ):
        Partitioner.__init__(self)
        self.num_partitions = num_partitions
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation
        self.num_simulations = num_simulations

    def get_output_range(self, input_range, propagator, num_partitions=None):
        input_shape = input_range.shape[:-1]

        output_range_sim, sampled_outputs, sampled_inputs = self.sample(
            input_range, propagator
        )

        propagator_computation_time = 0

        t_start = time.time()

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
        output_range = None

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
            output_range_, info_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1

            if output_range is None:
                output_range = np.empty(output_range_.shape)
                output_range[:, 0] = np.inf
                output_range[:, 1] = -np.inf

            tmp = np.dstack([output_range, output_range_])
            output_range[:, 1] = np.max(tmp[:, 1, :], axis=1)
            output_range[:, 0] = np.min(tmp[:, 0, :], axis=1)

            ranges.append((input_range_, output_range_))
            t_end = time.time()

        info = self.compile_info(
            output_range_sim,
            ranges,
            [],
            num_propagator_calls,
            t_end,
            t_start,
            propagator_computation_time,
            0,
        )

        return output_range, info
