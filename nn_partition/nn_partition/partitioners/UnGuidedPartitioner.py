from .Partitioner import Partitioner
import numpy as np
import imageio
import os
import time


class UnGuidedPartitioner(Partitioner):
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

    def grab_from_M(self, M, output_range_sim):
        input_range_, output_range_ = M.pop(0)
        return input_range_, output_range_

    def compile_animation(self, iteration):
        animation_save_dir = "{}/results/tmp/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        filenames = [
            animation_save_dir + "tmp_{}.png".format(str(i).zfill(6))
            for i in range(iteration)
        ]
        images = []
        for filename in filenames:
            try:
                image = imageio.imread(filename)
            except:
                continue
            images.append(imageio.imread(filename))
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            os.remove(filename)

        # Save the gif in a new animations sub-folder
        animation_filename = "tmp.gif"
        animation_save_dir = "{}/results/animations/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        os.makedirs(animation_save_dir, exist_ok=True)
        animation_filename = animation_save_dir + animation_filename
        imageio.mimsave(animation_filename, images)

    def get_output_range(self, input_range, propagator, verbose=False):
        t_start_overall = time.time()
        propagator_computation_time = 0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = "max"
        input_shape = input_range.shape[:-1]
        info = {}

        num_propagator_calls = 0

        # Get initial output reachable set (Line 3)
        t_start = time.time()
        output_range, _ = propagator.get_output_range(
            input_range, verbose=verbose
        )
        t_end = time.time()
        propagator_computation_time += t_end - t_start
        num_propagator_calls += 1

        M = [(input_range, output_range)]  # (Line 4)
        interior_M = []

        u_e = output_range.copy()

        output_range_sim_for_evaluation, sampled_outputs = self.sample(
            input_range, propagator
        )

        # output_range_sim is [-inf, inf] per dimension -- just a dummy
        output_range_sim = np.empty_like(output_range)
        output_range_sim[:, 1] = -np.inf
        output_range_sim[:, 0] = np.inf

        if self.make_animation:
            self.setup_visualization(
                input_range,
                output_range,
                propagator,
                show_input=self.show_input,
                show_output=self.show_output,
            )

        print(
            "Need to pass output_range_sim_for_evaluation to partition_loop somehow..."
        )
        raise NotImplementedError
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

    def check_if_partition_within_sim_bnds(
        self, output_range, output_range_sim
    ):
        return False
