from .GreedySimGuidedPartitioner import GreedySimGuidedPartitioner
import time
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
import numpy as np


class AdaptiveGreedySimGuidedPartitioner(GreedySimGuidedPartitioner):
    def __init__(
        self,
        num_simulations=1000,
        interior_condition="linf",
        make_animation=False,
        show_animation=False,
        termination_condition_type="interior_cell_size",
        termination_condition_value=0.02,
        show_input=True,
        show_output=True,
    ):
        GreedySimGuidedPartitioner.__init__(
            self,
            num_simulations=num_simulations,
            interior_condition=interior_condition,
            make_animation=make_animation,
            show_animation=show_animation,
            termination_condition_type=termination_condition_type,
            termination_condition_value=termination_condition_value,
            show_input=show_input,
            show_output=show_output,
            adaptive_flag=True,
        )

    def set_delta_step(self, range_, expanded_box, idx, stage=1):
        outer_points = np.zeros((2 ** range_.shape[0], 2))
        # print("range of input:" , range_)
        outer_points = [[i, j] for i in range_[0, :] for j in range_[1, :]]
        # print("bounding box", outer_points)
        # inner_points[i] = expanded_box[i][0]
        range_area = np.product(range_[..., 1] - range_[..., 0])

        if stage == 1:
            c = 0.5

            pairwise_distance = np.zeros(len(outer_points))
            for (i, points) in enumerate(outer_points):
                pairwise_distance[i] = (
                    (points[0] - expanded_box[0]) ** 2
                    + (points[1] - expanded_box[1]) ** 2
                ) ** 0.5

            delta_step = (
                c
                * np.max((pairwise_distance / range_area ** 0.5))
                * np.ones((idx, 2))
            )
            #  print('delta step:',delta_step)
            return delta_step
        else:
            c = 0.2
            # print(outer_points)
            # print(expanded_box)
            delta_step = np.zeros((range_.shape[0], 2))

            distances = pairwise_distances(outer_points, expanded_box)
            min_distance = np.min(distances)
            for (i, distance_pair) in enumerate(min_distance):
                if i == idx:
                    delta_step[i] = c * distance_pair / range_area ** 0.5
            # print('delta step:',delta_step)
            return delta_step

    def expand_partition(
        self,
        propagator,
        sampled_inputs,
        sampled_outputs,
        output_range_sim,
        input_range,
        num_propagator_calls,
        propagator_computation_time,
    ):

        # tolerance_eps = 0.05
        t_start_overall = time.time()

        tolerance_step = 0.05
        tolerance_range = 0.01
        num_propagator_calls = 0
        num_inputs = input_range.shape[0]
        input_shape = input_range.shape[:-1]

        M = []  # (Line 4)
        interior_M = []

        # sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (self.num_simulations,)+input_shape)
        # sampled_outputs = propagator.forward_pass(sampled_inputs)
        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull

            self.sim_convex_hull = ConvexHull(sampled_outputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        # output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        # output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        #  output_range_sim[:,0] = np.min(sampled_outputs, axis=0)

        # sampled_output_center=output_range_sim.mean(axis=1)
        if self.interior_condition == "convex_hull":
            sampled_output_center = (
                self.sim_convex_hull.points[self.sim_convex_hull.vertices]
            ).mean(axis=0)
        else:
            sampled_output_center = output_range_sim.mean(axis=1)

        kdt = KDTree(sampled_outputs, leaf_size=30, metric="euclidean")
        center_NN = kdt.query(
            sampled_output_center.reshape(1, -1), k=1, return_distance=False
        )

        input_range_new = np.empty_like(input_range)
        input_range_new[:, 0] = np.min(sampled_inputs[center_NN], axis=0)
        input_range_new[:, 1] = np.max(sampled_inputs[center_NN], axis=0)

        count = 0
        M = []

        M_e = []
        delta_step = self.set_delta_step(
            output_range_sim, sampled_output_center, num_inputs, stage=1
        )

        prev_range = np.inf
        terminating_condition = False
        input_range_new = input_range_new + delta_step
        #  print("input", input_range_new )
        while terminating_condition == False:

            t_start = time.time()
            output_range_new, _ = propagator.get_output_range(input_range_new)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1
            if self.termination_condition_type == "num_propagator_calls" and (
                num_propagator_calls
                == (self.termination_condition_value) * 0.8
            ):
                break
            if self.check_if_partition_within_sim_bnds(
                output_range_new, output_range_sim
            ):
                input_range_new = input_range_new + delta_step
                if np.all(
                    (input_range[..., 0] - input_range_new[..., 0]) >= 0
                ) or np.all(
                    (input_range[..., 1] - input_range_new[..., 1]) <= 0
                ):
                    input_range_new -= delta_step
                    break
            #  terminating_condition==False
            else:
                input_range_new = input_range_new - delta_step
                break

                #   diff=(input_range-input_range_new)

            #     max_range= (np.max(abs(diff)))
            #     if max_range<tolerance_range:# or abs(max_range-prev_range)<tol
            #         break
            #     else:
            #         tol = 5
            #         diff_rolled= np.concatenate(abs(diff))
            #         if abs(max_range-prev_range)<tol:
            #             count  = count+1
            #        idx= num_inputs*2-1-count;
            #         prev_range= np.inf;
            #         if idx<0:
            #             break
            #          else:
            #              prev_range = max_range
            #              delta_step =self.set_delta_step(input_range, diff_rolled,idx, stage=2)
            #              if np.max(abs(delta_step))<tolerance_step:
            #                  break

        if input_range[0, 0] != input_range_new[0, 0]:

            input_range_ = np.array(
                [
                    [input_range[0, 0], input_range_new[0, 0]],
                    [input_range[1, 0], input_range[1, 1]],
                ]
            )
            t_start = time.time()
            output_range_, _ = propagator.get_output_range(input_range_)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1

            M.append((input_range_, output_range_))

        if [input_range_new[0, 1] != input_range[0, 1]]:

            # approch2 only
            input_range_ = np.array(
                [
                    [input_range_new[0, 1], input_range[0, 1]],
                    [input_range[1, 0], input_range[1, 1]],
                ]
            )
            t_start = time.time()
            output_range_, _ = propagator.get_output_range(input_range_)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1

            M.append((input_range_, output_range_))

        if [input_range_new[1, 1] != input_range[1, 1]]:

            input_range_ = np.array(
                [
                    [input_range_new[0, 0], input_range_new[0, 1]],
                    [input_range_new[1, 1], input_range[1, 1]],
                ]
            )
            t_start = time.time()
            output_range_, _ = propagator.get_output_range(input_range_)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1

            M.append((input_range_, output_range_))

        if [input_range_new[1, 0] != input_range[1, 0]]:
            # common partition between two approaches

            input_range_ = np.array(
                [
                    [input_range_new[0, 0], input_range_new[0, 1]],
                    [input_range[1, 0], input_range_new[1, 0]],
                ]
            )
            t_start = time.time()
            output_range_, _ = propagator.get_output_range(input_range_)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1

            M.append((input_range_, output_range_))

        #  u_e = output_range_sim.copy()

        if self.make_animation:
            self.setup_visualization(
                input_range,
                output_range_sim,
                propagator,
                show_input=self.show_input,
                show_output=self.show_output,
            )
        M_e = [(input_range_new, output_range_new)]
        return M, M_e, num_propagator_calls, propagator_computation_time
