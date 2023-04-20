import time
from copy import deepcopy

import cvxpy as cp
import nfl_veripy.constraints as constraints
import numpy as np
import torch
from nfl_veripy.elements import Element, GuidedElement


class OptGuidedElement(GuidedElement):
    def __init__(self, ranges, policy, samples=None, A_edge=None):
        super().__init__(ranges, policy, samples)
        self.A_t_br = None
        if A_edge is None:
            A_edge = np.ones(ranges.shape)
        self.A_edge = A_edge

    def split(
        self,
        target_set,
        dynamics,
        problems,
        full_samples=None,
        br_set_range=None,
        nstep=False,
        time_info=None,
        min_split_volume=0.00001,
    ):
        max_prob = problems[0]
        min_prob = problems[1]
        params = problems[2]
        # Grab bounds of MC samples in backreachable set (not necessarily in current element)
        t_start = time.time()
        if len(full_samples) > 0:
            sample_limits = np.array(
                [np.min(full_samples, axis=0), np.max(full_samples, axis=0)]
            )

        # Choose where to cut element and along which direction
        if (
            len(self.samples) == 0
        ):  # No samples in element -> bisect it hamburger style
            split_dim = np.argmax(np.ptp(self.ranges, axis=1))

            cut = (self.ranges[split_dim, 0] + self.ranges[split_dim, 1]) / 2
        else:  # samples in element -> split element near border of samples such that we maximize volume of new element without samples and minimize volume of element containing samples
            buffer = (
                0  # Set buffer to zero if we're cutting out the target set
            )
            if len(full_samples) > 2:
                buffer = 0.02
            diff_magnitude = np.abs(self.ranges.T - sample_limits)
            flat_idx = np.argmax(diff_magnitude)

            # Index gives information by row (should we make the cut above or below the set of samples) and by column (which dimension are we going to split)
            idx = np.unravel_index(flat_idx, diff_magnitude.shape)

            # Shift the cut position away from the samples a bit to account for underapprox of samples;
            if idx[0] == 0:
                buffer = -np.abs(buffer)
            else:
                buffer = np.abs(buffer)

            split_dim = idx[1]
            cut = sample_limits[idx] + buffer * (
                sample_limits.T[split_dim, 1] - sample_limits.T[split_dim, 0]
            )

        split_samples = (
            self.samples[self.samples[:, split_dim] < cut],
            self.samples[self.samples[:, split_dim] > cut],
        )

        # Setup variables to construct new elements
        lower_split_range = np.array([self.ranges[split_dim, 0], cut])
        upper_split_range = np.array([cut, self.ranges[split_dim, 1]])

        new_ranges = deepcopy(self.ranges), deepcopy(self.ranges)
        new_A_edges = deepcopy(self.A_edge), deepcopy(self.A_edge)
        new_ranges[0][split_dim] = lower_split_range
        new_ranges[1][split_dim] = upper_split_range
        new_A_edges[0][split_dim, 0] = 0
        new_A_edges[1][split_dim, 1] = 0

        # Generate new elements
        elements = OptGuidedElement(
            new_ranges[0],
            self.policy,
            samples=split_samples[0],
            A_edge=new_A_edges[0],
        ), OptGuidedElement(
            new_ranges[1],
            self.policy,
            samples=split_samples[1],
            A_edge=new_A_edges[1],
        )

        # center of samples will be used to determine which direction to optimize in when testing constraint satisfiability
        if len(full_samples) > 0:
            sample_center = np.mean(full_samples, axis=0)
        else:
            sample_center = np.mean(br_set_range.T, axis=0)

        t_end = time.time()
        if time_info is not None:
            time_info["other"].append(t_end - t_start)
        for (
            el
        ) in (
            elements
        ):  # For each of the newly generated elements, we need to see if the BP constriants can be satisfied.
            # In the process, we'll conduct one optimization and set its el.prop property, which is used to sort the elements in the list of elements to partition
            time_info["br_set_partitions"] += [
                constraints.LpConstraint(range=el.ranges)
            ]
            t_start = time.time()
            num_control_inputs = dynamics.bt.shape[1]
            C = torch.eye(num_control_inputs).unsqueeze(0)

            nn_input_max = torch.Tensor(np.array([el.ranges[:, 1]]))
            nn_input_min = torch.Tensor(np.array([el.ranges[:, 0]]))
            norm = np.inf

            # Set up element's CROWN bounds
            el.crown_bounds = {}
            t_end = time.time()
            if time_info is not None:
                time_info["other"].append(t_end - t_start)
            t_start = time.time()
            (
                el.crown_bounds["lower_A"],
                el.crown_bounds["upper_A"],
                el.crown_bounds["lower_sum_b"],
                el.crown_bounds["upper_sum_b"],
            ) = self.policy(
                method_opt="full_backward_range",
                norm=norm,
                x_U=nn_input_max,
                x_L=nn_input_min,
                upper=True,
                lower=True,
                C=C,
                return_matrices=True,
            )
            t_end = time.time()
            if time_info is not None:
                time_info["crown"].append(t_end - t_start)
            t_start = time.time()

            (
                el.crown_bounds["lower_A"],
                el.crown_bounds["upper_A"],
                el.crown_bounds["lower_sum_b"],
                el.crown_bounds["upper_sum_b"],
            ) = (
                el.crown_bounds["lower_A"].detach().numpy()[0],
                el.crown_bounds["upper_A"].detach().numpy()[0],
                el.crown_bounds["lower_sum_b"].detach().numpy()[0],
                el.crown_bounds["upper_sum_b"].detach().numpy()[0],
            )

            lower_A, lower_sum_b, upper_A, upper_sum_b = (
                el.crown_bounds["lower_A"],
                el.crown_bounds["lower_sum_b"],
                el.crown_bounds["upper_A"],
                el.crown_bounds["upper_sum_b"],
            )

            # Determine if the element is the one containing samples (no need to split this one)
            is_terminal_cell = False
            if len(full_samples) > 0:
                diff_magnitude = np.abs(el.ranges.T - sample_limits)
                if np.max(diff_magnitude) < 0.05:
                    is_terminal_cell = True
            t_end = time.time()
            if time_info is not None:
                time_info["other"].append(t_end - t_start)

            # If element doesn't contain samples:
            if not is_terminal_cell:
                t_start = time.time()

                # Set up LP parameters
                if isinstance(target_set, constraints.LpConstraint):
                    xt1_min = target_set.range[..., 0]
                    xt1_max = target_set.range[..., 1]
                else:
                    raise NotImplementedError

                if not nstep:
                    params["lower_A"].value = lower_A
                    params["upper_A"].value = upper_A
                    params["lower_sum_b"].value = lower_sum_b
                    params["upper_sum_b"].value = upper_sum_b

                    params["xt_min"].value = el.ranges[:, 0]
                    params["xt_max"].value = el.ranges[:, 1]

                    params["xt1_min"].value = xt1_min
                    params["xt1_max"].value = xt1_max

                else:
                    params["lower_A"][0].value = lower_A
                    params["upper_A"][0].value = upper_A
                    params["lower_sum_b"][0].value = lower_sum_b
                    params["upper_sum_b"][0].value = upper_sum_b

                    params["xt_min"][0].value = el.ranges[:, 0]
                    params["xt_max"][0].value = el.ranges[:, 1]

                # Choose along which dimension to optimize (will be dimension with greatest component from sample_center)
                # TODO make it so that we optimize in the direction that is limiting the BP set (I think this is usually the case as is)
                diff = el.ranges.T - sample_center
                diff_magnitude = np.abs(diff)
                flat_idx = np.argmax(diff_magnitude)
                idx = np.unravel_index(flat_idx, diff_magnitude.shape)

                # Construct indicator for optimization
                A_t_i = np.zeros(el.ranges.shape[0])
                A_t_i[idx[1]] = 1.0

                # Record optimized direction so it can be skipped later
                el.A_t_br = A_t_i
                params["A_t"].value = A_t_i

                if (
                    idx[0] == 0
                ):  # If we're below the sample center, find min value and reverse the stored direction
                    prob = min_prob
                    el.A_t_br = -el.A_t_br
                else:  # If we're above the sample center, find the max value
                    prob = max_prob
                t_end = time.time()
                if time_info is not None:
                    time_info["other"].append(t_end - t_start)

                # Solve the LP and record the updated element range/if the BP constraints can be satisfied (stored in el.flag)
                try:
                    t_start = time.time()
                    prob.solve()
                    t_end = time.time()
                    if time_info is not None:
                        time_info["bp_lp"].append(t_end - t_start)
                    temp = deepcopy(el.ranges.T)
                    temp[idx] = prob.value
                except:
                    el.flag = "infeasible"
                    temp = deepcopy(el.ranges.T)

                el.ranges = temp.T
                el.flag = prob.status

            t_start = time.time()
            # If the element is not feasible (or is element bounding samples), assign el.prop = 0
            # (this will put it at the front of the partitioner's element list, effectively removing it from elements to be split)
            if el.flag == "infeasible" or is_terminal_cell:
                el.prop = 0
            # Else, el.prop is determined by distance element's furthest corner from the relavent corner of the samples (i.e. how much worse is this element making the BP set estimate)
            else:
                if "sample_limits" in locals():
                    dist = 0
                    for i in range(dynamics.At.shape[0]):
                        dist_min, dist_max = 0, 0
                        if el.ranges[i, 0] < sample_limits.T[i, 0]:
                            dist_min = np.abs(
                                el.ranges[i, 0] - sample_limits.T[i, 0]
                            )
                        if el.ranges[i, 1] > sample_limits.T[i, 1]:
                            dist_max = np.abs(
                                el.ranges[i, 1] - sample_limits.T[i, 1]
                            )

                        mod = 1
                        if dynamics.At.shape[0] == 6 and i > 2:
                            mod = 0
                        dist += np.max([dist_min, dist_max]) * mod
                else:  # If there are no samples/target set to use as reference, use the sample center (in this case would just be center of BR set)
                    if dynamics.At.shape[0] == 6:
                        dist = np.linalg.norm(
                            np.max(
                                np.abs(el.ranges.T - sample_center), axis=0
                            )[0:3],
                            1,
                        )
                    else:
                        dist = np.linalg.norm(
                            np.max(
                                np.abs(el.ranges.T - sample_center), axis=0
                            ),
                            1,
                        )
                el.prop = dist

                volume = el.volume
                br_set_volume = np.prod(
                    br_set_range[:, 1] - br_set_range[:, 0], axis=0
                )
                # If the element's volume is too small, set el.prop = 0 so it won't be split further
                if volume < min_split_volume * br_set_volume:
                    el.prop = 0

            t_end = time.time()
            if time_info is not None:
                time_info["other"].append(t_end - t_start)

        return elements
