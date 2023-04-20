from itertools import product
from typing import Union

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.elements as elements
import nfl_veripy.propagators as propagators
import numpy as np
import torch
from nfl_veripy.utils.utils import get_crown_matrices

from .ClosedLoopPartitioner import ClosedLoopPartitioner

"""
Note: This was an attempt to pull the partitioning logic from ACC23 out of the propagator.
It doesn't fully work but I'll leave this here as an example in case we want to use it someday.
For now, the ReBreach propagator will just do its partitioning internally.
"""


class ClosedLoopNickPartitioner(ClosedLoopPartitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
        num_partitions: Union[None, int, np.ndarray] = 16,
        make_animation: bool = False,
        show_animation: bool = False,
    ):
        ClosedLoopPartitioner.__init__(
            self,
            dynamics=dynamics,
            make_animation=make_animation,
            show_animation=show_animation,
        )
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

        # TODO: make these args of the class, set them in example_backward.py properly!
        self.slow_cvxpy = True
        self.refined = False
        self.heuristic = "uniform"
        self.all_lps = False

    """
    Inputs: 
        br_set_element: element object representing the whole backreachable set
        target_set: target set for current problem (i.e. can be previously calculated BP set)
        partition_budget: number of slices allowed to make

    Outputs: 
        element_list: list of partitioned elements to analyze
    """

    def partition(
        self,
        propagator,
        br_set_element,
        problems,
        target_set=None,
        partition_budget=1,
    ):
        min_split_volume = 0.00001  # Note: Discrete Quad 30 s
        i = 0
        element_list = [br_set_element]

        # If there are no samples that go to the target set and if the BR set contains the target set, I artificially set the sample set as the target set
        # This effectively makes it so that we don't partition within the target set (makes sense for obstacle avoidance where if the BP at t = -1 is the target set again, the BPs won't explode)
        if (
            len(br_set_element.samples) == 0
            and (
                br_set_element.ranges[:, 0] - target_set.range[:, 0] < 1e-6
            ).all()
            and (
                br_set_element.ranges[:, 1] - target_set.range[:, 1] > -1e-6
            ).all()
        ):
            br_set_element.samples = np.array(
                [target_set.range[:, 0], target_set.range[:, 1]]
            )

        # if the heuristic isn't uniform, we iteratively select elements to bisect
        if self.heuristic != "uniform":
            while (
                len(element_list) > 0
                and i < partition_budget
                and element_list[-1].prop > 0
            ):
                element_to_split = element_list.pop()

                # Determine if the element should be split (should be extreme element along some dimension)
                lower_list = []
                upper_list = []
                for el in element_list:
                    lower_list.append(
                        element_to_split.ranges[:, 0] <= el.ranges[:, 0]
                    )
                    upper_list.append(
                        element_to_split.ranges[:, 1] >= el.ranges[:, 1]
                    )
                lower_arr = np.array(lower_list)
                upper_arr = np.array(upper_list)
                try:
                    split_check_low = lower_arr.all(axis=0).any()
                    split_check_up = upper_arr.all(axis=0).any()
                except:
                    split_check_low = True
                    split_check_up = True

                # If element should be split, split it
                if element_to_split.A_edge.any() and (
                    split_check_low or split_check_up
                ):
                    new_elements = element_to_split.split(
                        target_set,
                        self.dynamics,
                        problems,
                        full_samples=br_set_element.samples,
                        br_set_range=br_set_element.ranges,
                        nstep=self.refined,
                        min_split_volume=min_split_volume,
                    )
                else:  # otherwise, cut el.prop in half and put it back into the list
                    element_to_split.prop = element_to_split.prop * 0.5
                    new_elements = [element_to_split]

                # Add newly generated elements to element_list
                import bisect

                for el in new_elements:
                    if el.get_volume() > 0:
                        bisect.insort(element_list, el)
                i += 1

        else:  # Uniform partitioning strategy; copy and pasted from earlier, but now gives a list of elements containing crown bounds
            element_list = []
            dim = br_set_element.ranges.shape[0]
            if not type(partition_budget).__module__ == np.__name__:
                num_partitions = np.array(
                    [partition_budget for i in range(dim)]
                )
                # import pdb; pdb.set_trace()
            else:
                num_partitions = partition_budget
            input_shape = br_set_element.ranges.shape[:-1]
            slope = np.divide(
                (
                    br_set_element.ranges[..., 1]
                    - br_set_element.ranges[..., 0]
                ),
                num_partitions,
            )
            for el in product(
                *[range(int(num)) for num in num_partitions.flatten()]
            ):
                element_ = np.array(el).reshape(input_shape)
                input_range_ = np.empty_like(
                    br_set_element.ranges, dtype=float
                )
                input_range_[..., 0] = br_set_element.ranges[
                    ..., 0
                ] + np.multiply(element_, slope)
                input_range_[..., 1] = br_set_element.ranges[
                    ..., 0
                ] + np.multiply(element_ + 1, slope)
                element = elements.Element(input_range_)

                num_control_inputs = self.dynamics.bt.shape[1]
                C = torch.eye(num_control_inputs).unsqueeze(0)
                nn_input_max = torch.Tensor(np.array([element.ranges[:, 1]]))
                nn_input_min = torch.Tensor(np.array([element.ranges[:, 0]]))
                norm = np.inf
                element.crown_bounds = {}
                (
                    element.crown_bounds["lower_A"],
                    element.crown_bounds["upper_A"],
                    element.crown_bounds["lower_sum_b"],
                    element.crown_bounds["upper_sum_b"],
                ) = propagator.network(
                    method_opt="full_backward_range",
                    norm=norm,
                    x_U=nn_input_max,
                    x_L=nn_input_min,
                    upper=True,
                    lower=True,
                    C=C,
                    return_matrices=True,
                )
                (
                    element.crown_bounds["lower_A"],
                    element.crown_bounds["upper_A"],
                    element.crown_bounds["lower_sum_b"],
                    element.crown_bounds["upper_sum_b"],
                ) = (
                    element.crown_bounds["lower_A"].detach().numpy()[0],
                    element.crown_bounds["upper_A"].detach().numpy()[0],
                    element.crown_bounds["lower_sum_b"].detach().numpy()[0],
                    element.crown_bounds["upper_sum_b"].detach().numpy()[0],
                )

                element_list.append(element)

        return element_list

    def get_one_step_backprojection_set(
        self,
        target_sets: constraints.MultiTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        num_partitions=None,
        overapprox: bool = False,
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        num_partitions = 5

        target_set = target_sets.get_constraint_at_time_index(-1)
        backreachable_set, info = self.get_one_step_backreachable_set(
            target_set
        )
        info["backreachable_set"] = backreachable_set

        backprojection_set = constraints.create_empty_constraint(
            boundary_type=propagator.boundary_type,
            num_facets=propagator.num_polytope_facets,
        )

        # Attempt at setting initial values for BP set other than +/- np.inf
        backprojection_set.range = backreachable_set.range

        # Generate samples to get an initial underapproximation of the BP set to start with
        # if using the refined flag, we want to use samples that actually reach the target set
        if self.refined:
            x_samples_inside_backprojection_set = (
                self.dynamics.get_true_backprojection_set(
                    backreachable_set,
                    target_set,
                    t_max=len(target_sets) * self.dynamics.dt,
                    controller=propagator.network,
                )
            )
        # otherwise, find samples that reach the previously calculated BP set
        else:
            x_samples_inside_backprojection_set = (
                self.dynamics.get_true_backprojection_set(
                    backreachable_set,
                    target_set,
                    t_max=self.dynamics.dt,
                    controller=propagator.network,
                )
            )

        br_set_element = elements.OptGuidedElement(
            backreachable_set.to_range(),
            propagator.network,
            samples=x_samples_inside_backprojection_set[:, 0, :],
        )

        # Set up backprojection LPs
        # TODO: make this also able to handle n-step
        problems = propagator.setup_LPs_1step()

        # Partition BR set
        element_list = self.partition(
            propagator,
            br_set_element,
            problems,
            target_set=target_sets,
            partition_budget=num_partitions,
        )

        if len(element_list) == 0:
            raise NotImplementedError
            # TODO: figure out what this condition means - it was in nick's code
            # backprojection_set = constraints.LpConstraint(range=np.hstack((np.inf*np.ones((num_states,1)), -np.inf*np.ones((num_states,1)))))
            # return backprojection_set, info

        # TODO: re-formulate idx2 check to avoid needing to define/check this
        nominal_facets = np.vstack(
            [
                np.eye(self.dynamics.num_states),
                -np.eye(self.dynamics.num_states),
            ]
        )

        for element in element_list:
            if element.flag != "infeasible":
                if not self.slow_cvxpy:
                    # TODO: implement a version of get_one_step_backprojection_set_overapprox that
                    # sets up the LP once, then each call just sets the params and solves it.
                    # This will be handled by a different propagator, rather than by the partitioner
                    raise NotImplementedError

                backreachable_set_this_cell = constraints.LpConstraint(
                    range=element.ranges
                )
                backreachable_set_this_cell.crown_matrices = (
                    element.crown_bounds
                )

                # Choose which dimensions of this cell's BP set we want to optimize over
                if self.all_lps:
                    # Solve an min + max LP for each state (which is often unnecessary)
                    facet_inds_to_optimize = np.arange(
                        2 * self.dynamics.num_states
                    )
                else:
                    # Check for which states the optimization can possibly make the BP set more conservative

                    # Assuming that we would nominally optimize over [I, -I], get a list/array of the indices
                    # of [I, -I] that could actually be worth optimizing over
                    min_idx = (
                        backreachable_set_this_cell.range[:, 0]
                        < backprojection_set.range[:, 0] - 1e-5
                    )
                    max_idx = (
                        backreachable_set_this_cell.range[:, 1]
                        > backprojection_set.range[:, 1] + 1e-5
                    )
                    facets_to_optimize = np.hstack((max_idx, min_idx))

                    if element.A_t_br is not None:
                        # Check which state (if any) was already optimized during the BR set calculation
                        idx2 = np.invert(
                            (nominal_facets == element.A_t_br).all(axis=1)
                        )
                        facets_to_optimize = np.vstack(
                            (facets_to_optimize, idx2)
                        ).all(axis=0)

                    facet_inds_to_optimize = np.where(facets_to_optimize)[0]

                backprojection_set_this_cell, this_info = (
                    propagator.get_one_step_backprojection_set(
                        backreachable_set_this_cell,
                        target_sets,
                        overapprox=overapprox,
                        facet_inds_to_optimize=facet_inds_to_optimize,
                    )
                )
                backprojection_set.add_cell_and_update_main_constraint(
                    backprojection_set_this_cell
                )

                # TODO: handle rotatedlpconstraints...
                # if isinstance(output_constraint, constraints.RotatedLpConstraint):
                #     info['mar_hull'], theta, xy, W, mar_vertices = find_MAR(info['bp_set_partitions'])
                #     input_constraint = constraints.RotatedLpConstraint(pose=xy, theta=theta, W=W, vertices=mar_vertices)
                #     t_start = time.time()
                #     lower_A_range, upper_A_range, lower_sum_b_range, upper_sum_b_range = self.network(
                #             method_opt=self.method_opt,
                #             norm=norm,
                #             x_U=torch.Tensor(np.array([input_constraint.bounding_box[:, 1]])),
                #             x_L=torch.Tensor(np.array([input_constraint.bounding_box[:, 0]])),
                #             upper=True,
                #             lower=True,
                #             C=C,
                #             return_matrices=True,
                #         )
                #     t_end = time.time()
                #     info['crown'].append(t_end-t_start)

                #     info['u_range'] = np.vstack((ut_min, ut_max)).T
                #     info['upper_A'] = upper_A_range.detach().numpy()[0]
                #     info['lower_A'] = lower_A_range.detach().numpy()[0]
                #     info['upper_sum_b'] = upper_sum_b_range.detach().numpy()[0]
                #     info['lower_sum_b'] = lower_sum_b_range.detach().numpy()[0]

        backprojection_set.update_main_constraint_with_cells(
            overapprox=overapprox
        )

        if overapprox:
            # These will be used to further backproject this set in time
            backprojection_set.crown_matrices = get_crown_matrices(
                propagator,
                backprojection_set,
                self.dynamics.num_inputs,
                self.dynamics.sensor_noise,
            )

        return backprojection_set, info


def find_MAR(partitions):
    import itertools

    points_list = []
    for constraint in partitions:
        A = constraint.range
        A_ = list(A[0, :])
        A__ = list(A[1, :])
        for r in itertools.product(A_, A__):
            if r[0] != np.inf and r[0] != -np.inf:
                points_list.append(np.array(r))

    points = np.array(points_list)
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)

    vertices = points[hull.vertices, :]

    min_area = np.inf
    mar_verts = np.empty((4, 2))

    for i in range(len(vertices)):
        next = i + 1
        if next == len(vertices):
            next = 0
        edge = vertices[next, :] - vertices[i, :]

        min_ext = np.inf
        max_ext = -np.inf
        max_idx, min_idx = 0, 0
        rays = vertices - vertices[i, :]
        for j, ray in enumerate(rays):
            projection = np.dot(edge, ray) / np.linalg.norm(edge)
            if projection > max_ext:
                max_idx = j
                max_ext = projection
            if projection < min_ext:
                min_idx = j
                min_ext = projection
        rect_verts = []
        rect_verts += [
            vertices[i, :] + max_ext * edge / np.linalg.norm(edge),
            vertices[i, :] + min_ext * edge / np.linalg.norm(edge),
        ]

        min_ext2 = np.inf
        max_ext2 = -np.inf
        max_idx2, min_idx2 = 0, 0
        rays2 = vertices - rect_verts[0]
        edge2 = vertices[max_idx, :] - rect_verts[0]
        for j, ray in enumerate(rays2):
            projection2 = np.dot(edge2, ray) / np.linalg.norm(edge2)
            if projection2 > max_ext2:
                max_idx2 = j
                max_ext2 = projection2
            if projection2 < min_ext2:
                min_idx2 = j
                min_ext2 = projection2
        rect_verts += [
            rect_verts[0] + max_ext2 * edge2 / np.linalg.norm(edge2),
            rect_verts[1] + max_ext2 * edge2 / np.linalg.norm(edge2),
        ]
        rect_verts = np.array(rect_verts)

        if not np.isnan(rect_verts).any():
            rect_hull = ConvexHull(rect_verts)
            area = rect_hull.volume
        else:
            area = np.inf

        if area < min_area:
            min_area = area
            mar_verts = rect_verts

    xy_idx = 0
    xy_min = np.inf
    for idx, vert in enumerate(mar_verts):
        if vert[1] <= xy_min:
            if (vert[1] == xy_min and vert[0] < mar_verts[xy_idx, 0]) or vert[
                1
            ] < xy_min:
                xy_idx = idx
                xy_min = vert[1]

    xy = mar_verts[xy_idx, :]

    theta_idx = 0
    theta = np.inf
    for idx, vert in enumerate(mar_verts):
        if idx != xy_idx:
            vert_theta = np.arctan2(vert[1] - xy[1], vert[0] - xy[0])
            if vert_theta < theta:
                theta = vert_theta
                theta_idx = idx

    mar_hull = ConvexHull(mar_verts)
    width = np.linalg.norm(mar_verts[theta_idx, :] - xy)
    height = mar_hull.volume / width
    W = np.array([width, height])

    return mar_hull, theta, xy, W, mar_verts
