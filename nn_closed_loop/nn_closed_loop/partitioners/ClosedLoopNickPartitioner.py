from .ClosedLoopPartitioner import ClosedLoopPartitioner
import nn_closed_loop.constraints as constraints
import numpy as np
import pypoman
from itertools import product
from copy import deepcopy
from nn_closed_loop.utils.utils import range_to_polytope, get_crown_matrices

import nn_closed_loop.dynamics as dynamics
import nn_closed_loop.propagators as propagators
from typing import Optional, Union


class ClosedLoopNickPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics: dynamics.Dynamics, num_partitions: Union[None, int, np.ndarray] = 16, make_animation: bool = False, show_animation: bool = False):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics, make_animation=make_animation, show_animation=show_animation)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    '''
    Inputs: 
        br_set_element: element object representing the whole backreachable set
        target_set: target set for current problem (i.e. can be previously calculated BP set)
        dynamics: system dynamics
        partition_budget: number of slices allowed to make
        heuristic: method used to decide which element to split next

    Outputs: 
        element_list: list of partitioned elements to analyze
    '''
    def partition(self, br_set_element, problems, target_set=None, dynamics=None, partition_budget=1, heuristic='guided', nstep=False, info=None):
        min_split_volume=0.00001 # Note: Discrete Quad 30 s
        i = 0
        element_list = [br_set_element]

        # If there are no samples that go to the target set and if the BR set contains the target set, I artificially set the sample set as the target set
        # This effectively makes it so that we don't partition within the target set (makes sense for obstacle avoidance where if the BP at t = -1 is the target set again, the BPs won't explode)
        if len(br_set_element.samples) == 0 and (br_set_element.ranges[:, 0] - target_set.range[:, 0] < 1e-6).all() and (br_set_element.ranges[:, 1] - target_set.range[:, 1] > -1e-6).all():
            br_set_element.samples = np.array([target_set.range[:, 0], target_set.range[:, 1]])
            

        # if the heuristic isn't uniform, we iteratively select elements to bisect
        if heuristic != 'uniform':
            while len(element_list) > 0 and i < partition_budget and element_list[-1].prop > 0:
                element_to_split = element_list.pop()

                # Determine if the element should be split (should be extreme element along some dimension)
                lower_list = []
                upper_list = []
                for el in element_list:
                    lower_list.append(element_to_split.ranges[:, 0] <= el.ranges[:, 0])
                    upper_list.append(element_to_split.ranges[:, 1] >= el.ranges[:, 1])
                lower_arr = np.array(lower_list)
                upper_arr = np.array(upper_list)
                try:
                    split_check_low = lower_arr.all(axis=0).any()
                    split_check_up = upper_arr.all(axis=0).any()
                except:
                    split_check_low = True
                    split_check_up = True
                
                # If element should be split, split it
                if element_to_split.A_edge.any() and (split_check_low or split_check_up):
                    new_elements = element_to_split.split(target_set, dynamics, problems, full_samples=br_set_element.samples, br_set_range=br_set_element.ranges, nstep=nstep, time_info=info, min_split_volume=min_split_volume)
                else: # otherwise, cut el.prop in half and put it back into the list
                    element_to_split.prop = element_to_split.prop*0.5
                    new_elements = [element_to_split]

                # Add newly generated elements to element_list
                t_start = time.time()
                import bisect
                for el in new_elements:
                    if el.get_volume() > 0:
                        bisect.insort(element_list, el)
                i+=1
                t_end = time.time()
                if info is not None:
                    info['other'].append(t_end-t_start)
                
        else: # Uniform partitioning strategy; copy and pasted from earlier, but now gives a list of elements containing crown bounds
            t_start = time.time()
            element_list = []
            dim = br_set_element.ranges.shape[0]
            if not type(partition_budget).__module__ == np.__name__:
                num_partitions = np.array([partition_budget for i in range(dim)])
                # import pdb; pdb.set_trace()
            else:
                num_partitions = partition_budget
            input_shape = br_set_element.ranges.shape[:-1]
            slope = np.divide(
                (br_set_element.ranges[..., 1] - br_set_element.ranges[..., 0]), num_partitions
            )
            t_end = time.time()
            if info is not None:
                info['other'].append(t_end-t_start)
            for el in product(*[range(int(num)) for num in num_partitions.flatten()]):
                t_start = time.time()
                element_ = np.array(el).reshape(input_shape)
                input_range_ = np.empty_like(br_set_element.ranges, dtype=float)
                input_range_[..., 0] = br_set_element.ranges[..., 0] + np.multiply(
                    element_, slope
                )
                input_range_[..., 1] = br_set_element.ranges[..., 0] + np.multiply(
                    element_ + 1, slope
                )
                element = elements.Element(input_range_)
                

                num_control_inputs = self.dynamics.bt.shape[1]
                C = torch.eye(num_control_inputs).unsqueeze(0)
                nn_input_max = torch.Tensor(np.array([element.ranges[:,1]]))
                nn_input_min = torch.Tensor(np.array([element.ranges[:,0]]))
                norm = np.inf
                element.crown_bounds = {}
                t_end = time.time()
                if info is not None:
                    info['other'].append(t_end-t_start)
                t_start = time.time()
                element.crown_bounds['lower_A'], element.crown_bounds['upper_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_sum_b'] = self.network(
                    method_opt='full_backward_range',
                    norm=norm,
                    x_U=nn_input_max,
                    x_L=nn_input_min,
                    upper=True,
                    lower=True,
                    C=C,
                    return_matrices=True,
                )
                element.crown_bounds['lower_A'], element.crown_bounds['upper_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_sum_b'] = element.crown_bounds['lower_A'].detach().numpy()[0], element.crown_bounds['upper_A'].detach().numpy()[0], element.crown_bounds['lower_sum_b'].detach().numpy()[0], element.crown_bounds['upper_sum_b'].detach().numpy()[0]
                t_end = time.time()
                if info is not None:
                    info['crown'].append(t_end-t_start)

                element_list.append(element)
        
        return element_list

    def get_one_step_backprojection_set(
        self, target_sets: constraints.MultiTimestepConstraint, propagator: propagators.ClosedLoopPropagator, num_partitions=None, overapprox: bool = False
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:

        backreachable_set, info = self.get_one_step_backreachable_set(target_sets.get_constraint_at_time_index(-1))
        info['backreachable_set'] = backreachable_set

        backprojection_set = constraints.create_empty_constraint(boundary_type=propagator.boundary_type, num_facets=propagator.num_polytope_facets)

        '''
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to backprojection_set
        '''

        # Generate samples to get an initial underapproximation of the BP set to start with
        # if using the refined flag, we want to use samples that actually reach the target set
        if refined:
            x_samples_inside_backprojection_set = self.dynamics.get_true_backprojection_set(backreachable_set, collected_input_constraints[0], t_max=len(collected_input_constraints)*self.dynamics.dt, controller=self.network)
        # otherwise, find samples that reach the previously calculated BP set
        else:
            x_samples_inside_backprojection_set = self.dynamics.get_true_backprojection_set(backreachable_set, output_constraint, t_max=self.dynamics.dt, controller=self.network)

        nstep = False
        if refined and len(collected_input_constraints) > 1:
            nstep = True
        
        br_set_element = elements.OptGuidedElement(ranges, self.network, samples=x_samples_inside_backprojection_set[:,0,:])


        # Set up backprojection LPs
        problems = self.setup_LPs(nstep, 0, infos, collected_input_constraints)

        # Partition BR set
        element_list = self.partition(
            br_set_element, 
            problems,
            target_set=target_sets, 
            dynamics=self.dynamics, 
            partition_budget=num_partitions, 
            heuristic=heuristic,
            nstep=nstep,
            info=info
        )

        if len(element_list) == 0:
            raise NotImplementedError
            # TODO: figure out what this condition means - it was in nick's code
            # backprojection_set = constraints.LpConstraint(range=np.hstack((np.inf*np.ones((num_states,1)), -np.inf*np.ones((num_states,1)))))
            # return backprojection_set, info

        for element in element_list:
            if element.flag is not 'infeasible':

                # TODO: populate backreachable_set_this_cell w/ element

                # TODO:
                # if overapprox && nstep, should use get_refined_one_step_backprojection_set_overapprox
                # if overapprox && nstep, should use get_one_step_backprojection_set_overapprox
                # if not overapprox, should use get_one_step_backprojection_set_underapprox

                ############################
                # Flag to use naive partitioning
                if all_lps:
                    idx1 = np.ones(2*num_states)

                # TODO: get this working -- check which states are worth optimizing over.
                # ... only do the optimization if idx[i] == True.
                # by default, BP set of this cell should be A_t, b_NN -- update just the entries of b_NN that have idx[i] == True
                # where A_NN, b_NN = range_to_polytope(ranges) and A_t_ = np.vstack([A_t, -A_t])

                # Check for which states the optimization can possibly make the BP set more conservative
                min_idx = xt_min < xt_range_min - 1e-5
                max_idx = xt_max > xt_range_max + 1e-5
                idx1 = np.hstack((max_idx, min_idx))
                

                # Check which state (if any) was already optimized during the BR set calculation
                idx2 = np.invert((A_t_ == element.A_t_br).all(axis=1))
                idx = np.vstack((idx1, idx2)).all(axis=0)
                ############################


                if not slow_cvxpy:
                    # TODO: implement a version of get_one_step_backprojection_set_overapprox that
                    # sets up the LP once, then each call just sets the params and solves it.
                    raise NotImplementedError
                backprojection_set_this_cell, this_info = propagator.get_one_step_backprojection_set(
                    backreachable_set_this_cell,
                    target_sets,
                    overapprox=overapprox,
                )
                backprojection_set.add_cell(backprojection_set_this_cell)


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

        backprojection_set.update_main_constraint_with_cells(overapprox=overapprox)

        if overapprox:

            # These will be used to further backproject this set in time
            backprojection_set.crown_matrices = get_crown_matrices(
                propagator,
                backprojection_set,
                self.dynamics.num_inputs,
                self.dynamics.sensor_noise
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
    from scipy.spatial import ConvexHull, convex_hull_plot_2d

    hull = ConvexHull(points)

    vertices = points[hull.vertices, :]

    min_area = np.inf
    mar_verts = np.empty((4,2))

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
            projection = np.dot(edge, ray)/np.linalg.norm(edge)
            if projection > max_ext:
                max_idx = j
                max_ext = projection
            if projection < min_ext:
                min_idx = j
                min_ext = projection
        rect_verts = []
        rect_verts += [vertices[i, :]+max_ext*edge/np.linalg.norm(edge), vertices[i, :]+min_ext*edge/np.linalg.norm(edge)]
        
        min_ext2 = np.inf
        max_ext2 = -np.inf
        max_idx2, min_idx2 = 0, 0
        rays2 = vertices - rect_verts[0]
        edge2 = vertices[max_idx, :] - rect_verts[0]
        for j, ray in enumerate(rays2):
            projection2 = np.dot(edge2, ray)/np.linalg.norm(edge2)
            if projection2 > max_ext2:
                max_idx2 = j
                max_ext2 = projection2
            if projection2 < min_ext2:
                min_idx2 = j
                min_ext2 = projection2
        rect_verts += [rect_verts[0]+max_ext2*edge2/np.linalg.norm(edge2), rect_verts[1]+max_ext2*edge2/np.linalg.norm(edge2)]
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
            if (vert[1] == xy_min and vert[0] < mar_verts[xy_idx, 0]) or vert[1] < xy_min:
                xy_idx = idx
                xy_min = vert[1]

    xy = mar_verts[xy_idx, :]

    theta_idx = 0
    theta = np.inf
    for idx, vert in enumerate(mar_verts):
        if idx != xy_idx:
            vert_theta = np.arctan2(vert[1]-xy[1], vert[0]-xy[0])
            if vert_theta < theta:
                theta = vert_theta
                theta_idx = idx

    
    mar_hull = ConvexHull(mar_verts)
    width = np.linalg.norm(mar_verts[theta_idx, :]-xy)
    height = mar_hull.volume/width
    W = np.array([width, height])

    return mar_hull, theta, xy, W, mar_verts




