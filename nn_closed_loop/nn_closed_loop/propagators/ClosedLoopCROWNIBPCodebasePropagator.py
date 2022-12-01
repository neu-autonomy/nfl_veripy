from bisect import bisect
from hashlib import new
import multiprocessing
from posixpath import split
from tkinter.messagebox import NO
from tokenize import Hexnumber

import matplotlib
from .ClosedLoopPropagator import ClosedLoopPropagator
import nn_closed_loop.nn_closed_loop.elements as elements
import numpy as np
import pypoman
import nn_closed_loop.nn_closed_loop.constraints as constraints
import torch
from nn_closed_loop.nn_closed_loop.utils.utils import range_to_polytope
import cvxpy as cp
from itertools import product
from copy import deepcopy

import time
import os

# class Element():
#     def __init__(self, ranges, x_samples_inside_backprojection_set=None, policy=None, heuristic='split_most'):
#         self.ranges = ranges
#         self.samples = x_samples_inside_backprojection_set
#         self.policy = policy
#         if heuristic is 'split_most':
#             self.prop = len(self.samples)
#         elif heuristic is 'box_out':
#             self.prop = self.get_volume()
#         elif heuristic is 'uniform':
#             self.prop = 0
#         elif heuristic is 'guided':
#             self.prop = 1
#         else:
#             raise NotImplementedError
#         self.flag = None
    
#     def __lt__(self, other):
#         if self.prop == other.prop:
#             return self.get_volume() < other.get_volume()
#         return self.prop < other.prop

#     def get_volume(self):
#         diff = self.ranges[:,1] - self.ranges[:,0]
#         return np.prod(diff, axis=0)

    
#     def split(self, target_set=None, dynamics=None, heuristic=None, full_samples=None):

#         ############################# Ignore this one #############################
#         if heuristic is 'split_most':
#             max_samples = -np.inf
#             split_dim = 0
#             for i,dim in enumerate(self.ranges):
                
#                 avg = (dim[0]+dim[1])/2
#                 # samples_above = self.samples[self.samples[:,i] > avg]
#                 # samples_below = self.samples[self.samples[:,i] <= avg]
#                 split_samples_candidate = self.samples[self.samples[:,i] < avg], self.samples[self.samples[:,i] > avg]
#                 for j,side in enumerate(split_samples_candidate):
#                     if len(side) > max_samples:
#                         max_samples = len(side)
#                         split_dim = i
#                         split_samples = split_samples_candidate

#             cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
        
#         ############################# Ignore this one #############################
#         elif heuristic is 'box_out':
#             buffer = 0
#             xtreme = np.array(
#                 [
#                     np.min(full_samples,axis=0),
#                     np.max(full_samples,axis=0)
#                 ]
#             )
#             for i,dim in enumerate(self.ranges):
#                 diff_magnitude = np.abs(self.ranges.T - xtreme)
#                 # import pdb; pdb.set_trace()
#                 flat_idx = np.argmax(diff_magnitude)
#                 idx = np.unravel_index(flat_idx, diff_magnitude.shape)
#                 split_dim = idx[1]
#                 if len(self.samples) == 0:
#                     # import pdb; pdb.set_trace()
#                     cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
#                 else:
#                     if  idx[0] == 0:
#                         buffer = -np.abs(buffer)
#                     else: 
#                         buffer = np.abs(buffer)

#                     cut = xtreme[idx] + buffer

        
#         ############################# This one has promise #############################
#         elif heuristic is 'guided':
#             # Grab bounds of MC samples in backreachable set (not necessarily in current element)
#             if len(full_samples) > 0: 
#                 xtreme = np.array(
#                     [
#                         np.min(full_samples,axis=0),
#                         np.max(full_samples,axis=0)
#                     ]
#                 )
#             elif target_set is not None:
#                 xtreme = target_set.range.T
            
#             # Choose where to cut element and along which direction
#             if len(self.samples) == 0:
#                 # # Possible idea of choosing dimension based on crown bounds
#                 # if not hasattr(self, 'crown_bounds'):
#                 #     split_dim = np.argmax(np.ptp(self.ranges, axis=1))
#                 # else: 
#                 #     split_dim = np.argmax(np.abs(self.crown_bounds['upper_A']-self.crown_bounds['lower_A']))

#                 # No samples in element -> bisect it hamburger style
#                 split_dim = np.argmax(np.ptp(self.ranges, axis=1))
#                 cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
#             else:

#                 # samples in element -> split element near border of samples such that we maximize volume of new element without samples and minimize volume of element containing samples
#                 buffer = 0.02
#                 diff_magnitude = np.abs(self.ranges.T - xtreme)
#                 flat_idx = np.argmax(diff_magnitude)
#                 idx = np.unravel_index(flat_idx, diff_magnitude.shape)

#                 if  idx[0] == 0:
#                     buffer = -np.abs(buffer)
#                 else: 
#                     buffer = np.abs(buffer)

                
#                 split_dim = idx[1]
#                 cut = xtreme[idx] + buffer*(xtreme.T[split_dim,1]-xtreme.T[split_dim,0])
            

#             # import pdb; pdb.set_trace()

#         elif heuristic is None:
#             raise NotImplementedError

#         # split samples into regions contained by new elements
#         split_samples = self.samples[self.samples[:,split_dim] < cut], self.samples[self.samples[:,split_dim] > cut]

        
#         lower_split_range = np.array([self.ranges[split_dim,0], cut])
#         upper_split_range = np.array([cut, self.ranges[split_dim,1]])
        
#         new_ranges = deepcopy(self.ranges), deepcopy(self.ranges)
#         new_ranges[0][split_dim] = lower_split_range
#         new_ranges[1][split_dim] = upper_split_range
        
#         # Generate new elements
#         elements = Element(new_ranges[0], split_samples[0], heuristic=heuristic, policy=self.policy), Element(new_ranges[1], split_samples[1], heuristic=heuristic, policy=self.policy)

#         # Assign value to new elements (used to sort list of elements to be partitioned)
#         if heuristic is 'box_out':
#             for el in elements:
#                 # import pdb; pdb.set_trace()
#                 if len(set(el.ranges.flatten()).intersection(set(np.hstack((xtreme.flatten(), xtreme.flatten()+buffer, xtreme.flatten()-buffer))))) == 0:
#                     el.prop = el.prop*0

#         elif heuristic is 'guided':
#             if len(full_samples) > 0:
#                 sample_center = np.mean(full_samples, axis=0)
#             else:
#                 sample_center = np.mean(self.ranges.T, axis=0)
#             for el in elements:
#                 # if len(el.samples) > 0: # if the element contains samples, prioritize it in the queue
#                 #     # el.prop += np.inf
#                 #     element_center = np.mean(self.ranges, axis=1)
#                 #     el.prop += np.linalg.norm(element_center-sample_center, 1)
#                 # else: # otherwise, determine if it is feasible to reach the target set from this element and if so assign a cost
#                 num_control_inputs = dynamics.bt.shape[1]
#                 C = torch.eye(num_control_inputs).unsqueeze(0)

#                 nn_input_max = torch.Tensor(np.array([el.ranges[:,1]]))
#                 nn_input_min = torch.Tensor(np.array([el.ranges[:,0]]))
#                 norm = np.inf

#                 el.crown_bounds = {}
#                 el.crown_bounds['lower_A'], el.crown_bounds['upper_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_sum_b'] = self.policy(
#                     method_opt='full_backward_range',
#                     norm=norm,
#                     x_U=nn_input_max,
#                     x_L=nn_input_min,
#                     upper=True,
#                     lower=True,
#                     C=C,
#                     return_matrices=True,
#                 )

#                 el.crown_bounds['lower_A'], el.crown_bounds['upper_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_sum_b'] = el.crown_bounds['lower_A'].detach().numpy()[0], el.crown_bounds['upper_A'].detach().numpy()[0], el.crown_bounds['lower_sum_b'].detach().numpy()[0], el.crown_bounds['upper_sum_b'].detach().numpy()[0]
                            
#                 lower_A, lower_sum_b, upper_A, upper_sum_b = el.crown_bounds['lower_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_A'], el.crown_bounds['upper_sum_b']
                
#                 if isinstance(target_set, constraints.LpConstraint):
#                     xt1_min = target_set.range[..., 0]
#                     xt1_max = target_set.range[..., 1]
#                 else:
#                     raise NotImplementedError
                
#                 xt = cp.Variable(xt1_min.shape)
#                 ut = cp.Variable(num_control_inputs)
#                 constrs = []

#                 # Constraints to ensure that xt stays within the backreachable set
#                 constrs += [el.ranges[:, 0]+0.0 <= xt]
#                 constrs += [xt <= el.ranges[:,1]-0.0]

#                 # Constraints to ensure that ut satisfies the affine bounds
#                 constrs += [lower_A@xt+lower_sum_b <= ut]
#                 constrs += [ut <= upper_A@xt+upper_sum_b]

#                 # Constraints to ensure xt reaches the target set given ut
#                 constrs += [dynamics.dynamics_step(xt, ut) <= xt1_max]
#                 constrs += [dynamics.dynamics_step(xt, ut) >= xt1_min]

#                 # print("element range: \n {}".format(el.ranges))
#                 obj = 0
#                 prob = cp.Problem(cp.Maximize(obj), constrs)
#                 t_start = time.time()
#                 prob.solve()
#                 t_end = time.time()
#                 del_t = t_end-t_start

#                 A_t_i = np.array([1., 0])
#                 new_obj = A_t_i@xt
#                 new_prob = cp.Problem(cp.Maximize(new_obj), constrs)
#                 new_t_start = time.time()
#                 new_prob.solve()
#                 new_t_end = time.time()
#                 new_del_t = new_t_end - new_t_start

#                 print('positive is good: {}'.format(del_t-new_del_t))
#                 print(del_t)
#                 print(new_del_t)

#                 # new_obj = np.array([0, 1])
#                 # print("feasibility checked in {} seconds".format(t_end-t_start))
#                 # import pdb; pdb.set_trace()
#                 el.flag = prob.status
#                 is_terminal_cell = False
#                 if len(full_samples) > 0:
#                     diff_magnitude = np.abs(el.ranges.T - xtreme)
#                     if np.max(diff_magnitude) < 0.05:
#                         is_terminal_cell = True
#                 # print("lp solution for feasibility: {}".format(xt.value))
#                 # print("lp status for feasibility: {}".format(element_feasibility))
                
                
#                 # If the element is not feasible (or is element bounding samples), assign value to zero
#                 if el.flag == 'infeasible' or is_terminal_cell:
#                     el.prop = 0
                
#                 # Else, value is determined by (distance of furthest corner from sample center) * (volume of cell)
#                 else:

#                     element_center = np.mean(el.ranges, axis=1)
#                     # import pdb; pdb.set_trace()
#                     # dist = np.linalg.norm(element_center-sample_center, 1)
#                     dist = np.linalg.norm(np.max(np.abs(el.ranges.T-sample_center), axis=0), 1)
#                     volume = el.get_volume()
#                     el.prop = dist*volume
#                     # print(el.ranges)
#                     # print(dist)
#                     # import pdb; pdb.set_trace()
#                     # if len(el.samples) > 0 and dist < 0.01:
#                     #     print('whoaaaaaaa')


#         return elements
        
            
                
            
            



                
                


class ClosedLoopCROWNIBPCodebasePropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )

    def torch2network(self, torch_model):
        from nn_closed_loop.nn_closed_loop.utils.nn_bounds import BoundClosedLoopController

        torch_model_cl = BoundClosedLoopController.convert(
            torch_model, dynamics=self.dynamics, bound_opts=self.params
        )
        return torch_model_cl

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_one_step_reachable_set(self, input_constraint, output_constraint):

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # Get bounds on each state from A_inputs, b_inputs
            try:
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(A_inputs, b_inputs)
                )
            except:
                # Sometimes get arithmetic error... this may fix it
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(
                        A_inputs, b_inputs + 1e-6
                    )
                )
            x_max = np.max(vertices, 0)
            x_min = np.min(vertices, 0)
            norm = np.inf
        elif isinstance(input_constraint, constraints.LpConstraint):
            x_min = input_constraint.range[..., 0]
            x_max = input_constraint.range[..., 1]
            norm = input_constraint.p
            A_inputs = None
            b_inputs = None
        else:
            raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeConstraint):
            A_out = output_constraint.A
            num_facets = A_out.shape[0]
            bs = np.zeros((num_facets))
        elif isinstance(output_constraint, constraints.LpConstraint):
            A_out = np.eye(x_min.shape[0])
            num_facets = A_out.shape[0]
            ranges = np.zeros((num_facets, 2))
        else:
            raise NotImplementedError

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        prev_state_max = torch.Tensor([x_max])
        prev_state_min = torch.Tensor([x_min])
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor(np.array([self.dynamics.sensor_noise[:, 1]]))
            nn_input_min += torch.Tensor(np.array([self.dynamics.sensor_noise[:, 0]]))

        # Compute the NN output matrices (for the input constraints)
        num_control_inputs = self.dynamics.bt.shape[1]
        C = torch.eye(num_control_inputs).unsqueeze(0)
        # import pdb; pdb.set_trace()
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
            method_opt=self.method_opt,
            norm=norm,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )


        for i in range(num_facets):
            # For each dimension of the output constraint (facet/lp-dimension):
            # compute a bound of the NN output using the pre-computed matrices
            if A_out is None:
                A_out_torch = None
            else:
                A_out_torch = torch.Tensor(np.array([A_out[i, :]]))

            # CROWN was initialized knowing dynamics, no need to pass them here
            # (unless they've changed, e.g., time-varying At matrix)
            (
                A_out_xt1_max,
                A_out_xt1_min,
            ) = self.network.compute_bound_from_matrices(
                lower_A,
                lower_sum_b,
                upper_A,
                upper_sum_b,
                prev_state_max,
                prev_state_min,
                norm,
                A_out_torch,
                A_in=A_inputs,
                b_in=b_inputs,
            )
            # if self.dynamics.x_limits is not None:
            #     A_out_xt1_max = np.clip(A_out_xt1_max, self.dynamics.x_limits[i,0], self.dynamics.x_limits[i,1])
            #     A_out_xt1_min = np.clip(A_out_xt1_min, self.dynamics.x_limits[i,0], self.dynamics.x_limits[i,1])
            if isinstance(
                output_constraint, constraints.PolytopeConstraint
            ):
                bs[i] = A_out_xt1_max
            elif isinstance(output_constraint, constraints.LpConstraint):
                ranges[i, 0] = A_out_xt1_min
                ranges[i, 1] = A_out_xt1_max
            else:
                raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, constraints.LpConstraint):
            output_constraint.range = ranges
        else:
            raise NotImplementedError

        # Auxiliary info
        nn_matrices = {
            'lower_A': lower_A.data.numpy()[0],
            'upper_A': upper_A.data.numpy()[0],
            'lower_sum_b': lower_sum_b.data.numpy()[0],
            'upper_sum_b': upper_sum_b.data.numpy()[0],
        }

        return output_constraint, {'nn_matrices': nn_matrices}


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


    '''
    Inputs: 
        output_constraint: target set defining the set of final states to backproject from
        input_constriant: empty constraint object
        num_partitions: array of length nx defining how to partition for each dimension
        overapprox: flag to calculate over approximations of the BP set (set to true gives algorithm 1 in CDC 2022)

    Outputs: 
        input_constraint: one step BP set estimate
        info: dict with extra info (e.g. backreachable set, etc.)
    '''
    def get_one_step_backprojection_set(
        self,
        output_constraint,
        input_constraint,
        num_partitions=None,
        overapprox=False,
        collected_input_constraints=None,
        infos = None,
        refined = False,
        heuristic='guided', 
        all_lps=False,
        slow_cvxpy=False
    ):
        # Given an output_constraint, compute the input_constraint
        # that ensures that starting from within the input_constraint
        # will lead to a state within the output_constraint
        info = {}
        info['bp_set_partitions'] = []
        info['br_lp'] = []
        info['bp_lp'] = []
        info['crown'] = []
        info['other'] = []
        info['br_set_partitions'] = []
        t_start = time.time()
        

        # Extract elementwise bounds on xt1 from the lp-ball or polytope constraint
        if isinstance(output_constraint, constraints.PolytopeConstraint):
            A_t1 = output_constraint.A
            b_t1 = output_constraint.b[0]

            # Get bounds on each state from A_t1, b_t1
            try:
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(A_t1, b_t1)
                )
            except:
                # Sometimes get arithmetic error... this may fix it
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(
                        A_t1, b_t1 + 1e-6
                    )
                )
            xt1_max = np.max(vertices, 0)
            xt1_min = np.min(vertices, 0)
            norm = np.inf
        elif isinstance(output_constraint, constraints.LpConstraint):
            xt1_min = output_constraint.range[..., 0]
            xt1_max = output_constraint.range[..., 1]
            norm = output_constraint.p
            A_t1 = None
            b_t1 = None
        elif isinstance(output_constraint, constraints.RotatedLpConstraint):
            norm = np.inf
            xt1_min = output_constraint.bounding_box[:, 0]
            xt1_max = output_constraint.bounding_box[:, 1]
            theta = output_constraint.theta
            R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            pose = output_constraint.pose
            W = output_constraint.width
            A_t1 = None
            b_t1 = None
        else:
            raise NotImplementedError

        '''
        Step 1: 
        Find backreachable set: all the xt for which there is
        some u in U that leads to a state xt1 in output_constraint
        '''

        if self.dynamics.u_limits is None:
            print(
                "self.dynamics.u_limits is None ==> \
                The backreachable set is probably the whole state space. \
                Giving up."
                )
            raise NotImplementedError
        else:
            u_min = self.dynamics.u_limits[:, 0]
            u_max = self.dynamics.u_limits[:, 1]

        num_states = xt1_min.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]
        
        xt = cp.Variable(xt1_min.shape+(2,))
        ut = cp.Variable(num_control_inputs)

        A_t = np.eye(xt1_min.shape[0])
        num_facets = A_t.shape[0]
        coords = np.empty((2*num_states, num_states))

        # For each dimension of the output constraint (facet/lp-dimension):
        # compute a bound of the NN output using the pre-computed matrices
        xt = cp.Variable(xt1_min.shape)
        ut = cp.Variable(num_control_inputs)
        constrs = []
        constrs += [u_min <= ut]
        constrs += [ut <= u_max]

        # Included state limits to reduce size of backreachable sets by eliminating states that are not physically possible (e.g., maximum velocities)
        if self.dynamics.x_limits is not None:
            for state in self.dynamics.x_limits:
                constrs += [self.dynamics.x_limits[state][0] <= xt[state]]
                constrs += [xt[state] <= self.dynamics.x_limits[state][1]]


        # Dynamics must be satisfied
        if isinstance(output_constraint, constraints.LpConstraint):
            constrs += [self.dynamics.dynamics_step(xt,ut) <= xt1_max]
            constrs += [self.dynamics.dynamics_step(xt,ut) >= xt1_min]
        elif isinstance(output_constraint, constraints.RotatedLpConstraint):
            constrs += [R@self.dynamics.dynamics_step(xt,ut)-R@pose <= W]
            constrs += [R@self.dynamics.dynamics_step(xt,ut)-R@pose >= np.array([0, 0])]
        else:
            raise NotImplementedError

        A_t_i = cp.Parameter(num_states)
        obj = A_t_i@xt
        min_prob = cp.Problem(cp.Minimize(obj), constrs)
        max_prob = cp.Problem(cp.Maximize(obj), constrs)
        
        t_end = time.time()
        info['other'].append(t_end-t_start)
        for i in range(num_facets):
            t_start = time.time()
            A_t_i.value = A_t[i, :]
            min_prob.solve()
            t_end = time.time()
            info['br_lp'].append(t_end-t_start)

            coords[2*i, :] = xt.value
            t_start = time.time()
            max_prob.solve()
            t_end = time.time()
            info['br_lp'].append(t_end-t_start)

            coords[2*i+1, :] = xt.value
        t_start = time.time()
        # min/max of each element of xt in the backreachable set
        ranges = np.vstack([coords.min(axis=0), coords.max(axis=0)]).T

        backreachable_set = constraints.LpConstraint(range=ranges)
        info['backreachable_set'] = backreachable_set
        info['target_set'] = deepcopy(output_constraint)

        '''
        Step 2: 
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to input_constraint
        '''

        # Generate samples to get an initial underpapproximation of the BP set to start with
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
        t_end = time.time()
        info['other'].append(t_end-t_start)


        # Set up backprojection LPs
        problems = self.setup_LPs(nstep, 0, infos, collected_input_constraints)
        
        # Partition BR set
        element_list = self.partition(
            br_set_element, 
            problems,
            target_set=output_constraint, 
            dynamics=self.dynamics, 
            partition_budget=num_partitions, 
            heuristic=heuristic,
            nstep=nstep,
            info=info
        )
        t_start = time.time()
        # info['br_set_partitions'] = [constraints.LpConstraint(range=element.ranges) for element in element_list]

        # Set an empty Constraint that will get filled in
        if isinstance(output_constraint, constraints.PolytopeConstraint):
            input_constraint = constraints.PolytopeConstraint(A=[], b=[])
        elif isinstance(output_constraint, constraints.LpConstraint):
            input_constraint = constraints.LpConstraint(p=np.inf)
        elif isinstance(output_constraint, constraints.RotatedLpConstraint):
            input_constraint = constraints.RotatedLpConstraint()
        ut_max = -np.inf*np.ones(num_control_inputs)
        ut_min = np.inf*np.ones(num_control_inputs)
        xt_range_max = -np.inf*np.ones(xt1_min.shape)
        xt_range_min = np.inf*np.ones(xt1_min.shape)


        t_end = time.time()
        info['other'].append(t_end-t_start)

        if len(element_list) > 0:
            for element in element_list:
                if element.flag is not 'infeasible':
                    t_start = time.time()
                    ranges = element.ranges


                    # Because there might sensor noise, the NN could see a different
                    # set of states than the system is actually in
                    xt_min = ranges[..., 0]
                    xt_max = ranges[..., 1]
                    prev_state_max = torch.Tensor(np.array([xt_max]))
                    prev_state_min = torch.Tensor(np.array([xt_min]))
                    nn_input_max = prev_state_max
                    nn_input_min = prev_state_min
                    if self.dynamics.sensor_noise is not None:
                        raise NotImplementedError
                        # nn_input_max += torch.Tensor([self.dynamics.sensor_noise[:, 1]])
                        # nn_input_min += torch.Tensor([self.dynamics.sensor_noise[:, 0]])

                    # Compute the NN output matrices (for this xt partition)
                    num_control_inputs = self.dynamics.bt.shape[1]
                    C = torch.eye(num_control_inputs).unsqueeze(0)

                    t_end = time.time()
                    info['other'].append(t_end-t_start)
                    if hasattr(element, 'crown_bounds'):
                        t_start = time.time()
                        lower_A, lower_sum_b, upper_A, upper_sum_b = element.crown_bounds['lower_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_A'], element.crown_bounds['upper_sum_b']
                        t_end = time.time()
                        info['other'].append(t_end-t_start)
                    else:
                        t_start = time.time()
                        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
                            method_opt=self.method_opt,
                            norm=norm,
                            x_U=nn_input_max,
                            x_L=nn_input_min,
                            upper=True,
                            lower=True,
                            C=C,
                            return_matrices=True,
                        )
                        
                        # Extract numpy array from pytorch tensors
                        upper_A = upper_A.detach().numpy()[0]
                        lower_A = lower_A.detach().numpy()[0]
                        upper_sum_b = upper_sum_b.detach().numpy()[0]
                        lower_sum_b = lower_sum_b.detach().numpy()[0]
                        t_end = time.time()
                        info['crown'].append(t_end-t_start)   

                    if overapprox:
                        if nstep:
                            input_constraint, xt_range_min, xt_range_max, ut_min, ut_max = self.get_refined_one_step_backprojection_set_overapprox(
                                element,
                                ranges,
                                upper_A,
                                lower_A,
                                upper_sum_b,
                                lower_sum_b,
                                xt1_max,
                                xt1_min,
                                A_t,
                                xt_range_min,
                                xt_range_max,
                                ut_min,
                                ut_max,
                                input_constraint,
                                collected_input_constraints,
                                infos,
                                info,
                                problems,
                                all_lps,
                                slow_cvxpy
                            )
                        else:
                            input_constraint, xt_range_min, xt_range_max, ut_min, ut_max = self.get_one_step_backprojection_set_overapprox(
                                element,
                                xt1_max,
                                xt1_min,
                                A_t,
                                xt_range_min,
                                xt_range_max,
                                ut_min,
                                ut_max,
                                input_constraint,
                                collected_input_constraints,
                                info,
                                problems,
                                all_lps,
                                slow_cvxpy
                            )

                    else:
                        input_constraint = self.get_one_step_backprojection_set_underapprox(
                            ranges,
                            upper_A,
                            lower_A,
                            upper_sum_b,
                            lower_sum_b,
                            xt1_max,
                            xt1_min,
                            input_constraint
                        )

            # input_constraint should contain [A] and [b]
            # TODO: Store the detailed partitions in info
            x_overapprox = np.vstack((xt_range_min, xt_range_max)).T
            A_overapprox, b_overapprox = range_to_polytope(x_overapprox)
            input_constraint.A = [A_overapprox]
            input_constraint.b = [b_overapprox]
            t_start = time.time()
            lower_A_range, upper_A_range, lower_sum_b_range, upper_sum_b_range = self.network(
                    method_opt=self.method_opt,
                    norm=norm,
                    x_U=torch.Tensor(np.array([xt_range_max])),
                    x_L=torch.Tensor(np.array([xt_range_min])),
                    upper=True,
                    lower=True,
                    C=C,
                    return_matrices=True,
                )
            t_end = time.time()
            info['crown'].append(t_end-t_start)

            info['u_range'] = np.vstack((ut_min, ut_max)).T
            info['upper_A'] = upper_A_range.detach().numpy()[0]
            info['lower_A'] = lower_A_range.detach().numpy()[0]
            info['upper_sum_b'] = upper_sum_b_range.detach().numpy()[0]
            info['lower_sum_b'] = lower_sum_b_range.detach().numpy()[0]

            if isinstance(output_constraint, constraints.RotatedLpConstraint):
                # rangee = input_constraint.range
                info['mar_hull'], theta, xy, W, mar_vertices = find_MAR(info['bp_set_partitions'])
                # theta = 0
                # xy = rangee[:, 0]
                # W = rangee[:, 1] - rangee[:, 0]
                # mar_vertices = np.vstack((rangee.T, np.hstack((np.array([rangee[0, :]]).T, np.flip([rangee[1, :]]).T))))
                # import pdb; pdb.set_trace()
                # info['mar_hull'], theta, xy, W = find_MAR(info['bp_set_partitions'])
                # print(W)
                # print(theta)
                input_constraint = constraints.RotatedLpConstraint(pose=xy, theta=theta, W=W, vertices=mar_vertices)
                # import pdb; pdb.set_trace()
                t_start = time.time()
                lower_A_range, upper_A_range, lower_sum_b_range, upper_sum_b_range = self.network(
                        method_opt=self.method_opt,
                        norm=norm,
                        x_U=torch.Tensor(np.array([input_constraint.bounding_box[:, 1]])),
                        x_L=torch.Tensor(np.array([input_constraint.bounding_box[:, 0]])),
                        upper=True,
                        lower=True,
                        C=C,
                        return_matrices=True,
                    )
                t_end = time.time()
                info['crown'].append(t_end-t_start)

                info['u_range'] = np.vstack((ut_min, ut_max)).T
                info['upper_A'] = upper_A_range.detach().numpy()[0]
                info['lower_A'] = lower_A_range.detach().numpy()[0]
                info['upper_sum_b'] = upper_sum_b_range.detach().numpy()[0]
                info['lower_sum_b'] = lower_sum_b_range.detach().numpy()[0]

            # print(info['upper_A'])
            # print(info['lower_A'])
            # print(info['upper_sum_b'])
            # print(info['lower_sum_b'])
        else:
            input_constraint = constraints.LpConstraint(range=np.hstack((np.inf*np.ones((num_states,1)), -np.inf*np.ones((num_states,1)))))
        
        return input_constraint, info

    '''
    Inputs: 
        ranges: section of backreachable set
        upper_A, lower_A, upper_sum_b, lower_sum_b: CROWN variables
        xt1max, xt1min: target set max values
        input_constraint: empty constraint object
    Outputs: 
        input_constraint: one step BP set under-approximation
    '''
    def get_one_step_backprojection_set_underapprox(
        self,
        ranges,
        upper_A,
        lower_A,
        upper_sum_b,
        lower_sum_b,
        xt1_max,
        xt1_min,
        input_constraint
    ):
        # For our under-approximation, refer to the Access21 paper.

        # The NN matrices define three types of constraints:
        # - NN's resulting lower bnds on xt1 >= lower bnds on xt1
        # - NN's resulting upper bnds on xt1 <= upper bnds on xt1
        # - NN matrices are only valid within the partition
        A_NN, b_NN = range_to_polytope(ranges)
        A_ = np.vstack([
                (self.dynamics.At+self.dynamics.bt@upper_A),
                -(self.dynamics.At+self.dynamics.bt@lower_A),
                A_NN
            ])
        b_ = np.hstack([
                xt1_max - self.dynamics.bt@upper_sum_b,
                -xt1_min + self.dynamics.bt@lower_sum_b,
                b_NN
                ])

        # If those constraints to a non-empty set, then add it to
        # the list of input_constraints. Otherwise, skip it.
        try:
            pypoman.polygon.compute_polygon_hull(A_, b_+1e-10)
            input_constraint.A.append(A_)
            input_constraint.b.append(b_)
        except:
            pass

        return input_constraint

    '''
    CDC 2022 Paper Alg 1

    Inputs: 
        ranges: section of backreachable set
        upper_A, lower_A, upper_sum_b, lower_sum_b: CROWN variables defining upper and lower affine bounds
        xt1max, xt1min: target set max values
        input_constraint: empty constraint object
        xt_range_min, xt_range_max: min and max x values current overall BP set estimate (expanded as new partitions are analyzed)
        ut_min, ut_max: min and max u values current overall BP set estimate (expanded as new partitions are analyzed)
    Outputs: 
        input_constraint: one step BP set under-approximation
        xt_range_min, xt_range_max: min and max x values current overall BP set estimate (expanded as new partitions are analyzed)
        ut_min, ut_max: min and max u values current overall BP set estimate (expanded as new partitions are analyzed)
    '''
    def get_one_step_backprojection_set_overapprox(
        self,
        element,
        xt1_max,
        xt1_min,
        A_t,
        xt_range_min,
        xt_range_max,
        ut_min,
        ut_max,
        input_constraint,
        collected_input_constraints,
        info,
        problems,
        all_lps,
        slow_cvxpy
    ):
        t_start = time.time()
        ranges = element.ranges
        lower_A, lower_sum_b, upper_A, upper_sum_b = element.crown_bounds['lower_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_A'], element.crown_bounds['upper_sum_b']

        prob = problems[0]
        params = problems[2]

        xt_min = ranges[..., 0]
        xt_max = ranges[..., 1]

        num_states = xt1_min.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]

        # An over-approximation of the backprojection set is the set of:
        # all xt s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
        ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)

        A_NN, b_NN = range_to_polytope(ranges)
        
        if slow_cvxpy:
            xt = cp.Variable(xt1_min.shape)
            ut = cp.Variable(num_control_inputs)
            constrs = []

            # Constraints to ensure that xt stays within the backreachable set
            constrs += [xt_min <= xt]
            constrs += [xt <= xt_max]

            # Constraints to ensure that ut satisfies the affine bounds
            constrs += [lower_A@xt+lower_sum_b <= ut]
            constrs += [ut <= upper_A@xt+upper_sum_b]

            # Constraints to ensure xt reaches the target set given ut
            constrs += [self.dynamics.dynamics_step(xt, ut) <= xt1_max]
            constrs += [self.dynamics.dynamics_step(xt, ut) >= xt1_min]

            A_t_i = cp.Parameter(num_states)
            obj = A_t_i@xt
            prob = cp.Problem(cp.Maximize(obj), constrs)
        else:
            params['lower_A'].value = lower_A
            params['upper_A'].value = upper_A
            params['lower_sum_b'].value = lower_sum_b
            params['upper_sum_b'].value = upper_sum_b
            
            params['xt_min'].value = element.ranges[:,0]
            params['xt_max'].value = element.ranges[:,1]

            params['xt1_min'].value = xt1_min
            params['xt1_max'].value = xt1_max

        # Solve optimization problem (min and max) for each state
        A_t_ = np.vstack([A_t, -A_t])

        # Check for which states the optimization can possibly make the BP set more conservative
        min_idx = xt_min < xt_range_min - 1e-5
        max_idx = xt_max > xt_range_max + 1e-5
        idx1 = np.hstack((max_idx, min_idx))
        
        # Flag to use naive partitioning
        if all_lps:
            idx1 = np.ones(2*num_states)

        # Check which state (if any) was already optimized during the BR set calculation
        idx2 = np.invert((A_t_ == element.A_t_br).all(axis=1))
        idx = np.vstack((idx1, idx2)).all(axis=0)

        # Only optimize over states without constraints; doesn't have an effect with guided partitioner
        # if self.dynamics.x_limits is not None:
        #     for key in self.dynamics.x_limits:
        #         idx[key] = False
        #         idx[key+num_states] = False

        A_ = A_t_
        b_ = b_NN
        t_end = time.time()
        info['other'].append(t_end-t_start)

        for i, row in enumerate(A_t_):
            if idx[i]:
                if slow_cvxpy:
                    A_t_i.value = A_t_[i, :]
                else:
                    params['A_t'].value = A_t_[i, :]
                t_start = time.time()
                prob.solve()
                t_end = time.time()
                info['bp_lp'].append(t_end-t_start)
                b_[i] = prob.value

                # print('status: {}; time: {}'.format(prob.status,t_end-t_start))
            
        t_start = time.time()

        # This cell of the backprojection set is upper-bounded by the
        # cell of the backreachable set that we used in the NN relaxation
        # ==> the polytope is the intersection (i.e., concatenation)
        # of the polytope used for relaxing the NN and the soln to the LP
        A_stack = np.vstack([A_, A_NN])
        b_stack = np.hstack([b_, b_NN])

        # Add newly calculated BP region from partioned backreachable set to overall BP set estimate
        if isinstance(input_constraint, constraints.LpConstraint) or isinstance(input_constraint, constraints.RotatedLpConstraint):
            b_max = b_[0:int(len(b_)/2)]
            b_min = -b_[int(len(b_)/2):int(len(b_))]

            # import pdb;  pdb.set_trace()
            # if any(b_max > xt_range_max) or any(b_min < xt_range_min):
            info['bp_set_partitions'].append(constraints.LpConstraint(range=np.array([b_min, b_max]).T)) 

            ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
            ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)

            ut_min = np.minimum(ut_min, ut_min_candidate)
            ut_max = np.maximum(ut_max, ut_max_candidate)

            xt_range_max = np.max((xt_range_max, b_max),axis=0)
            xt_range_min = np.min((xt_range_min, b_min),axis=0)

            input_constraint.range = np.array([xt_range_min,xt_range_max]).T

        elif isinstance(input_constraint, constraints.PolytopeConstraint):
            # Only add that polytope to the list if it's non-empty
            vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
            if len(vertices) > 0:
                xt_max_candidate = np.max(vertices, axis=0)
                xt_min_candidate = np.min(vertices, axis=0)
                temp_max = xt_range_max
                temp_min = xt_range_min
                xt_range_max = np.maximum(xt_range_max, xt_max_candidate)
                xt_range_min = np.minimum(xt_range_min, xt_min_candidate)

                ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
                ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)

                ut_min = np.minimum(ut_min, ut_min_candidate)
                ut_max = np.maximum(ut_max, ut_max_candidate)
                
                input_constraint.A.append(A_)
                input_constraint.b.append(b_)
        else:
            raise NotImplementedError
        t_end = time.time()
        info['other'].append(t_end-t_start)

        return input_constraint, xt_range_min, xt_range_max, ut_min, ut_max



    def get_refined_one_step_backprojection_set_overapprox(
        self,
        element,
        ranges,
        upper_A,
        lower_A,
        upper_sum_b,
        lower_sum_b,
        xt1_max,
        xt1_min,
        A_t,
        xt_range_min,
        xt_range_max,
        ut_min,
        ut_max,
        input_constraint,
        collected_input_constraints,
        infos,
        info,
        problems,
        all_lps,
        slow_cvxpy
    ):  
        t_start = time.time()
        ranges = element.ranges
        lower_A, lower_sum_b, upper_A, upper_sum_b = element.crown_bounds['lower_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_A'], element.crown_bounds['upper_sum_b']
        prob = problems[0]
        params = problems[2]
        # infos = None
        # import pdb; pdb.set_trace()
        
        xt_min = ranges[..., 0]
        xt_max = ranges[..., 1]

        num_states = xt1_min.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]

        # An over-approximation of the backprojection set is the set of:
        # all xt s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
        ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)


        num_facets = 2*num_states
        num_steps = len(collected_input_constraints)
        
        A_NN, b_NN = range_to_polytope(ranges)

        if slow_cvxpy:
            xt = cp.Variable((num_states, num_steps+1))
            ut = cp.Variable((num_control_inputs, num_steps))
            constrs = []

            # x_{t=0} \in this partition of 0-th backreachable set
            constrs += [xt_min <= xt[:, 0]]
            constrs += [xt[:, 0] <= xt_max]

            # # if self.dynamics.x_limits is not None:
            # #     x_llim = self.dynamics.x_limits[:, 0]
            # #     x_ulim = self.dynamics.x_limits[:, 1]
            

            # # # Each xt must be in a backprojection overapprox
            # # for t in range(num_steps - 1):
            # #     A, b = input_constraints[t].A[0], input_constraints[t].b[0]
            # #     constrs += [A@xt[:, t+1] <= b]

            # # x_{t=T} must be in target set
            # if isinstance(collected_input_constraints[0], constraints.LpConstraint):
            #     goal_set_A, goal_set_b = range_to_polytope(collected_input_constraints[0].range)
            # elif isinstance(collected_input_constraints[0], constraints.PolytopeConstraint):
            #     goal_set_A, goal_set_b = collected_input_constraints[0].A, collected_input_constraints[0].b[0]
            # constrs += [goal_set_A@xt[:, -1] <= goal_set_b]

            # # Each ut must not exceed CROWN bounds
            # # import pdb; pdb.set_trace()
            # for t in range(num_steps):
            #     # if t == 0:
            #     #     lower_A, upper_A, lower_sum_b, upper_sum_b = self.get_crown_matrices(xt_min, xt_max, num_control_inputs)
            #     # else:
            #     # Gather CROWN bounds for full backprojection overapprox
            #     if t > 0:
            #         # import pdb; pdb.set_trace()
            #         upper_A = infos[-t]['upper_A']
            #         lower_A = infos[-t]['lower_A']
            #         upper_sum_b = infos[-t]['upper_sum_b']
            #         lower_sum_b = infos[-t]['lower_sum_b']

            #     # u_t bounded by CROWN bounds
            #     constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
            #     constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]

            # # import pdb; pdb.set_trace()
            # # Each xt must fall in the original backprojection
            # for t in range(1,num_steps):
            #     constrs += [collected_input_constraints[-t].range[:,0] <= xt[:,t]]
            #     constrs += [xt[:,t] <= collected_input_constraints[-t].range[:,1]]


            # # x_t and x_{t+1} connected through system dynamics
            # for t in range(num_steps):
            #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]

            #     # if self.dynamics.x_limits is not None:
            #     #     x_llim = self.dynamics.x_limits[:, 0]
            #     #     x_ulim = self.dynamics.x_limits[:, 1]
            #     #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) <= x_ulim]
            #     #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) >= x_llim]

            # # u_t satisfies control limits (TODO: Necessary? CROWN should account for these)
            # # for t in range(num_steps):
            # #     constrs += [-1 <= ut[:, t]]
            # #     constrs += [1 >= ut[:, t]]

            # A_t_i = cp.Parameter(num_states)
            # obj = A_t_i@xt[:, 0]
            # prob = cp.Problem(cp.Maximize(obj), constrs)


            for t in range(num_steps):
                # Gather CROWN bounds and previous BP bounds
                if t > 0:
                    upper_A = infos[-t]['upper_A']
                    lower_A = infos[-t]['lower_A']
                    upper_sum_b = infos[-t]['upper_sum_b']
                    lower_sum_b = infos[-t]['lower_sum_b']

                    # Each xt must fall in the original backprojection
                    constrs += [collected_input_constraints[-t].range[:,0] <= xt[:, t]]
                    constrs += [xt[:, t] <= collected_input_constraints[-t].range[:,1]]

                # u_t bounded by CROWN bounds
                constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
                constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]

                

                # x_t and x_{t+1} connected through system dynamics
                constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]
            
            A_t_i = cp.Parameter(num_states)
            obj = A_t_i@xt[:,0]
            prob = cp.Problem(cp.Maximize(obj), constrs)
        else:
            ##### Solve Problem Parametrically #####
            params['lower_A'][0].value = lower_A
            params['upper_A'][0].value = upper_A
            params['lower_sum_b'][0].value = lower_sum_b
            params['upper_sum_b'][0].value = upper_sum_b
            
            params['xt_min'][0].value = element.ranges[:, 0]
            params['xt_max'][0].value = element.ranges[:, 1]


        # Solve optimization problem (min and max) for each state
        A_t_ = np.vstack([A_t, -A_t])
        
        min_idx = xt_min < xt_range_min
        max_idx = xt_max > xt_range_max
        idx1 = np.hstack((max_idx, min_idx))
        
        # Flag to use naive partitioning
        if all_lps:
            idx1 = np.ones(2*num_states)

        # Check which state (if any) was already optimized during the BR set calculation
        idx2 = np.invert((A_t_ == element.A_t_br).all(axis=1))
        idx = np.vstack((idx1, idx2)).all(axis=0)

        # Only optimize over states without constraints; doesn't have an effect with guided partitioner
        # if self.dynamics.x_limits is not None:
        #     for key in self.dynamics.x_limits:
        #         idx[key] = False
        #         idx[key+num_states] = False


        A_ = A_t_
        b_ = b_NN
        t_end = time.time()
        info['other'].append(t_end-t_start)

        for i, row in enumerate(A_t_):
            if idx[i]:
                if slow_cvxpy:
                    A_t_i.value = A_t_[i, :]
                else:
                    params['A_t'].value = A_t_[i, :]
                t_start = time.time()
                try:
                    # TODO: Make this robust to ECOS failure
                    prob.solve()
                    # prob2.solve()
                    t_end = time.time()
                    info['bp_lp'].append(t_end-t_start)
                    b_[i] = prob.value
                    # print(prob.status)
                except:
                    print('ECOS failed')
                    print(collected_input_constraints[0].range)
                    prob.solve(solver=cp.OSQP, verbose=True)
                    import pdb; pdb.set_trace()
                    pass
                
                # print('{} vs {}'.format(prob.value, prob2.value))

                # print('status: {}; time: {}'.format(prob.status,t_end-t_start))




        # A_facets = np.vstack([A_t, -A_t])
        # A_facets_i = cp.Parameter(num_states)
        # obj = A_facets_i@xt[:, 0]
        # prob = cp.Problem(cp.Maximize(obj), constrs)
        # A_ = A_facets
        # b_ = np.empty(num_facets)
        # t_end = time.time()
        # info['other'].append(t_end-t_start)
        # for i in range(num_facets):
        #     t_start = time.time()
        #     A_facets_i.value = A_facets[i, :]
        #     prob.solve()
        #     t_end = time.time()
        #     info['bp_lp'].append(t_end-t_start)
        #     # prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
        #     b_[i] = prob.value

        # This cell of the backprojection set is upper-bounded by the
        # cell of the backreachable set that we used in the NN relaxation
        # ==> the polytope is the intersection (i.e., concatenation)
        # of the polytope used for relaxing the NN and the soln to the LP
        t_start = time.time()
        A_stack = np.vstack([A_, A_NN])
        b_stack = np.hstack([b_, b_NN])


        # Add newly calculated BP region from partioned backreachable set to overall BP set estimate
        if isinstance(input_constraint, constraints.LpConstraint) or isinstance(input_constraint, constraints.RotatedLpConstraint):
            b_max = b_[0:int(len(b_)/2)]
            b_min = -b_[int(len(b_)/2):int(len(b_))]

            # if any(b_max > xt_range_max) or any(b_min < xt_range_min):
            info['bp_set_partitions'].append(constraints.LpConstraint(range=np.array([b_min, b_max]).T))

            ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
            ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)

            ut_min = np.minimum(ut_min, ut_min_candidate)
            ut_max = np.maximum(ut_max, ut_max_candidate)

            # print('--------------start--------------')
            # print(idx)
            # print('{} vs {}'.format(xt_range_max, b_max))
            # print('{} vs {}'.format(xt_range_min, b_min))
            # print('---------------------------------')
            xt_range_max = np.max((xt_range_max, b_max),axis=0)
            xt_range_min = np.min((xt_range_min, b_min),axis=0)
            # print('{} vs {}'.format(xt_range_max, b_max))
            # print('{} vs {}'.format(xt_range_min, b_min))
            # import pdb; pdb.set_trace()

            input_constraint.range = np.array([xt_range_min,xt_range_max]).T

        elif isinstance(input_constraint, constraints.PolytopeConstraint):
            # Only add that polytope to the list if it's non-empty
            vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
            if len(vertices) > 0:
                # pypoman.polygon.compute_polygon_hull(A_stack, b_stack+1e-10)
                # vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                
                xt_max_candidate = np.max(vertices, axis=0)
                xt_min_candidate = np.min(vertices, axis=0)
                xt_range_max = np.maximum(xt_range_max, xt_max_candidate)
                xt_range_min = np.minimum(xt_range_min, xt_min_candidate)

                ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
                ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)

                ut_min = np.minimum(ut_min, ut_min_candidate)
                ut_max = np.maximum(ut_max, ut_max_candidate)
                
                input_constraint.A.append(A_)
                input_constraint.b.append(b_)
        else:
            raise NotImplementedError
        
        t_end = time.time()
        info['other'].append(t_end-t_start)

        return input_constraint, xt_range_min, xt_range_max, ut_min, ut_max

    def get_crown_matrices(self, xt_min, xt_max, num_control_inputs):
        nn_input_max = torch.Tensor([xt_max])
        nn_input_min = torch.Tensor([xt_min])

        # Compute the NN output matrices (for this xt partition)
        C = torch.eye(num_control_inputs).unsqueeze(0)
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
            method_opt=self.method_opt,
            norm=np.inf,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )
        upper_A = upper_A.detach().numpy()[0]
        lower_A = lower_A.detach().numpy()[0]
        upper_sum_b = upper_sum_b.detach().numpy()[0]
        lower_sum_b = lower_sum_b.detach().numpy()[0]

        return lower_A, upper_A, lower_sum_b, upper_sum_b

    def setup_LPs(self, nstep=False, modifier=0, infos=None, collected_input_constraints=None):
        num_states = self.dynamics.At.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]
        if not nstep:
            xt = cp.Variable(num_states)
            ut = cp.Variable(num_control_inputs)

            lower_A = cp.Parameter((num_control_inputs, num_states))
            upper_A = cp.Parameter((num_control_inputs, num_states))
            lower_sum_b = cp.Parameter(num_control_inputs)
            upper_sum_b = cp.Parameter(num_control_inputs)
            if isinstance(collected_input_constraints[0], constraints.LpConstraint) or isinstance(collected_input_constraints[0], constraints.RotatedLpConstraint):
                xt_min = cp.Parameter(num_states)
                xt_max = cp.Parameter(num_states)

                xt1_min = cp.Parameter(num_states)
                xt1_max = cp.Parameter(num_states)

                A_t = cp.Parameter(num_states)

                params = {
                    'lower_A': lower_A,
                    'upper_A': upper_A,
                    'lower_sum_b': lower_sum_b,
                    'upper_sum_b': upper_sum_b,
                    'xt_min': xt_min,
                    'xt_max': xt_max,
                    'xt1_min': xt1_min,
                    'xt1_max': xt1_max,
                    'A_t': A_t
                }

                constrs = []
                constrs += [lower_A@xt + lower_sum_b <= ut]
                constrs += [ut <= upper_A@xt + upper_sum_b]

                # Included state limits to reduce size of backreachable sets by eliminating states that are not physically possible (e.g., maximum velocities)
                constrs += [xt_min <= xt]
                constrs += [xt <= xt_max]


                # Dynamics must be satisfied

                constrs += [self.dynamics.dynamics_step(xt, ut) <= xt1_max]
                constrs += [self.dynamics.dynamics_step(xt, ut) >= xt1_min]
            
            elif isinstance(collected_input_constraints[0], constraints.RotatedLpConstraint):
                xt_min = cp.Parameter(num_states)
                xt_max = cp.Parameter(num_states)

                R = cp.Parameter((num_states, num_states))
                pose = cp.Parameter(num_states)
                W = cp.Parameter(num_states)

                A_t = cp.Parameter(num_states)

                params = {
                    'lower_A': lower_A,
                    'upper_A': upper_A,
                    'lower_sum_b': lower_sum_b,
                    'upper_sum_b': upper_sum_b,
                    'xt_min': xt_min,
                    'xt_max': xt_max,
                    'R': R,
                    'W': W,
                    'pose': pose,
                    'A_t': A_t
                }

                constrs = []
                constrs += [lower_A@xt + lower_sum_b <= ut]
                constrs += [ut <= upper_A@xt + upper_sum_b]

                # Included state limits to reduce size of backreachable sets by eliminating states that are not physically possible (e.g., maximum velocities)
                constrs += [xt_min <= xt]
                constrs += [xt <= xt_max]


                # Dynamics must be satisfied

                constrs += [R@self.dynamics.dynamics_step(xt,ut)-R@pose <= W]
                constrs += [R@self.dynamics.dynamics_step(xt,ut)-R@pose >= np.array([0, 0])]
            
            obj = A_t@xt
            min_prob = cp.Problem(cp.Minimize(obj), constrs)
            max_prob = cp.Problem(cp.Maximize(obj), constrs)
        else:
            if isinstance(collected_input_constraints[0], constraints.LpConstraint):
                num_steps = len(collected_input_constraints)
                xt = cp.Variable((num_states, num_steps+1))
                ut = cp.Variable((num_control_inputs, num_steps))


                lower_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
                upper_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
                lower_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]
                upper_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]

                xt_min = [cp.Parameter(num_states) for i in range(num_steps+1)]
                xt_max = [cp.Parameter(num_states) for i in range(num_steps+1)]


                A_t = cp.Parameter(num_states)

                params = {
                    'lower_A': lower_A_list,
                    'upper_A': upper_A_list,
                    'lower_sum_b': lower_sum_b_list,
                    'upper_sum_b': upper_sum_b_list,
                    'xt_min': xt_min,
                    'xt_max': xt_max,
                    'A_t': A_t
                }


                constrs = []

                for t in range(num_steps):
                    # Gather CROWN bounds and previous BP bounds
                    if t > 0:
                        upper_A_list[t].value = infos[-t-modifier]['upper_A']
                        lower_A_list[t].value = infos[-t-modifier]['lower_A']
                        upper_sum_b_list[t].value = infos[-t-modifier]['upper_sum_b']
                        lower_sum_b_list[t].value = infos[-t-modifier]['lower_sum_b']

                        xt_min[t].value = collected_input_constraints[-t-modifier].range[:, 0]
                        xt_max[t].value = collected_input_constraints[-t-modifier].range[:, 1]

                    # u_t bounded by CROWN bounds
                    constrs += [lower_A_list[t]@xt[:, t]+lower_sum_b_list[t] <= ut[:, t]]
                    constrs += [ut[:, t] <= upper_A_list[t]@xt[:, t]+upper_sum_b_list[t]]

                    # Each xt must fall in the original backprojection
                    constrs += [xt_min[t] <= xt[:, t]]
                    constrs += [xt[:, t] <= xt_max[t]]

                    # x_t and x_{t+1} connected through system dynamics
                    constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]

            elif isinstance(collected_input_constraints[0], constraints.RotatedLpConstraint):

                num_steps = len(collected_input_constraints)
                xt = cp.Variable((num_states, num_steps+1))
                ut = cp.Variable((num_control_inputs, num_steps))


                lower_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
                upper_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
                lower_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]
                upper_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]

                xt_min = [cp.Parameter(num_states)]
                xt_max = [cp.Parameter(num_states)]

                R = [cp.Parameter((num_states, num_states)) for i in range(num_steps+1)]
                pose = [cp.Parameter(num_states) for i in range(num_steps+1)] 
                W = [cp.Parameter(num_states) for i in range(num_steps+1)]


                A_t = cp.Parameter(num_states)

                params = {
                    'lower_A': lower_A_list,
                    'upper_A': upper_A_list,
                    'lower_sum_b': lower_sum_b_list,
                    'upper_sum_b': upper_sum_b_list,
                    'xt_min': xt_min,
                    'xt_max': xt_max,
                    'A_t': A_t
                }


                constrs = []

                constrs += [xt_min[0] <= xt[:, 0]]
                constrs += [xt[:, 0] <= xt_max[0]]

                for t in range(num_steps):
                    # Gather CROWN bounds and previous BP bounds
                    if t > 0:
                        upper_A_list[t].value = infos[-t-modifier]['upper_A']
                        lower_A_list[t].value = infos[-t-modifier]['lower_A']
                        upper_sum_b_list[t].value = infos[-t-modifier]['upper_sum_b']
                        lower_sum_b_list[t].value = infos[-t-modifier]['lower_sum_b']

                        theta = collected_input_constraints[-t-modifier].theta

                        R[t].value = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
                        W[t].value = collected_input_constraints[-t-modifier].width
                        pose[t].value = collected_input_constraints[-t-modifier].pose

                        # Each xt must fall in the previously calculated (rotated) backprojection
                        constrs += [R[t]@xt[:, t] <= W[t] + R[t]@pose[t]]
                        constrs += [R[t]@xt[:, t] >= R[t]@pose[t]]

                    # u_t bounded by CROWN bounds
                    constrs += [lower_A_list[t]@xt[:, t]+lower_sum_b_list[t] <= ut[:, t]]
                    constrs += [ut[:, t] <= upper_A_list[t]@xt[:, t]+upper_sum_b_list[t]]

                    # x_t and x_{t+1} connected through system dynamics
                    constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]
            obj = A_t@xt[:, 0]
            min_prob = cp.Problem(cp.Minimize(obj), constrs)
            max_prob = cp.Problem(cp.Maximize(obj), constrs)

        return max_prob, min_prob, params



class ClosedLoopIBPPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        raise NotImplementedError
        # TODO: Write nn_bounds.py:BoundClosedLoopController:interval_range
        # (using bound_layers.py:BoundSequential:interval_range)
        self.method_opt = "interval_range"
        self.params = {}


class ClosedLoopCROWNPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": False, "zero-lb": True}
        # self.params = {"same-slope": False}


class ClosedLoopCROWNLPPropagator(ClosedLoopCROWNPropagator):
    # Same as ClosedLoopCROWNPropagator but don't allow the
    # use of closed-form soln to the optimization, even if it's possible
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.params['try_to_use_closed_form'] = False


class ClosedLoopFastLinPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": True}


class ClosedLoopCROWNNStepPropagator(ClosedLoopCROWNPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )

    def get_reachable_set(self, input_constraint, output_constraint, t_max):

        output_constraints, infos = super().get_reachable_set(
            input_constraint, output_constraint, t_max)

        # Symbolically refine all N steps
        tightened_output_constraints = []
        tightened_infos = deepcopy(infos)
        N = len(output_constraints)
        for t in range(2, N + 1):

            # N-step analysis accepts as input:
            # [1st output constraint, {2,...,t} tightened output constraints,
            #  dummy output constraint]
            output_constraints_better = deepcopy(output_constraints[:1])
            output_constraints_better += deepcopy(tightened_output_constraints)
            output_constraints_better += deepcopy(output_constraints[:1]) # just a dummy

            output_constraint, info = self.get_N_step_reachable_set(
                input_constraint, output_constraints_better, infos['per_timestep'][:t]
            )
            tightened_output_constraints.append(output_constraint)
            tightened_infos['per_timestep'][t-1] = info

        output_constraints = output_constraints[:1] + tightened_output_constraints

        # Do N-step "symbolic" refinement
        # output_constraint, new_infos = self.get_N_step_reachable_set(
        #     input_constraint, output_constraints, infos['per_timestep']
        # )
        # output_constraints[-1] = output_constraint
        # tightened_infos = infos  # TODO: fix this...

        return output_constraints, tightened_infos

    def get_backprojection_set(
        self,
        output_constraints,
        input_constraint,
        t_max=1,
        num_partitions=None,
        overapprox=True,
        refined=False,
        heuristic='guided', 
        all_lps=False,
        slow_cvxpy=False
    ):
        input_constraint_list = []
        tightened_infos_list = []
        if not isinstance(output_constraints, list):
            output_constraint_list = [deepcopy(output_constraints)]
        else:
            output_constraint_list = deepcopy(output_constraints)

        # Initialize backprojections and u bounds (in infos)
        input_constraints, infos = super().get_backprojection_set(
            output_constraint_list, 
            input_constraint, 
            t_max,
            num_partitions=num_partitions, 
            overapprox=overapprox,
            refined=refined,
            heuristic=heuristic,
            all_lps=all_lps,
            slow_cvxpy=slow_cvxpy
        )
        # import pdb; pdb.set_trace()
        for i in range(len(output_constraint_list)):
            tightened_input_constraints, tightened_infos = self.get_single_target_N_step_backprojection_set(output_constraint_list[i], input_constraints[i], infos[i], t_max=t_max, num_partitions=num_partitions, overapprox=overapprox, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy)

            input_constraint_list.append(deepcopy(tightened_input_constraints))
            tightened_infos_list.append(deepcopy(tightened_infos))

        return input_constraint_list, tightened_infos_list


    def get_single_target_N_step_backprojection_set(
        self,
        output_constraint,
        input_constraints,
        infos,
        t_max=1,
        num_partitions=None,
        overapprox=True,
        heuristic='guided', 
        all_lps=False,
        slow_cvxpy=False
    ):
        # Symbolically refine all N steps
        tightened_input_constraints = []
        tightened_infos = deepcopy(infos)
        N = len(input_constraints)

        for t in range(2, N + 1):

            # N-step analysis accepts as input:
            # [1st input constraint, {2,...,t} tightened input constraints,
            #  dummy input constraint]
            input_constraints_better = deepcopy(input_constraints[:1])
            input_constraints_better += deepcopy(tightened_input_constraints)
            input_constraints_better += deepcopy([input_constraints[t-1]])  # the first pass' backproj overapprox

            input_constraint, info = self.get_N_step_backprojection_set(
                output_constraint, input_constraints_better, infos['per_timestep'][:t], overapprox=overapprox, num_partitions=num_partitions, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy
            )
            tightened_input_constraints.append(input_constraint)
            tightened_infos['per_timestep'][t-1] = info

        input_constraints = input_constraints[:1] + tightened_input_constraints

        return input_constraints, tightened_infos

    '''
    Inputs: 
        output_constraint: target set defining the set of final states to backproject from
        input_constriant: empty constraint object
        num_partitions: array of length nx defining how to partition for each dimension
        overapprox: flag to calculate over approximations of the BP set (set to true gives algorithm 1 in CDC 2022)

    Outputs: 
        input_constraint: one step BP set estimate
        info: dict with extra info (e.g. backreachable set, etc.)
    '''
    def get_N_step_backprojection_set(
        self,
        output_constraint,
        input_constraints,
        infos,
        num_partitions=None,
        overapprox=True,
        heuristic='guided', 
        all_lps=False,
        slow_cvxpy=False
    ):

        if overapprox:
            input_constraint, info = self.get_N_step_backprojection_set_overapprox(
                output_constraint,
                input_constraints,
                infos,
                num_partitions=num_partitions,
                heuristic=heuristic, 
                all_lps=all_lps,
                slow_cvxpy=slow_cvxpy
            )
        else:
            input_constraint, info = self.get_N_step_backprojection_set_underapprox()

        return input_constraint, info

    def get_N_step_backprojection_set_underapprox(self):
        raise NotImplementedError

    def get_N_step_backprojection_set_overapprox(
        self,
        output_constraint,
        input_constraints,
        infos,
        num_partitions=None,
        heuristic='guided', 
        all_lps=False,
        slow_cvxpy=False
    ):
        if 'nstep_bp_lp' in infos[-1]:
            print('something is wrong')
        infos[-1]['nstep_bp_lp'] = []
        infos[-1]['nstep_other'] = []
        infos[-1]['nstep_crown'] = []
        t_start = time.time()
        # import pdb; pdb.set_trace()
        # Get range of "earliest" backprojection overapprox
        vertices = np.array(
            pypoman.duality.compute_polytope_vertices(
                input_constraints[-1].A[0],
                input_constraints[-1].b[0]
            )
        )
        if isinstance(output_constraint, constraints.LpConstraint):
            tightened_constraint = constraints.LpConstraint(p=np.inf)
        elif isinstance(output_constraint, constraints.PolytopeConstraint):
            tightened_constraint = constraints.PolytopeConstraint(A=[], b=[])
        else:
            raise NotImplementedError
        ranges = np.vstack([vertices.min(axis=0), vertices.max(axis=0)]).T
        input_range = ranges

        ###### Partitioning parameters ##############
        # partition_budget=1000
        # heuristic='guided'
        # heuristic = 'uniform'
        #############################################

        # if using the refined flag, we want to use samples that actually reach the target set
        x_samples_inside_backprojection_set = self.dynamics.get_true_backprojection_set(input_constraints[-1], output_constraint, t_max=len(input_constraints), controller=self.network)

        br_set_element = elements.OptGuidedElement(ranges, self.network, samples=x_samples_inside_backprojection_set[:,0,:])

        # element_list = self.partition(
        #     br_set_element, 
        #     target_set=input_constraints[-2], 
        #     dynamics=self.dynamics, 
        #     partition_budget=partition_budget, 
        #     heuristic=heuristic,
        #     nstep=True
        # )

        # Set up backprojection LPs
        problems = self.setup_LPs(True, 1, infos, input_constraints)
        
        # import pdb; pdb.set_trace()
        # Partition BR set
        element_list = self.partition(
            br_set_element, 
            problems,
            target_set=output_constraint, 
            dynamics=self.dynamics, 
            partition_budget=num_partitions, 
            heuristic=heuristic,
            nstep=True,
            info=infos[-1]
        )
        
        

        # Partition "earliest" backproj overapprox
        # if num_partitions is None:
        #     num_partitions = np.array([10, 10])
        # slope = np.divide(
        #     (input_range[..., 1] - input_range[..., 0]), num_partitions
        # )

        num_states = self.dynamics.At.shape[1]
        num_control_inputs = self.dynamics.bt.shape[1]
        num_steps = len(input_constraints)
        xt_range_max = -np.inf*np.ones(num_states)
        xt_range_min = np.inf*np.ones(num_states)
        A_facets = np.vstack([np.eye(num_states), -np.eye(num_states)])
        num_facets = A_facets.shape[0]
        input_shape = input_range.shape[:-1]
        t_end = time.time()
        infos[-1]['nstep_other'].append(t_end-t_start)

        # Iterate through each partition
        # for element in product(
        #     *[range(num) for num in num_partitions.flatten()]
        # ):
        # import pdb; pdb.set_trace()
        # print('hmmmmmmmmmmmmmmm')
        for element in element_list:
            if element.flag != 'infeasible':
                lower_A, lower_sum_b, upper_A, upper_sum_b = element.crown_bounds['lower_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_A'], element.crown_bounds['upper_sum_b']
                # t_start = time.time()
                # element_ = np.array(element).reshape(input_shape)
                # input_range_ = np.empty_like(input_range)
                # input_range_[..., 0] = input_range[..., 0] + np.multiply(
                #     element_, slope
                # )
                # input_range_[..., 1] = input_range[..., 0] + np.multiply(
                #     element_ + 1, slope
                # )

                # ranges = input_range_
                ranges = element.ranges
                xt_min = ranges[..., 0]
                xt_max = ranges[..., 1]
                


                if slow_cvxpy:
                    # Initialize cvxpy variables
                    xt = cp.Variable((num_states, num_steps+1))
                    ut = cp.Variable((num_control_inputs, num_steps))
                    constrs = []

                    # x_{t=0} \in this partition of 0-th backreachable set
                    constrs += [xt_min <= xt[:, 0]]
                    constrs += [xt[:, 0] <= xt_max]

                    # # if self.dynamics.x_limits is not None:
                    # #     x_llim = self.dynamics.x_limits[:, 0]
                    # #     x_ulim = self.dynamics.x_limits[:, 1]
                    

                    # # # Each xt must be in a backprojection overapprox
                    # # for t in range(num_steps - 1):
                    # #     A, b = input_constraints[t].A[0], input_constraints[t].b[0]
                    # #     constrs += [A@xt[:, t+1] <= b]

                    # # x_{t=T} must be in target set
                    # if isinstance(output_constraint, constraints.LpConstraint):
                    #     goal_set_A, goal_set_b = range_to_polytope(output_constraint.range)
                    # elif isinstance(output_constraint, constraints.PolytopeConstraint):
                    #     goal_set_A, goal_set_b = output_constraint.A, output_constraint.b[0]
                    # constrs += [goal_set_A@xt[:, -1] <= goal_set_b]
                    # t_end = time.time()
                    # infos[-1]['nstep_other'].append(t_end-t_start)
                    # # Each ut must not exceed CROWN bounds
                    # for t in range(num_steps):
                    #     t_start = time.time()
                    #     if t == 0:
                    #         lower_A, lower_sum_b, upper_A, upper_sum_b = element.crown_bounds['lower_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_A'], element.crown_bounds['upper_sum_b']
                    #     else:
                    #         # Gather CROWN bounds for full backprojection overapprox
                    #         # import pdb; pdb.set_trace()
                    #         upper_A = infos[-t-1]['upper_A']
                    #         lower_A = infos[-t-1]['lower_A']
                    #         upper_sum_b = infos[-t-1]['upper_sum_b']
                    #         lower_sum_b = infos[-t-1]['lower_sum_b']
                    #     t_end = time.time()
                    #     infos[-1]['nstep_crown'].append(t_end-t_start)

                    #     # u_t bounded by CROWN bounds
                    #     constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
                    #     constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]
                    
                    # t_start = time.time()

                    # # Each xt must fall in the original backprojection
                    # for t in range(num_steps):
                    #     constrs += [input_constraints[-t-1].range[:,0] <= xt[:,t]]
                    #     constrs += [xt[:,t] <= input_constraints[-t-1].range[:,1]]


                    # # x_t and x_{t+1} connected through system dynamics
                    # for t in range(num_steps):
                    #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]

                    #     # if self.dynamics.x_limits is not None:
                    #     #     x_llim = self.dynamics.x_limits[:, 0]
                    #     #     x_ulim = self.dynamics.x_limits[:, 1]
                    #     #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) <= x_ulim]
                    #     #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) >= x_llim]

                    # # u_t satisfies control limits (TODO: Necessary? CROWN should account for these)
                    # # for t in range(num_steps):
                    # #     constrs += [-1 <= ut[:, t]]
                    # #     constrs += [1 >= ut[:, t]]

                    A_facets_i = cp.Parameter(num_states)
                    obj = A_facets_i@xt[:, 0]
                    prob = cp.Problem(cp.Maximize(obj), constrs)

                    for t in range(num_steps):
                        # Gather CROWN bounds and previous BP bounds
                        if t > 0:
                            upper_A = infos[-t-1]['upper_A']
                            lower_A = infos[-t-1]['lower_A']
                            upper_sum_b = infos[-t-1]['upper_sum_b']
                            lower_sum_b = infos[-t-1]['lower_sum_b']

                            # Each xt must fall in the original backprojection
                            constrs += [input_constraints[-t-1].range[:,0] <= xt[:, t]]
                            constrs += [xt[:, t] <= input_constraints[-t-1].range[:,1]]

                        # u_t bounded by CROWN bounds
                        constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
                        constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]

                        

                        # x_t and x_{t+1} connected through system dynamics
                        constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]
                    
                    A_t_i = cp.Parameter(num_states)
                    obj = A_t_i@xt[:,0]
                    prob = cp.Problem(cp.Maximize(obj), constrs)
                else:
                    lower_A, lower_sum_b, upper_A, upper_sum_b = element.crown_bounds['lower_A'], element.crown_bounds['lower_sum_b'], element.crown_bounds['upper_A'], element.crown_bounds['upper_sum_b']
                    prob = problems[0]
                    params = problems[2]
                    ##### Solve Problem Parametrically #####
                    params['lower_A'][0].value = lower_A
                    params['upper_A'][0].value = upper_A
                    params['lower_sum_b'][0].value = lower_sum_b
                    params['upper_sum_b'][0].value = upper_sum_b
                    
                    params['xt_min'][0].value = element.ranges[:,0]
                    params['xt_max'][0].value = element.ranges[:,1]

                min_idx = xt_min < xt_range_min
                max_idx = xt_max > xt_range_max
                idx1 = np.hstack((max_idx, min_idx))
                
                # Flag to use naive partitioning
                if all_lps:
                    idx1 = np.ones(2*num_states)

                # Check which state (if any) was already optimized during the BR set calculation
                idx2 = np.invert((A_facets == element.A_t_br).all(axis=1))
                idx = np.vstack((idx1, idx2)).all(axis=0)

                A_ = A_facets
                A_NN, b_NN = range_to_polytope(ranges)
                b_ = b_NN
                t_end = time.time()
                infos[-1]['nstep_other'].append(t_end-t_start)
                for i in range(num_facets):
                    if idx[i]:
                        if slow_cvxpy:
                            A_t_i.value = A_facets[i, :]
                        else:
                            params['A_t'].value = A_facets[i, :]
                        t_start = time.time()
                        prob.solve()
                        t_end = time.time()
                        infos[-1]['nstep_bp_lp'].append(t_end-t_start)
                        # prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
                        b_[i] = prob.value


                # This cell of the backprojection set is upper-bounded by the
                # cell of the backreachable set that we used in the NN relaxation
                # ==> the polytope is the intersection (i.e., concatenation)
                # of the polytope used for relaxing the NN and the soln to the LP
                t_start = time.time()
                A_stack = np.vstack([A_, A_NN])
                b_stack = np.hstack([b_, b_NN])

                if isinstance(output_constraint, constraints.LpConstraint):
                    b_max = b_[0:int(len(b_)/2)]
                    b_min = -b_[int(len(b_)/2):int(len(b_))]

                    # print('--------------start--------------')
                    # print('{} vs {}'.format(xt_range_max, b_max))
                    # print('{} vs {}'.format(xt_range_min, b_min))
                    # print('---------------------------------')
                    xt_range_max = np.max((xt_range_max, b_max),axis=0)
                    xt_range_min = np.min((xt_range_min, b_min),axis=0)
                    # print('{} vs {}'.format(xt_range_max, b_max))
                    # print('{} vs {}'.format(xt_range_min, b_min))
                    # import pdb; pdb.set_trace()

                    tightened_constraint.range = np.array([xt_range_min,xt_range_max]).T

                elif isinstance(output_constraint, constraints.PolytopeConstraint):
                    vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                    if len(vertices) > 0:
                        # import pdb; pdb.set_trace()
                        # pypoman.polygon.compute_polygon_hull(A_stack, b_stack+1e-10)
                        # vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                        
                        xt_max_candidate = np.max(vertices, axis=0)
                        xt_min_candidate = np.min(vertices, axis=0)
                        xt_range_max = np.maximum(xt_range_max, xt_max_candidate)
                        xt_range_min = np.minimum(xt_range_min, xt_min_candidate)

                        
                        tightened_constraint.A.append(A_)
                        tightened_constraint.b.append(b_)
                else:
                    raise NotImplementedError

        if isinstance(output_constraint, constraints.LpConstraint):
            input_constraint = deepcopy(input_constraints[-1])
            input_constraint.range = np.vstack((xt_range_min, xt_range_max)).T
        elif isinstance(output_constraint, constraints.PolytopeConstraint):
            x_overapprox = np.vstack((xt_range_min, xt_range_max)).T
            A_overapprox, b_overapprox = range_to_polytope(x_overapprox)

            # infos[-1]['tightened_constraint'] = tightened_constraint
            # infos[-1]['tightened_overapprox'] = constraints.PolytopeConstraint(A_overapprox, b_overapprox)
            
            input_constraint = deepcopy(input_constraints[-1])
            input_constraint.A = [A_overapprox]
            input_constraint.b = [b_overapprox]
        
        t_end = time.time()
        infos[-1]['nstep_other'].append(t_end-t_start)

        info = infos[-1]
        info['one_step_backprojection_overapprox'] = input_constraints[-1]
        # info['nstep_bp_set_partitions'] = [constraints.LpConstraint(range=element.ranges) for element in element_list]

        return input_constraint, info

    def get_N_step_reachable_set(
        self,
        input_constraint,
        output_constraints,
        infos,
    ):

        # TODO: Get this to work for Polytopes too
        # TODO: Confirm this works with partitioners
        # TODO: pull the cvxpy out of this function
        # TODO: add back in noise, observation model

        A_out = np.eye(self.dynamics.At.shape[0])
        num_facets = A_out.shape[0]
        ranges = np.zeros((num_facets, 2))
        num_steps = len(output_constraints)

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        x_min = output_constraints[-2].range[..., 0]
        x_max = output_constraints[-2].range[..., 1]
        norm = output_constraints[-2].p
        prev_state_max = torch.Tensor([x_max])
        prev_state_min = torch.Tensor([x_min])
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor([self.dynamics.sensor_noise[:, 1]])
            nn_input_min += torch.Tensor([self.dynamics.sensor_noise[:, 0]])

        # Compute the NN output matrices (for the input constraints)
        num_control_inputs = self.dynamics.bt.shape[1]
        C = torch.eye(num_control_inputs).unsqueeze(0)
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
            method_opt=self.method_opt,
            norm=norm,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )
        infos[-1]['nn_matrices'] = {
            'lower_A': lower_A.data.numpy()[0],
            'upper_A': upper_A.data.numpy()[0],
            'lower_sum_b': lower_sum_b.data.numpy()[0],
            'upper_sum_b': upper_sum_b.data.numpy()[0],
        }

        import cvxpy as cp

        num_states = self.dynamics.bt.shape[0]
        xs = []
        for t in range(num_steps+1):
            xs.append(cp.Variable(num_states))

        constraints = []

        # xt \in Xt
        constraints += [
            xs[0] <= input_constraint.range[..., 1],
            xs[0] >= input_constraint.range[..., 0]
        ]

        # xt+1 \in our one-step overbound on Xt+1
        for t in range(num_steps - 1):
            constraints += [
                xs[t+1] <= output_constraints[t].range[..., 1],
                xs[t+1] >= output_constraints[t].range[..., 0]
            ]

        # xt1 connected to xt via dynamics
        for t in range(num_steps):
            # Handle case of Bt ! >= 0 by adding a constraint per state
            for j in range(num_states):

                upper_A = np.where(np.tile(self.dynamics.bt[j, :], (num_states, 1)).T >= 0, infos[t]['nn_matrices']['upper_A'], infos[t]['nn_matrices']['lower_A'])
                lower_A = np.where(np.tile(self.dynamics.bt[j, :], (num_states, 1)).T >= 0, infos[t]['nn_matrices']['lower_A'], infos[t]['nn_matrices']['upper_A'])
                upper_sum_b = np.where(self.dynamics.bt[j, :] >= 0, infos[t]['nn_matrices']['upper_sum_b'], infos[t]['nn_matrices']['lower_sum_b'])
                lower_sum_b = np.where(self.dynamics.bt[j, :] >= 0, infos[t]['nn_matrices']['lower_sum_b'], infos[t]['nn_matrices']['upper_sum_b'])

                if self.dynamics.continuous_time:
                    constraints += [
                        xs[t+1][j] <= xs[t][j] + self.dynamics.dt * (self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(upper_A@xs[t]+upper_sum_b)+self.dynamics.ct[j]),
                        xs[t+1][j] >= xs[t][j] + self.dynamics.dt * (self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(lower_A@xs[t]+lower_sum_b)+self.dynamics.ct[j]),
                    ]
                else:
                    constraints += [
                        xs[t+1][j] <= self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(upper_A@xs[t]+upper_sum_b)+self.dynamics.ct[j],
                        xs[t+1][j] >= self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(lower_A@xs[t]+lower_sum_b)+self.dynamics.ct[j],
                    ]

        A_out_facet = cp.Parameter(num_states)

        obj = cp.Maximize(A_out_facet@xs[-1])
        prob_max = cp.Problem(obj, constraints)

        obj = cp.Minimize(A_out_facet@xs[-1])
        prob_min = cp.Problem(obj, constraints)

        for i in range(num_facets):

            A_out_facet.value = A_out[i, :]

            prob_max.solve()
            A_out_xtN_max = prob_max.value

            prob_min.solve()
            A_out_xtN_min = prob_min.value

            ranges[i, 0] = A_out_xtN_min
            ranges[i, 1] = A_out_xtN_max

        output_constraint = deepcopy(output_constraints[-1])
        output_constraint.range = ranges
        return output_constraint, infos

    def get_crown_matrices(self, xt_min, xt_max, num_control_inputs):
        nn_input_max = torch.Tensor([xt_max])
        nn_input_min = torch.Tensor([xt_min])

        # Compute the NN output matrices (for this xt partition)
        C = torch.eye(num_control_inputs).unsqueeze(0)
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
            method_opt=self.method_opt,
            norm=np.inf,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )
        upper_A = upper_A.detach().numpy()[0]
        lower_A = lower_A.detach().numpy()[0]
        upper_sum_b = upper_sum_b.detach().numpy()[0]
        lower_sum_b = lower_sum_b.detach().numpy()[0]

        return lower_A, upper_A, lower_sum_b, upper_sum_b


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
        
        # import pdb; pdb.set_trace()
        if not np.isnan(rect_verts).any():
            rect_hull = ConvexHull(rect_verts)
            area = rect_hull.volume
        else:
            area = np.inf

        if area < min_area:
            min_area = area
            mar_verts = rect_verts
        
        # import pdb; pdb.set_trace()

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

    
    # width_idx = 0
    # width_max = -np.inf
    # for idx, vert in enumerate(mar_verts):
    #     if vert[0] >= width_max:
    #         if (vert[0] == width_max and vert[0] < mar_verts[width_idx, 1]) or vert[0] > width_max:
    #             width_idx = idx
    #             width_max = vert[0]

    # xy = mar_verts[xy_idx, :]
    # base = mar_verts[width_idx, :] - xy
    # width = np.linalg.norm(base)
    # height = min_area/width
    # angle = np.arctan2(base[1], base[0])

    mar_hull = ConvexHull(mar_verts)
    width = np.linalg.norm(mar_verts[theta_idx, :]-xy)
    height = mar_hull.volume/width
    W = np.array([width, height])



    # import pdb; pdb.set_trace()

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # fig, ax = plt.subplots()
    # ax.scatter(points[:, 0], points[:, 1])
    # ax.scatter(vertices[:, 0], vertices[:, 1], color='r')
    # ax.scatter(rect_verts[:, 0], rect_verts[:, 1], color='b')
    # convex_hull_plot_2d(mar_hull, ax=ax)
    # rect = Rectangle(xy, width, height, angle, ec='b', fill=False)
    # ax.add_patch(rect)
    plt.show()

    # import pdb; pdb.set_trace()
    return mar_hull, theta, xy, W, mar_verts
