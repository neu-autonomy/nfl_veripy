import numpy as np
import cvxpy as cp
from copy import deepcopy
import torch
import time
from nn_closed_loop.nn_closed_loop.elements import Element, GuidedElement
import nn_closed_loop.constraints as constraints

class OptGuidedElement(GuidedElement):
    def __init__(self, ranges, policy, samples=None):
        super().__init__(ranges, policy, samples)
        self.A_t_br = None


    def split(self, target_set, dynamics, problems, full_samples=None, br_set_range=None, nstep=False, infos=None, input_constraints=None, time_info=None, max_prob=None, min_prob=None, params=None):
        max_prob = problems[0]
        min_prob = problems[1]
        params = problems[2]
        # Grab bounds of MC samples in backreachable set (not necessarily in current element)
        t_start = time.time()
        if len(full_samples) > 0: 
            sample_limits = np.array(
                [
                    np.min(full_samples,axis=0),
                    np.max(full_samples,axis=0)
                ]
            )
        
        # Choose where to cut element and along which direction
        if len(self.samples) == 0:
            # No samples in element -> bisect it hamburger style
            split_dim = np.argmax(np.ptp(self.ranges, axis=1))
            cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
        else:
            # samples in element -> split element near border of samples such that we maximize volume of new element without samples and minimize volume of element containing samples
            buffer = 0.02
            diff_magnitude = np.abs(self.ranges.T - sample_limits)
            flat_idx = np.argmax(diff_magnitude)

            # Index gives information by row (should we make the cut above or below the set of samples) and by column (which dimension are we going to split)
            idx = np.unravel_index(flat_idx, diff_magnitude.shape)

            if  idx[0] == 0:
                buffer = -np.abs(buffer)
            else: 
                buffer = np.abs(buffer)

            split_dim = idx[1]
            cut = sample_limits[idx] + buffer*(sample_limits.T[split_dim,1]-sample_limits.T[split_dim,0])


        split_samples = self.samples[self.samples[:,split_dim] < cut], self.samples[self.samples[:,split_dim] > cut]

        
        lower_split_range = np.array([self.ranges[split_dim,0], cut])
        upper_split_range = np.array([cut, self.ranges[split_dim,1]])
        
        new_ranges = deepcopy(self.ranges), deepcopy(self.ranges)
        new_ranges[0][split_dim] = lower_split_range
        new_ranges[1][split_dim] = upper_split_range
        
        # Generate new elements
        elements = OptGuidedElement(new_ranges[0],self.policy,samples=split_samples[0]), OptGuidedElement(new_ranges[1],self.policy,samples=split_samples[1])

        
        if len(full_samples) > 0:
            sample_center = np.mean(full_samples, axis=0)
        else:
            sample_center = np.mean(br_set_range.T, axis=0)

        t_end = time.time()
        if time_info is not None:
            time_info['other'].append(t_end-t_start)
        for el in elements:
            time_info['br_set_partitions'] += [constraints.LpConstraint(range=el.ranges)]
            t_start = time.time()
            # if len(el.samples) > 0: # if the element contains samples, prioritize it in the queue
            #     # el.prop += np.inf
            #     element_center = np.mean(self.ranges, axis=1)
            #     el.prop += np.linalg.norm(element_center-sample_center, 1)
            # else: # otherwise, determine if it is feasible to reach the target set from this element and if so assign a cost
            num_control_inputs = dynamics.bt.shape[1]
            C = torch.eye(num_control_inputs).unsqueeze(0)

            nn_input_max = torch.Tensor(np.array([el.ranges[:,1]]))
            nn_input_min = torch.Tensor(np.array([el.ranges[:,0]]))
            norm = np.inf

            el.crown_bounds = {}
            t_end = time.time()
            if time_info is not None:
                time_info['other'].append(t_end-t_start)
            t_start = time.time()
            el.crown_bounds['lower_A'], el.crown_bounds['upper_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_sum_b'] = self.policy(
                method_opt='full_backward_range',
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
                time_info['crown'].append(t_end-t_start)
            t_start = time.time()

            el.crown_bounds['lower_A'], el.crown_bounds['upper_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_sum_b'] = el.crown_bounds['lower_A'].detach().numpy()[0], el.crown_bounds['upper_A'].detach().numpy()[0], el.crown_bounds['lower_sum_b'].detach().numpy()[0], el.crown_bounds['upper_sum_b'].detach().numpy()[0]
                        
            lower_A, lower_sum_b, upper_A, upper_sum_b = el.crown_bounds['lower_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_A'], el.crown_bounds['upper_sum_b']
            
            
            is_terminal_cell = False
            if len(full_samples) > 0:
                diff_magnitude = np.abs(el.ranges.T - sample_limits)
                if np.max(diff_magnitude) < 0.05:
                    is_terminal_cell = True
            t_end = time.time()
            if time_info is not None:
                time_info['other'].append(t_end-t_start)
            if not is_terminal_cell:
                t_start = time.time()
                if isinstance(target_set, constraints.LpConstraint):
                    xt1_min = target_set.range[..., 0]
                    xt1_max = target_set.range[..., 1]
                else:
                    raise NotImplementedError
                
                if not nstep:
                    xt = cp.Variable(xt1_min.shape)
                    ut = cp.Variable(num_control_inputs)
                    # constrs = []

                    # # Constraints to ensure that xt stays within the backreachable set
                    # constrs += [el.ranges[:, 0] <= xt]
                    # constrs += [xt <= el.ranges[:,1]]

                    # # Constraints to ensure that ut satisfies the affine bounds
                    # constrs += [lower_A@xt+lower_sum_b <= ut]
                    # constrs += [ut <= upper_A@xt+upper_sum_b]

                    # # Constraints to ensure xt reaches the target set given ut
                    # constrs += [dynamics.dynamics_step(xt, ut) <= xt1_max]
                    # constrs += [dynamics.dynamics_step(xt, ut) >= xt1_min]

                    # # print("element range: \n {}".format(el.ranges))
                    # diff = el.ranges.T-sample_center
                    # diff_magnitude = np.abs(diff)
                    # flat_idx = np.argmax(diff_magnitude)
                    # idx = np.unravel_index(flat_idx, diff_magnitude.shape)
                    # A_t_i = np.zeros(el.ranges.shape[0])
                    # A_t_i[idx[1]] = 1.
                    # el.A_t_br = A_t_i
                    # obj = A_t_i@xt

                    

                    params['lower_A'].value = lower_A
                    params['upper_A'].value = upper_A
                    params['lower_sum_b'].value = lower_sum_b
                    params['upper_sum_b'].value = upper_sum_b
                    
                    params['xt_min'].value = el.ranges[:,0]
                    params['xt_max'].value = el.ranges[:,1]

                    params['xt1_min'].value = xt1_min
                    params['xt1_max'].value = xt1_max

                    diff = el.ranges.T-sample_center
                    diff_magnitude = np.abs(diff)
                    flat_idx = np.argmax(diff_magnitude)
                    idx = np.unravel_index(flat_idx, diff_magnitude.shape)
                    A_t_i = np.zeros(el.ranges.shape[0])
                    A_t_i[idx[1]] = 1.
                    el.A_t_br = A_t_i

                    params['A_t'].value = A_t_i


                else: 
                    num_steps = len(input_constraints)
                    # xt = cp.Variable((el.ranges.shape[0], num_steps+1))
                    # ut = cp.Variable((num_control_inputs, num_steps))
                    # constrs = []

                    # # x_{t=0} \in this partition of 0-th backreachable set
                    # constrs += [el.ranges[:, 0] <= xt[:, 0]]
                    # constrs += [xt[:, 0] <= el.ranges[:, 1]]

                    # # if self.dynamics.x_limits is not None:
                    # #     x_llim = self.dynamics.x_limits[:, 0]
                    # #     x_ulim = self.dynamics.x_limits[:, 1]
                    

                    # # # Each xt must be in a backprojection overapprox
                    # # for t in range(num_steps - 1):
                    # #     A, b = input_constraints[t].A[0], input_constraints[t].b[0]
                    # #     constrs += [A@xt[:, t+1] <= b]

                    # # x_{t=T} must be in target set
                    # from nn_closed_loop.nn_closed_loop.utils.utils import range_to_polytope
                    # if isinstance(input_constraints[0], constraints.LpConstraint):
                    #     goal_set_A, goal_set_b = range_to_polytope(input_constraints[0].range)
                    # elif isinstance(input_constraints[0], constraints.PolytopeConstraint):
                    #     goal_set_A, goal_set_b = input_constraints[0].A, input_constraints[0].b[0]
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
                    #     constrs += [input_constraints[-t].range[:,0] <= xt[:,t]]
                    #     constrs += [xt[:,t] <= input_constraints[-t].range[:,1]]


                    # # x_t and x_{t+1} connected through system dynamics
                    # for t in range(num_steps):
                    #     constrs += [dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]


                    params['lower_A'][0].value = lower_A
                    params['upper_A'][0].value = upper_A
                    params['lower_sum_b'][0].value = lower_sum_b
                    params['upper_sum_b'][0].value = upper_sum_b
                    
                    params['xt_min'][0].value = el.ranges[:,0]
                    params['xt_max'][0].value = el.ranges[:,1]


                    diff = el.ranges.T-sample_center
                    diff_magnitude = np.abs(diff)
                    flat_idx = np.argmax(diff_magnitude)
                    idx = np.unravel_index(flat_idx, diff_magnitude.shape)
                    A_t_i = np.zeros(el.ranges.shape[0])
                    A_t_i[idx[1]] = 1.
                    el.A_t_br = A_t_i
                    params['A_t'].value = A_t_i









                    # diff = el.ranges.T-sample_center
                    # diff_magnitude = np.abs(diff)
                    # flat_idx = np.argmax(diff_magnitude)
                    # idx = np.unravel_index(flat_idx, diff_magnitude.shape)
                    # A_t_i = np.zeros(el.ranges.shape[0])
                    # A_t_i[idx[1]] = 1.
                    # el.A_t_br = A_t_i
                    # obj = A_t_i@xt[:,0]

                if idx[0] == 0:
                    # prob = cp.Problem(cp.Minimize(obj), constrs)
                    prob = min_prob
                    el.A_t_br = -el.A_t_br
                else:
                    # prob = cp.Problem(cp.Maximize(obj), constrs)
                    prob = max_prob
                t_end = time.time()
                if time_info is not None:
                    time_info['other'].append(t_end-t_start)
                t_start = time.time()
                prob.solve()
                t_end = time.time()
                if time_info is not None:
                    time_info['bp_lp'].append(t_end-t_start)
                del_t = t_end - t_start
                # import pdb; pdb.set_trace()
                temp = deepcopy(el.ranges.T)
                temp[idx] = prob.value
                el.ranges = temp.T
                el.flag = prob.status

                # print('status: {}; time: {}'.format(prob.status,del_t))

                # if len(input_constraints) == 18:
                #     import pdb; pdb.set_trace()
                # print(el.ranges)
                # print(temp.T)

                # import pdb; pdb.set_trace()

            # new_obj = np.array([0, 1])
            # print("feasibility checked in {} seconds".format(t_end-t_start))
            # import pdb; pdb.set_trace()
            
            # print("lp solution for feasibility: {}".format(xt.value))
            # print("lp status for feasibility: {}".format(element_feasibility))
            
            t_start = time.time()
            # If the element is not feasible (or is element bounding samples), assign value to zero
            if el.flag == 'infeasible' or is_terminal_cell:
                el.prop = 0            
            # Else, value is determined by (distance of furthest corner from sample center) * (volume of cell)
            else:

                element_center = np.mean(el.ranges, axis=1)
                # import pdb; pdb.set_trace()
                # dist = np.linalg.norm(element_center-sample_center, 1)
                dist = np.linalg.norm(np.max(np.abs(el.ranges.T-sample_center), axis=0), 1)
                volume = el.volume
                el.prop = dist*volume
                if volume < 0.0001:
                    el.prop = 0
                # print(el.ranges)
                # print(dist)
                # import pdb; pdb.set_trace()
                # if len(el.samples) > 0 and dist < 0.01:
                #     print('whoaaaaaaa')
            t_end = time.time()
            if time_info is not None:
                time_info['other'].append(t_end-t_start)


        return elements