import numpy as np

class Element():
    def __init__(self, ranges):
        self.ranges = ranges
        width = self.ranges[:,1] - self.ranges[:,0]
        self.volume = np.prod(width, axis=0)
        self.prop = 1
        self.flag = None
        self.A_t_br = None

    def __lt__(self, other):
        if self.prop == other.prop:
            return self.get_volume() < other.get_volume()
        return self.prop < other.prop


    def get_volume(self):
        width = self.ranges[:,1] - self.ranges[:,0]
        return np.prod(width, axis=0)

    # def split(self, target_set, dynamics, full_samples=None):
    #     element_list = []
    #         dim = len(self.ranges[0])
    #         if not isinstance(partition_budget, list):
    #             num_partitions = np.array([partition_budget for i in range(dim)])
    #             # import pdb; pdb.set_trace()
    #         else:
    #             num_partitions = partition_budget
    #         input_shape = br_set_element.ranges.shape[:-1]
    #         slope = np.divide(
    #             (br_set_element.ranges[..., 1] - br_set_element.ranges[..., 0]), num_partitions
    #         )
    #         for el in product(*[range(int(num)) for num in num_partitions.flatten()]):
    #             element_ = np.array(el).reshape(input_shape)
    #             input_range_ = np.empty_like(br_set_element.ranges)
    #             input_range_[..., 0] = br_set_element.ranges[..., 0] + np.multiply(
    #                 element_, slope
    #             )
    #             input_range_[..., 1] = br_set_element.ranges[..., 0] + np.multiply(
    #                 element_ + 1, slope
    #             )
    #             element_list.append(elements.Element(input_range_))
        

  
    # def split(self, target_set=None, dynamics=None, heuristic=None, full_samples=None):

    #     ############################# Ignore this one #############################
    #     if heuristic is 'split_most':
    #         max_samples = -np.inf
    #         split_dim = 0
    #         for i,dim in enumerate(self.ranges):
                
    #             avg = (dim[0]+dim[1])/2
    #             # samples_above = self.samples[self.samples[:,i] > avg]
    #             # samples_below = self.samples[self.samples[:,i] <= avg]
    #             split_samples_candidate = self.samples[self.samples[:,i] < avg], self.samples[self.samples[:,i] > avg]
    #             for j,side in enumerate(split_samples_candidate):
    #                 if len(side) > max_samples:
    #                     max_samples = len(side)
    #                     split_dim = i
    #                     split_samples = split_samples_candidate

    #         cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
        
    #     ############################# Ignore this one #############################
    #     elif heuristic is 'box_out':
    #         buffer = 0
    #         xtreme = np.array(
    #             [
    #                 np.min(full_samples,axis=0),
    #                 np.max(full_samples,axis=0)
    #             ]
    #         )
    #         for i,dim in enumerate(self.ranges):
    #             diff_magnitude = np.abs(self.ranges.T - xtreme)
    #             # import pdb; pdb.set_trace()
    #             flat_idx = np.argmax(diff_magnitude)
    #             idx = np.unravel_index(flat_idx, diff_magnitude.shape)
    #             split_dim = idx[1]
    #             if len(self.samples) == 0:
    #                 # import pdb; pdb.set_trace()
    #                 cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
    #             else:
    #                 if  idx[0] == 0:
    #                     buffer = -np.abs(buffer)
    #                 else: 
    #                     buffer = np.abs(buffer)

    #                 cut = xtreme[idx] + buffer

        
    #     ############################# This one has promise #############################
    #     elif heuristic is 'guided':
    #         # Grab bounds of MC samples in backreachable set (not necessarily in current element)
    #         if len(full_samples) > 0: 
    #             xtreme = np.array(
    #                 [
    #                     np.min(full_samples,axis=0),
    #                     np.max(full_samples,axis=0)
    #                 ]
    #             )
    #         elif target_set is not None:
    #             xtreme = target_set.range.T
            
    #         # Choose where to cut element and along which direction
    #         if len(self.samples) == 0:
    #             # # Possible idea of choosing dimension based on crown bounds
    #             # if not hasattr(self, 'crown_bounds'):
    #             #     split_dim = np.argmax(np.ptp(self.ranges, axis=1))
    #             # else: 
    #             #     split_dim = np.argmax(np.abs(self.crown_bounds['upper_A']-self.crown_bounds['lower_A']))

    #             # No samples in element -> bisect it hamburger style
    #             split_dim = np.argmax(np.ptp(self.ranges, axis=1))
    #             cut = (self.ranges[split_dim,0]+self.ranges[split_dim,1])/2
    #         else:

    #             # samples in element -> split element near border of samples such that we maximize volume of new element without samples and minimize volume of element containing samples
    #             buffer = 0.02
    #             diff_magnitude = np.abs(self.ranges.T - xtreme)
    #             flat_idx = np.argmax(diff_magnitude)
    #             idx = np.unravel_index(flat_idx, diff_magnitude.shape)

    #             if  idx[0] == 0:
    #                 buffer = -np.abs(buffer)
    #             else: 
    #                 buffer = np.abs(buffer)

                
    #             split_dim = idx[1]
    #             cut = xtreme[idx] + buffer*(xtreme.T[split_dim,1]-xtreme.T[split_dim,0])
            

    #         # import pdb; pdb.set_trace()

    #     elif heuristic is None:
    #         raise NotImplementedError

    #     # split samples into regions contained by new elements
    #     split_samples = self.samples[self.samples[:,split_dim] < cut], self.samples[self.samples[:,split_dim] > cut]

        
    #     lower_split_range = np.array([self.ranges[split_dim,0], cut])
    #     upper_split_range = np.array([cut, self.ranges[split_dim,1]])
        
    #     new_ranges = deepcopy(self.ranges), deepcopy(self.ranges)
    #     new_ranges[0][split_dim] = lower_split_range
    #     new_ranges[1][split_dim] = upper_split_range
        
    #     # Generate new elements
    #     elements = Element(new_ranges[0], split_samples[0], heuristic=heuristic, policy=self.policy), Element(new_ranges[1], split_samples[1], heuristic=heuristic, policy=self.policy)

    #     # Assign value to new elements (used to sort list of elements to be partitioned)
    #     if heuristic is 'box_out':
    #         for el in elements:
    #             # import pdb; pdb.set_trace()
    #             if len(set(el.ranges.flatten()).intersection(set(np.hstack((xtreme.flatten(), xtreme.flatten()+buffer, xtreme.flatten()-buffer))))) == 0:
    #                 el.prop = el.prop*0

    #     elif heuristic is 'guided':
    #         if len(full_samples) > 0:
    #             sample_center = np.mean(full_samples, axis=0)
    #         else:
    #             sample_center = np.mean(self.ranges.T, axis=0)
    #         for el in elements:
    #             # if len(el.samples) > 0: # if the element contains samples, prioritize it in the queue
    #             #     # el.prop += np.inf
    #             #     element_center = np.mean(self.ranges, axis=1)
    #             #     el.prop += np.linalg.norm(element_center-sample_center, 1)
    #             # else: # otherwise, determine if it is feasible to reach the target set from this element and if so assign a cost
    #             num_control_inputs = dynamics.bt.shape[1]
    #             C = torch.eye(num_control_inputs).unsqueeze(0)

    #             nn_input_max = torch.Tensor(np.array([el.ranges[:,1]]))
    #             nn_input_min = torch.Tensor(np.array([el.ranges[:,0]]))
    #             norm = np.inf

    #             el.crown_bounds = {}
    #             el.crown_bounds['lower_A'], el.crown_bounds['upper_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_sum_b'] = self.policy(
    #                 method_opt='full_backward_range',
    #                 norm=norm,
    #                 x_U=nn_input_max,
    #                 x_L=nn_input_min,
    #                 upper=True,
    #                 lower=True,
    #                 C=C,
    #                 return_matrices=True,
    #             )

    #             el.crown_bounds['lower_A'], el.crown_bounds['upper_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_sum_b'] = el.crown_bounds['lower_A'].detach().numpy()[0], el.crown_bounds['upper_A'].detach().numpy()[0], el.crown_bounds['lower_sum_b'].detach().numpy()[0], el.crown_bounds['upper_sum_b'].detach().numpy()[0]
                            
    #             lower_A, lower_sum_b, upper_A, upper_sum_b = el.crown_bounds['lower_A'], el.crown_bounds['lower_sum_b'], el.crown_bounds['upper_A'], el.crown_bounds['upper_sum_b']
                
    #             if isinstance(target_set, constraints.LpConstraint):
    #                 xt1_min = target_set.range[..., 0]
    #                 xt1_max = target_set.range[..., 1]
    #             else:
    #                 raise NotImplementedError
                
    #             xt = cp.Variable(xt1_min.shape)
    #             ut = cp.Variable(num_control_inputs)
    #             constrs = []

    #             # Constraints to ensure that xt stays within the backreachable set
    #             constrs += [el.ranges[:, 0]+0.0 <= xt]
    #             constrs += [xt <= el.ranges[:,1]-0.0]

    #             # Constraints to ensure that ut satisfies the affine bounds
    #             constrs += [lower_A@xt+lower_sum_b <= ut]
    #             constrs += [ut <= upper_A@xt+upper_sum_b]

    #             # Constraints to ensure xt reaches the target set given ut
    #             constrs += [dynamics.dynamics_step(xt, ut) <= xt1_max]
    #             constrs += [dynamics.dynamics_step(xt, ut) >= xt1_min]

    #             # print("element range: \n {}".format(el.ranges))
    #             obj = 0
    #             prob = cp.Problem(cp.Maximize(obj), constrs)
    #             t_start = time.time()
    #             prob.solve()
    #             t_end = time.time()
    #             del_t = t_end-t_start

    #             A_t_i = np.array([1., 0])
    #             new_obj = A_t_i@xt
    #             new_prob = cp.Problem(cp.Maximize(new_obj), constrs)
    #             new_t_start = time.time()
    #             new_prob.solve()
    #             new_t_end = time.time()
    #             new_del_t = new_t_end - new_t_start

    #             print('positive is good: {}'.format(del_t-new_del_t))
    #             print(del_t)
    #             print(new_del_t)

    #             # new_obj = np.array([0, 1])
    #             # print("feasibility checked in {} seconds".format(t_end-t_start))
    #             # import pdb; pdb.set_trace()
    #             el.flag = prob.status
    #             is_terminal_cell = False
    #             if len(full_samples) > 0:
    #                 diff_magnitude = np.abs(el.ranges.T - xtreme)
    #                 if np.max(diff_magnitude) < 0.05:
    #                     is_terminal_cell = True
    #             # print("lp solution for feasibility: {}".format(xt.value))
    #             # print("lp status for feasibility: {}".format(element_feasibility))
                
                
    #             # If the element is not feasible (or is element bounding samples), assign value to zero
    #             if el.flag == 'infeasible' or is_terminal_cell:
    #                 el.prop = 0
                
    #             # Else, value is determined by (distance of furthest corner from sample center) * (volume of cell)
    #             else:

    #                 element_center = np.mean(el.ranges, axis=1)
    #                 # import pdb; pdb.set_trace()
    #                 # dist = np.linalg.norm(element_center-sample_center, 1)
    #                 dist = np.linalg.norm(np.max(np.abs(el.ranges.T-sample_center), axis=0), 1)
    #                 volume = el.get_volume()
    #                 el.prop = dist*volume
    #                 # print(el.ranges)
    #                 # print(dist)
    #                 # import pdb; pdb.set_trace()
    #                 # if len(el.samples) > 0 and dist < 0.01:
    #                 #     print('whoaaaaaaa')


    #     return elements