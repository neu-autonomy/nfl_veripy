from .ClosedLoopPartitioner import ClosedLoopPartitioner

class ClosedLoopProbabilisticPartitioner(ClosedLoopPartitioner):

    def __init__(self, dynamics, num_partitions=16):
        print("I don't think this fully works yet.")
        raise NotImplementedError
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    def get_one_step_reachable_set(self, input_constraint, output_constraint, propagator, num_partitions=None):
        reachable_set, info, prob = self.get_reachable_set(input_constraint, output_constraint, propagator, t_max=1, num_partitions=num_partitions)
        return reachable_set, info, prob

    def get_reachable_set(self, input_constraint, output_constraint, propagator, t_max, num_partitions=None):

        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # only used to compute slope in non-closedloop manner...
            input_polytope_verts = pypoman.duality.compute_polytope_vertices(A_inputs, b_inputs)
            input_range = np.empty((A_inputs.shape[1],2))
            input_range[:,0] = np.min(np.stack(input_polytope_verts), axis=0)
            input_range[:,1] = np.max(np.stack(input_polytope_verts), axis=0)
           
        elif isinstance(input_constraint, LpInputConstraint):
            input_range = input_constraint.range
        else:
            raise NotImplementedError
        likelihood_set=[]
        input_range_0 = input_range
        info = {}
        num_propagator_calls = 0
        offset = 0
        input_shape = input_range.shape[:-1]
        sampled_output_ranges = self.get_sampled_out_range(input_constraint, propagator, t_max, num_samples =50)
        output_constraint_=[]

        if num_partitions is None:
           num_partitions = 7
        output_w_likelihood_onestep = [(input_range,1)]
        num_layers = t_max*num_partitions
       # slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)
        for t in range(t_max):
            output_w_likelihood =[]   

            sampled_output_range = sampled_output_ranges[t,:,:]
            for (input_range,likelihood_output) in output_w_likelihood_onestep:
                ranges = []
                reachable_set = None
                output_constraint_0 = None
       
                if isinstance(input_constraint, PolytopeInputConstraint):
                    raise NotImplementedError
              
                elif isinstance(input_constraint, LpInputConstraint):
                    input_constraint_ = input_constraint.__class__(range=input_range, p=input_constraint.p)
                else:
                    raise NotImplementedError

                output_constraint_init, info_ = propagator.get_reachable_set(input_constraint_, deepcopy(output_constraint), 1)
         
                num_propagator_calls+=1
                if likelihood_output is None:
                    likelihood_output =np.empty(num_partitions)

                if t==0:
                    idx =num_partitions
                    range_diff = output_constraint_init[0].range-sampled_output_range
                    layer_slope = range_diff/num_partitions
                    output_constraint_0 =sampled_output_range
                    for layer_idx in range(num_partitions):
                        output_constraint_0=layer_slope+output_constraint_0
                        output_w_likelihood.append((output_constraint_0,1-layer_idx/num_partitions-offset))
                else:
                   

 
                    range_diff = output_constraint_init[0].range-sampled_output_range
                    range_diff[...,0] = np.min(range_diff[...,0,:], axis=-1)
                    range_diff[...,1] = np.max(range_diff[...,1,:], axis=-1)
                    likelihood_output*= 1-min(1,np.max(range_diff)+offset)             
                    output_w_likelihood.append((output_constraint_init[0].range,likelihood_output))
         
        ## compare sampled output with the estimated one
        ## partition the area between estimated and sampled into layers
        ### assign gaussian distribution to the layers
        ## each layer is propagated with associated probability

            if t!=0:
                output_w_likelihood.append((sampled_output_range,1-offset) )   
            output_w_likelihood_onestep = output_w_likelihood.copy()   
        if isinstance(output_constraint, PolytopeOutputConstraint):
            raise NotImplementedError
        elif isinstance(output_constraint, LpOutputConstraint):
            #reachable_set_ = output_constraint_
            reachable_set_=[]
            output_onestep_range = output_w_likelihood_onestep.copy()
            while  output_w_likelihood_onestep!=[]:
                reach_set,likelihood_final=output_w_likelihood_onestep.pop()
                reachable_set_.append(reach_set)
                likelihood_set.append(likelihood_final)
            if output_constraint.range is None:
                output_constraint.range= np.stack(reachable_set_)
            tmp = np.stack([output_constraint.range, np.stack(reachable_set_)], axis=-1)
            output_constraint.range[...,0] = np.min(tmp[...,0,:], axis=-1)
            output_constraint.range[...,1] = np.max(tmp[...,1,:], axis=-1)
            ranges.append((input_range_0, np.stack(reachable_set_)))
        else:
            raise NotImplementedError

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)
        return output_constraint, info, likelihood_set