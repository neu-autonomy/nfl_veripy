from .Partitioner import Partitioner
import numpy as np
from nn_partition.utils.object_boundary import getboundary

class BoundarySimGuidedPartitioner(Partitioner):
    def __init__(self, N=20, tolerance_eps=0.01):
        Partitioner.__init__(self)
        self.N= N
        self.tolerance_eps= tolerance_eps

    def get_output_range(self, input_range, propagator):
        input_shape = input_range.shape[:-1]
        num_propagator_calls = 0

        interior_M = []
        M=[]
        M_inside=[]
        info = {}
        boundary_shape="convex"
        sect_method = 'max'
        num_inputs = input_range.shape[0]
        num_partitions = (self.N)*np.ones((num_inputs,), dtype=int)
        slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
        sampling_cell = np.empty(((self.N)**2,num_inputs,2))
        output_range_ = np.empty(((self.N)**2,num_inputs,2))
        sampled_output_ = np.empty(((self.N)**2,num_inputs))
        sampled_output_boundary_ = np.empty(((self.N)**2))

   
        input_range_ = np.empty((self.N**2,num_inputs,2))

        element=-1
        for idx in product(*[range(num) for num in num_partitions]):
            element = element+1
            input_range_[element,:,0] = input_range[:,0]+np.multiply(idx, slope)
            input_range_[element,:,1] = input_range[:,0]+np.multiply(np.array(idx)+1, slope)
            sampled_input_= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))
            sampled_output_[element,:] =  propagator.forward_pass( propagator.forward_pass(sampled_input_))
            sampled_output_boundary_[element] =0;
           # output_range_[element,:,:],_= propagator.get_output_range(input_range_[element,:,:])
           # num_propagator_calls += 1

        if boundary_shape=="convex":
            boundary_points= getboundary(sampled_output_,0.0)
        else:
            boundary_points= getboundary(sampled_output_,0.4)

        for i in range(self.N**2):
            if sampled_output_[i,:] in boundary_points:
                sampled_output_boundary_[i]=1
                propagator.get_output_range(input_range_[i,:,:])
                num_propagator_calls += 1
                M.append((input_range_[i,:,:], output_range_[i,:,:], sampled_output_boundary_[i]))# (Line 4)  

            else:
                M_inside.append((input_range_[i,:,:]))    

        

        output_range_sim = np.empty_like(output_range_[0,:,:])
        output_range_sim[:,0] = np.min(sampled_output_, axis=0)
        output_range_sim[:,1] = np.max(sampled_output_, axis=0)
            
        u_e = np.empty_like(output_range_sim)
        u_e[:,0] = np.inf
        u_e[:,1] = -np.inf


        M_e = np.empty_like(output_range_sim)
        M_e[:,0] = np.inf
        M_e[:,1] = -np.inf

        while len(M) != 0:
            input_range_, output_range_,sampled_output_boundary_ = M.pop(0) # Line 9
            if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0) or sampled_output_boundary_==0:
            # Line 11
                tmp = np.dstack([u_e, output_range_])
                u_e[:,1] = np.max(tmp[:,1,:], axis=1)
                u_e[:,0] = np.min(tmp[:,0,:], axis=1)
                interior_M.append((input_range_, output_range_))
            else:
            # Line 14
                if np.max(input_range_[:,1] - input_range_[:,0]) > self.tolerance_eps:
                # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                    for input_range_ in input_ranges_:
                        output_range_,_= propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
   
                        M.append((input_range_, output_range_,sampled_output_boundary_)) # Line 18
                else: # Lines 19-20
                    interior_M.append((input_range_, output_range_))
                    break


        if len(M) > 0:
            M_numpy = np.dstack([output_range_ for (_, output_range_,_) in M])
            M_range = np.empty_like(u_e)
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
    
            tmp1 = np.dstack([u_e, M_range])
            u_e[:,1] = np.max(tmp1[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp1[:,0,:], axis=1)
        M1=[]
        if len(M_inside) > 0:
            for (input_range) in M_inside:
                output_range_,_= propagator.get_output_range(input_range_)
                num_propagator_calls += 1
                M2 = np.dstack([M_e, output_range_])
                M_e[:,1] = np.max(M2[:,1,:], axis=1)
                M_e[:,0] = np.min(M2[:,0,:], axis=1)

            tmp2 = np.dstack([u_e, M_e])
            u_e[:,1] = np.max(tmp2[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp2[:,0,:], axis=1)
    

     
        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        return u_e, info

