import numpy as np
from itertools import product
from partition.xiang import sect, bisect


class Partitioner():
    def __init__(self):
        return

    def get_output_range(self):
        raise NotImplementedError

class NoPartitioner(Partitioner):
    def __init__(self):
        Partitioner.__init__(self)

    def get_output_range(self, input_range, propagator):
        output_range, info = propagator.get_output_range(input_range)
        return output_range, info

class UniformPartitioner(Partitioner):
    def __init__(self, num_partitions=16):
        Partitioner.__init__(self)
        self.num_partitions = num_partitions

    def get_output_range(self, input_range, propagator, num_partitions=None):
        info = {}
        num_propagator_calls = 0

        input_shape = input_range.shape[:-1]
        if num_partitions is None:
            num_partitions = np.ones(input_shape, dtype=int)
            if isinstance(self.num_partitions, np.ndarray) and input_shape == self.num_partitions.shape:
                num_partitions = self.num_partitions
            elif len(input_shape) > 1:
                num_partitions[0,0] = self.num_partitions
            else:
                num_partitions *= self.num_partitions
        slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)
        
        ranges = []
        output_range = None
        
        for element in product(*[range(num) for num in num_partitions.flatten()]):
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[...,0] = input_range[...,0]+np.multiply(element_, slope)
            input_range_[...,1] = input_range[...,0]+np.multiply(element_+1, slope)
            output_range_, info_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1
            
            if output_range is None:
                output_range = np.empty(output_range_.shape)
                output_range[:,0] = np.inf
                output_range[:,1] = -np.inf

            tmp = np.dstack([output_range, output_range_])
            output_range[:,1] = np.max(tmp[:,1,:], axis=1)
            output_range[:,0] = np.min(tmp[:,0,:], axis=1)
            
            ranges.append((input_range_, output_range_))

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)

        return output_range, info

class SimGuidedPartitioner(Partitioner):
    def __init__(self, num_simulations=1000, tolerance_eps=0.01):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.tolerance_eps = tolerance_eps

    def get_output_range(self, input_range, propagator):

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = 'max'
        input_shape = input_range.shape[:-1]
        info = {}

        num_propagator_calls = 0

        # Get initial output reachable set (Line 3)
        output_range, _ = propagator.get_output_range(input_range)
        num_propagator_calls += 1

        M = [(input_range, output_range)] # (Line 4)
        interior_M = []
        
        # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
        # (Line 5)
        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (self.num_simulations,)+input_shape)
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:,0] = np.min(sampled_outputs, axis=0)
        
        u_e = np.empty_like(output_range_sim)
        u_e[:,0] = np.inf
        u_e[:,1] = -np.inf
        while len(M) != 0:
            input_range_, output_range_ = M.pop(0) # Line 9

            if np.all((output_range_sim[...,0] - output_range_[...,0]) <= 0) and \
                np.all((output_range_sim[...,1] - output_range_[...,1]) >= 0):
                # Line 11
                tmp = np.dstack([u_e, output_range_])
                u_e[:,1] = np.max(tmp[:,1,:], axis=1)
                u_e[:,0] = np.min(tmp[:,0,:], axis=1)
                interior_M.append((input_range_, output_range_))
            else:
                # Line 14
                if np.max(input_range_[...,1] - input_range_[...,0]) > self.tolerance_eps:
                    # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                    # Lines 16-17
                    for input_range_ in input_ranges_:
                        output_range_, _ = propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
                        M.append((input_range_, output_range_)) # Line 18
                else: # Lines 19-20
                    M.append((input_range_, output_range_))
                    break

        # Line 24
        if len(M) > 0:
            # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            M_range = np.empty_like(u_e)
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
        
            # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([u_e, M_range])
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)

        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        
        return u_e, info


