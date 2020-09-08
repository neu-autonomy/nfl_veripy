import numpy as np
from partition.Partitioner import Partitioner, UniformPartitioner

class ClosedLoopPartitioner(Partitioner):
    def __init__(self, At=None, bt=None, ct=None):
        Partitioner.__init__(self)
        self.At = At
        self.bt = bt
        self.ct = ct

    def get_one_step_reachable_set(self, A_inputs, b_inputs, A_out, propagator):
        reachable_set, info = propagator.get_one_step_reachable_set(A_inputs, b_inputs, A_out)
        return reachable_set, info

    def get_reachable_set(self, A_inputs, b_inputs, A_out, propagator, t_max):
        reachable_set, info = propagator.get_reachable_set(A_inputs, b_inputs, A_out, t_max)
        return reachable_set, info

class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, At=None, bt=None, ct=None):
        ClosedLoopPartitioner.__init__(self, At=At, bt=bt, ct=ct)

class ClosedLoopUniformPartitioner(ClosedLoopPartitioner):
    def __init__(self, num_partitions=16, At=None, bt=None, ct=None):
        ClosedLoopPartitioner.__init__(self, At=At, bt=bt, ct=ct)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    # def get_output_range(self, input_range, propagator, num_partitions=None):
    def get_one_step_reachable_set(self, A_inputs, b_inputs, A_out, propagator):
        raise NotImplementedError
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
