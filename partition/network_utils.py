import numpy as np

def get_sampled_outputs(input_range, propagator, N=1000):
    input_shape = input_range.shape[:-1]
    sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (N,)+input_shape)
    sampled_outputs = propagator.forward_pass(sampled_inputs)
    return sampled_outputs

def samples_to_range(sampled_outputs):
    num_outputs = sampled_outputs.shape[-1]
    output_range = np.empty((num_outputs, 2))
    output_range[:,1] = np.max(sampled_outputs, axis=0)
    output_range[:,0] = np.min(sampled_outputs, axis=0)
    return output_range