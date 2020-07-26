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

# def stablebaselines2torch(sess, obs_ph, numlayer, w_tsr, b_tsr, activation='relu'):
def stablebaselines2torch(good_sess, network_params, activation='relu'):
    import torch
    import tensorflow as tf

    obs_ph, numlayer, w_tsr, b_tsr = network_params

    modules = []
    # with tf.compat.v1.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)
    for i in range(len(w_tsr)):
        w = good_sess.run(w_tsr[i]).T
        b = good_sess.run(b_tsr[i])
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        # print("Layer {}: Input: {}, Output: {}".format(i, w.shape[1], w.shape[0]))
        # print("Bias {}: shape: {}".format(b, b.shape))
        # print(w)
        # import pdb; pdb.set_trace()
        linear.weight.data.copy_(torch.Tensor(w))
        linear.bias.data.copy_(torch.Tensor(b))
        modules.append(linear)
        if i < len(w_tsr) - 1:
            # print('adding relu...')
            modules.append(torch.nn.ReLU())
        # else:
            # print('not adding relu.')
    torch_model = torch.nn.Sequential(*modules)
    return torch_model
