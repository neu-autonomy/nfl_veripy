import numpy as np
import time


def bisect(input_range):
    return sect(input_range, num_sects=2)


def sect(input_range, num_sects=3, select="random"):
    input_shape = input_range.shape[:-1]
    if select == "random":
        # doesnt work
        input_dim_to_sect = np.random.randint(0, num_inputs)
    else:
        lengths = input_range[..., 1] - input_range[..., 0]
        input_dim_to_sect = np.unravel_index(lengths.argmax(), lengths.shape)
    input_ranges = np.tile(
        input_range,
        (num_sects,) + tuple([1 for _ in range(len(input_shape) + 1)]),
    )
    diff = (
        input_range[input_dim_to_sect + (1,)]
        - input_range[input_dim_to_sect + (0,)]
    ) / float(num_sects)
    for i in range(num_sects - 1):
        new_endpt = input_range[input_dim_to_sect + (0,)] + (i + 1) * diff
        input_ranges[(i,) + input_dim_to_sect + (1,)] = new_endpt
        input_ranges[(i + 1,) + input_dim_to_sect + (0,)] = new_endpt
    return input_ranges


def get_sampled_outputs(input_range, propagator, N=1000):
    input_shape = input_range.shape[:-1]
    sampled_inputs = np.random.uniform(
        input_range[..., 0], input_range[..., 1], (N,) + input_shape
    )

    sampled_outputs = propagator.forward_pass(sampled_inputs)

    return sampled_outputs


def samples_to_range(sampled_outputs):
    num_outputs = sampled_outputs.shape[-1]
    output_range = np.empty((num_outputs, 2))
    output_range[:, 1] = np.max(sampled_outputs, axis=0)
    output_range[:, 0] = np.min(sampled_outputs, axis=0)
    return output_range


def stablebaselines2torch(good_sess, network_params, activation="relu"):
    import torch

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
