from robust_sdp.network import Network
from robust_sdp.robust_tools import test_robustness

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from nn_partition.models.models import (
    model_xiang_2017,
    model_xiang_2020_robot_arm,
)


def mfe_keras2torch():
    from tensorflow.keras.models import model_from_json
    from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model

    # load json and create model
    json_file = open("/Users/mfe/Downloads/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/Users/mfe/Downloads/model.h5")
    print("Loaded model from disk")

    torch_model = keras2torch(model, "torch_model")
    return torch_model


def torch2net(torch_model):

    dims = []
    act = None
    for idx, m in enumerate(torch_model.modules()):
        if isinstance(m, torch.nn.Sequential):
            continue
        elif isinstance(m, torch.nn.ReLU):
            if act is None or act == "relu":
                act = "relu"
            else:
                print(
                    "Multiple types of activations in your model --- unsuported by robust_sdp."
                )
                assert 0
        elif isinstance(m, torch.nn.Linear):
            dims.append(m.in_features)
        else:
            print("That layer isn't supported.")
            assert 0
    dims.append(m.out_features)
    if len(dims) != 3:
        print("robust sdp only supports 1 hidden layer (for now).")
        assert 0
    net = Network(dims, act, "rand")

    for name, param in torch_model.named_parameters():
        layer, typ = name.split(".")
        layer = int(int(layer) / 2)
        if typ == "weight":
            net._W[layer] = param.data.numpy()
        elif typ == "bias":
            net._b[layer] = np.expand_dims(param.data.numpy(), axis=-1)
        else:
            print("this layer isnt a weight or bias ???")

    return net


def robust_sdp(
    net=None, epsilon=0.1, viz=False, input_range=None, verbose=True
):

    if net is None:
        in_dim, out_dim = 2, 2
        hidden_dim = 5
        net = Network(
            [in_dim, hidden_dim, out_dim], "relu", "rand", load_weights=False
        )

    lower, upper, out_data = test_robustness(
        net,
        epsilon=epsilon,
        input_range=input_range,
        parallel=True,
        verbose=verbose,
    )
    if viz:
        plot_robust_sdp(lower, upper, out_data)

    output_range = np.vstack([lower, upper]).T

    return output_range


def plot_robust_sdp(lower, upper, out_data):
    plt.scatter(out_data[0, :], out_data[1, :])
    rect = Rectangle(
        lower,
        upper[0] - lower[0],
        upper[1] - lower[1],
        fc="none",
        linewidth=1,
        edgecolor="k",
    )
    plt.gca().add_patch(rect)
    plt.show()


def example_robust_sdp(viz=False):
    torch_model = model_xiang_2020_robot_arm()
    net = torch2net(torch_model)

    # data = np.array([[1.5, 2.5], [2.,3.], [3.,5.]])
    # torch_out = torch_model(torch.Tensor([data]))
    # print("torch_out:", torch_out)

    # net_out = net.forward_prop(data.T).T
    # print("net_out:", net_out)

    input_range = np.array(
        [  # (num_inputs, 2)
            [np.pi / 3, 2 * np.pi / 3],  # x0min, x0max
            [np.pi / 3, 2 * np.pi / 3],  # x1min, x1max
        ]
    )
    robust_sdp(net=net, input_range=input_range, viz=viz, verbose=True)


if __name__ == "__main__":
    example_robust_sdp(viz=True)
    # torch_model = mfe_keras2torch()

    # robust_sdp()
