import torch
from torch.nn import Sequential, Linear, ReLU, Tanh, Sigmoid
import os

model_dir = "{}/model_files".format(os.path.dirname(os.path.abspath(__file__)))

activations = {"tanh": Tanh, "relu": ReLU, "sigmoid": Sigmoid}


def model_dynamics(env_name="CartPole-v0"):
    from partition.dynamics import load_model
    from crown_ibp.conversions.keras2torch import keras2torch

    model = load_model(env_name + "_model")
    torch_model = keras2torch(model, "torch_model")
    return torch_model


def model_xiang_2017(activation="tanh"):
    neurons = [2, 5, 2]
    model = Sequential(
        Linear(neurons[0], neurons[1]),
        activations[activation](),
        Linear(neurons[1], neurons[2]),
    )
    state_dict = model.state_dict()
    state_dict["0.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [-0.9507, -0.7680],
                    [0.9707, 0.0270],
                    [-0.6876, -0.0626],
                    [0.4301, 0.1724],
                    [0.7408, -0.7948],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["0.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([1.1836, -0.9087, -0.3463, 0.2626, -0.6768]),
            requires_grad=True,
        )
    )
    state_dict["2.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [0.8280, 0.6839, 1.0645, -0.0302, 1.7372],
                    [1.4436, 0.0824, 0.8721, 0.1490, -1.9154],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["2.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([-1.4048, -0.4827]), requires_grad=True
        )
    )

    model_info = make_model_info_dict(neurons=neurons, activation=activation)

    return model, model_info


def model_xiang_2020_robot_arm(activation="tanh"):

    neurons = [2, 5, 2]
    model = Sequential(
        Linear(neurons[0], neurons[1]),
        activations[activation](),
        Linear(neurons[1], neurons[2]),
    )
    state_dict = model.state_dict()
    state_dict["0.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [-1.87296, -0.02866],
                    [-0.84023, -2.25227],
                    [-1.10904, -0.6002],
                    [-0.84835, -1.04995],
                    [0.07309, -8.852],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["0.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([3.58326, 5.82976, 2.09246, 2.65733, 13.50541]),
            requires_grad=True,
        )
    )
    state_dict["2.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [2.04445, -1.86677, 14.2524, -4.47312, -0.01326],
                    [3.18875, 1.1107, -5.24184, 8.51545, 0.00277],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["2.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([-0.52256, 7.34787]), requires_grad=True
        )
    )

    model_info = make_model_info_dict(neurons=neurons, activation=activation)

    return model, model_info


def make_model_info_dict(neurons=None, activation=None, seed=None):
    model_info = {
        "model_neurons": neurons,
        "model_activation": activation,
        "model_seed": seed,
    }
    return model_info


def model_simple():
    model = Sequential(Linear(2, 2), Tanh(), Linear(2, 2))
    state_dict = model.state_dict()
    state_dict["0.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["0.bias"].copy_(
        torch.nn.Parameter(data=torch.Tensor([0.0, 0.0]), requires_grad=True)
    )
    state_dict["2.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [1.0, 0.0],
                    [-0.2, 1.0],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["2.bias"].copy_(
        torch.nn.Parameter(data=torch.Tensor([0.0, 0.0]), requires_grad=True)
    )
    return model


def model_gh1():
    model = Sequential(
        Linear(2, 6),
        ReLU(),
        Linear(6, 2),
    )
    state_dict = model.state_dict()
    state_dict["0.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [-1.8, 0.001],
                    [-1.4, 0.002],
                    [-4, 5],
                    [-4, -1],
                    [-2, 1.3],
                    [-1, -1.2],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["0.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([5, 5, 7, 8, -6, 15]), requires_grad=True
        )
    )
    state_dict["2.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [1, 2.04445, -1.86677, 2, -1.47312, 0],
                    [1, -2, -1, -5, 8.0, -2],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["2.bias"].copy_(
        torch.nn.Parameter(data=torch.Tensor([-0.3, 1]), requires_grad=True)
    )

    return model


def model_gh2():
    model = Sequential(
        Linear(2, 6),
        ReLU(),
        Linear(6, 5),
        ReLU(),
        Linear(5, 2),
    )
    state_dict = model.state_dict()
    state_dict["0.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [-1.8, 0.1],
                    [-1.4, 0.2],
                    [-2, 5],
                    [-4, -1],
                    [-2, 1.3],
                    [-1, -1.6],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["0.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([5, 5, 7, 8, -6, 15]), requires_grad=True
        )
    )
    state_dict["2.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [1, 2.03, -1.2, 2, -1.47312, 0],
                    [-1.3, 21, 1.3, -5, 8.0, -2],
                    [1, -2, -4, -5, 8.0, -2],
                    [1, -2, 1.4, -1, 4.3, -2],
                    [1.3, 2, 5.6, -3.4, 2.5, 6.4],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["2.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([-0.3, 3, 0.3, 4.1, 1]), requires_grad=True
        )
    )
    state_dict["4.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [1, 2.04445, -1.86677, 2, -1.47312],
                    [1, -2, -1, -5, 8],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["4.bias"].copy_(
        torch.nn.Parameter(data=torch.Tensor([-0.3, 1]), requires_grad=True)
    )

    return model


def random_model(activation="relu", neurons=[2, 5, 20, 40, 40, 20, 2], seed=0):
    filename = get_model_filename(
        activation=activation, neurons=neurons, seed=seed
    )
    try:
        model = torch.load(filename)
    except:
        torch.manual_seed(seed)
        layers = []
        for i in range(len(neurons) - 1):
            layers.append(Linear(neurons[i], neurons[i + 1]))
            if i != len(neurons) - 2:
                if activation == "relu":
                    layers.append(ReLU())
        model = Sequential(*layers)
        filename = get_model_filename(
            activation=activation, neurons=neurons, seed=seed
        )
        torch.save(model, filename)
    model_info = make_model_info_dict(
        neurons=neurons, activation=activation, seed=seed
    )
    return model, model_info


def get_model_filename(
    activation="relu", neurons=[2, 5, 20, 40, 40, 20, 2], seed=0
):
    filename = "_".join(map(str, neurons)) + "_" + activation + "_" + str(seed)
    filename = os.path.join(model_dir, filename)
    return filename


def lstm(hidden_size=64, num_classes=10, input_size=784, num_slices=8, seed=0):
    torch.manual_seed(seed)
    # A disastrous hack...
    import sys
    import os
    import auto_LiRPA

    sequence_path = (
        os.path.dirname(os.path.dirname(auto_LiRPA.__file__))
        + "/examples/sequence"
    )
    sys.path.append(sequence_path)
    from lstm import LSTM
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cuda", "cpu"]
    )
    parser.add_argument("--norm", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_slices", type=int, default=num_slices)
    parser.add_argument("--hidden_size", type=int, default=hidden_size)
    parser.add_argument("--num_classes", type=int, default=num_classes)
    parser.add_argument("--input_size", type=int, default=input_size)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument(
        "--dir",
        type=str,
        default=sequence_path + "/model",
        help="directory to load or save the model",
    )
    parser.add_argument(
        "--num_epochs_warmup",
        type=int,
        default=1,
        help="number of epochs for the warmup stage when eps is \
            linearly increased from 0 to the full value",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="interval of printing the log during training",
    )
    args = parser.parse_args()
    torch_model = LSTM(args).to(args.device)

    info = {
        "model_neurons": [input_size, num_classes],
    }

    return torch_model, info


def model_gh3():
    model = Sequential(
        Linear(2, 6),
        ReLU(),
        Linear(6, 2),
    )
    state_dict = model.state_dict()
    state_dict["0.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [-1.8, 0.001],
                    [-1.4, 0.002],
                    [-4, 5],
                    [-4, -1],
                    [-2, 3.3],
                    [-3, -1.2],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["0.bias"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor([5, 5, 7, 8, -6, 15]), requires_grad=True
        )
    )
    state_dict["2.weight"].copy_(
        torch.nn.Parameter(
            data=torch.Tensor(
                [
                    [1, 2.04445, -1.86677, 2, -1.47312, 0],
                    [1, -2, -1, -5, 8.0, -2],
                ]
            ),
            requires_grad=True,
        )
    )
    state_dict["2.bias"].copy_(
        torch.nn.Parameter(data=torch.Tensor([-0.3, 1]), requires_grad=True)
    )

    return model
