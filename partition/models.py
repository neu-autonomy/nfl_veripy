import torch
from torch.nn import Sequential, Conv2d, Linear, ReLU, Tanh

def model_xiang_2017():
    model = Sequential(
        Linear(2, 5),
        Tanh(),
        Linear(5, 2),
    )
    state_dict = model.state_dict()
    state_dict['0.weight'].copy_(torch.nn.Parameter(data=torch.Tensor(
            [
                [-0.9507, -0.7680],
                [0.9707, 0.0270],
                [-0.6876, -0.0626],
                [0.4301, 0.1724],
                [0.7408, -0.7948],
             ]), requires_grad=True))
    state_dict['0.bias'].copy_(torch.nn.Parameter(data=torch.Tensor(
            [1.1836, -0.9087, -0.3463, 0.2626, -0.6768]), requires_grad=True))
    state_dict['2.weight'].copy_(torch.nn.Parameter(data=torch.Tensor(
        [
            [0.8280, 0.6839, 1.0645, -0.0302, 1.7372],
            [1.4436, 0.0824, 0.8721, 0.1490, -1.9154],
        ]), requires_grad=True))
    state_dict['2.bias'].copy_(torch.nn.Parameter(data=torch.Tensor(
        [-1.4048, -0.4827]), requires_grad=True))

    return model

def model_xiang_2020_robot_arm():
    model = Sequential(
        Linear(2, 5),
        ReLU(),
        Linear(5, 2),
    )
    state_dict = model.state_dict()
    state_dict['0.weight'].copy_(torch.nn.Parameter(data=torch.Tensor(
            [
                [-1.87296, -0.02866],
                [-0.84023, -2.25227],
                [-1.10904, -0.6002 ],
                [-0.84835, -1.04995],
                [ 0.07309, -8.852  ],
             ]), requires_grad=True))
    state_dict['0.bias'].copy_(torch.nn.Parameter(data=torch.Tensor(
            [ 3.58326,  5.82976,  2.09246,  2.65733, 13.50541]), requires_grad=True))
    state_dict['2.weight'].copy_(torch.nn.Parameter(data=torch.Tensor(
        [
            [ 2.04445, -1.86677, 14.2524 , -4.47312, -0.01326],
            [ 3.18875,  1.1107 , -5.24184,  8.51545,  0.00277],
        ]), requires_grad=True))
    state_dict['2.bias'].copy_(torch.nn.Parameter(data=torch.Tensor(
        [-0.52256, 7.34787]), requires_grad=True))

    return model