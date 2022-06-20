import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn import Sequential, Conv2d, Linear, ReLU, Tanh
from crown_ibp.model_defs import Flatten, model_mlp_any
from crown_ibp.bound_layers import BoundSequential
import torch.nn.functional as F
from itertools import chain
import logging
from itertools import product
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    from tensorflow.keras.models import model_from_json
    from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
    import matplotlib.pyplot as plt
    from nn_closed_loop.utils.nn import load_controller

    model_name = 'random_weight_controller'

    path = "{}/test_controllers/{}".format(dir_path, model_name)
    with open(path + "/model.json", "r") as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(path + "/model.h5")
    torch_model = keras2torch(model, "torch_model")
    # torch_model = load_controller(system='GroundRobotSI', model_name='avoid_origin_controller_simple2')
    num_control_inputs = 1

    ###########################
    # To get NN output bounds:
    print('---')
    print("Example of bounding the NN output associated with an input set.")

    ###########################
    ## 'same_slope: True' makes U of smaller set to be contained by U of larger set,
    ## but I don't think this is the answer
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": False, "zero_lb": True})
    # torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})

    ###########################
    ## Select larger/smaller initial set
    # # x0_min, x0_max, x1_min, x1_max = [-6.75, -5.5, 0, 1.5]
    # # x0_min, x0_max, x1_min, x1_max = [-6.28125, -6.125, 0, 0.1875]
    # x0_min, x0_max, x1_min, x1_max = [-8, -6.75, 0, 1.5]
    x0_min, x0_max, x1_min = [-10, 10, 0]
    x1_max = np.linspace(20, 60, num=500)
    # x0_min, x0_max, x1_min = [0, 10, 0]
    # x1_max = np.linspace(1, 20, num=50)
    umax_list = []
    umin_list = []

    ## Compute matrices for u as function of x
    for x1_maxi in x1_max:
        lower_A, upper_A, lower_sum_b, upper_sum_b = torch_model_.full_backward_range(norm=np.inf,
                                    x_U=torch.Tensor([[x0_max, x1_maxi]]),
                                    x_L=torch.Tensor([[x0_min, x1_min]]),
                                    upper=True, lower=True,
                                    C=torch.eye(num_control_inputs).unsqueeze(0),
                                    return_matrices=True
                                    )
        
        x_max = np.array([x0_max, x1_maxi])
        x_min = np.array([x0_min, x1_min])

        upper_A_numpy = upper_A.detach().numpy()
        upper_sum_b_numpy = upper_sum_b.detach().numpy()
        lower_A_numpy = lower_A.detach().numpy()
        lower_sum_b_numpy = lower_sum_b.detach().numpy()
        umax = np.maximum(upper_A_numpy@x_max+upper_sum_b_numpy, upper_A_numpy@x_min+upper_sum_b_numpy)
        umin = np.minimum(lower_A_numpy@x_max+lower_sum_b_numpy, lower_A_numpy@x_min+lower_sum_b_numpy)
        umax_list.append(umax)
        umin_list.append(umin)
    # print("upper: {}".format(umax))
    # print("lower: {}".format(umin))
    # import pdb; pdb.set_trace()
    fig, axs = plt.subplots(2)
    axs[0].plot(x1_max, np.array(umax_list)[:,0,0])
    axs[0].set_ylabel('u max')
    axs[1].plot(x1_max, np.array(umin_list)[:,0,0])
    axs[1].set_ylabel('u min')
    axs[1].set_xlabel('x max')
    plt.show()



    ######################################################
    # Old evaluation code from bound_layers.py

    # Sample a grid of pts from the input set, to get exact NN output polytope
    # x0 = np.linspace(x0_min, x0_max, num=100)
    # x1 = np.linspace(x1_min, x1_max[-1], num=100)
    # xx,yy = np.meshgrid(x0, x1)
    # pts = np.reshape(np.dstack([xx,yy]), (-1,2))
    # sampled_outputs = torch_model.forward(torch.Tensor(pts))
    # # import pdb; pdb.set_trace()
    # # Print and compare the two bounds numerically    
    # # sampled_output_min = np.min(sampled_outputs.data.numpy()[:,0])
    # # sampled_output_max = np.max(sampled_outputs.data.numpy()[:,0])
    # # crown_min = out_min_crown.data.numpy()[0,0]
    # # crown_max = out_max_crown.data.numpy()[0,0]
    # # print("The sampled outputs lie between: [{},{}]".format(
    # #     sampled_output_min, sampled_output_max))
    # # print("CROWN bounds are: [{},{}]".format(
    # #     crown_min, crown_max))
    # # conservatism_above = crown_max - sampled_output_max
    # # conservatism_below = sampled_output_min - crown_min
    # # print("Over-conservatism: [{},{}]".format(
    # #     conservatism_below, conservatism_above))
    # # print("^ These should both be positive! {}".format(
    # #     "They are :)" if conservatism_above>=0 and conservatism_below>=0 else "*** THEY AREN'T ***"))
    
    # # Plot vertical lines for CROWN bounds, x's for sampled outputs
    # # plt.axvline(out_min_crown.data.numpy()[0,0], ls='--', label='CROWN Bounds')
    # # plt.axvline(out_max_crown.data.numpy()[0,0], ls='--')


    # #####################################################
    # # Plot vx over state space (also uncomment block above)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(pts[:,0], pts[:,1], sampled_outputs.data.numpy()[:,0])

    # print("Showing plot...")
    # plt.show()
    # print('---')

    #
    ###########################