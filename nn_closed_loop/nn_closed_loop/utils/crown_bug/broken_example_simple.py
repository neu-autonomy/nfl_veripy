from hashlib import pbkdf2_hmac
from importlib import import_module
from turtle import pd
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

if __name__ == '__main__':
    from tensorflow.keras.models import model_from_json
    from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
    import matplotlib.pyplot as plt
    from nn_closed_loop.utils.nn import load_controller

    torch_model = load_controller(system='debug', model_name='corner_policy3')
    num_control_inputs = 1

    ###########################
    # To get NN output bounds:
    print('---')
    print("Example of bounding the NN output associated with an input set.")

    ###########################
    ## 'same_slope: True' makes U of smaller set to be contained by U of larger set,
    ## but I don't think this is the answer
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": False, "zero-lb": True})
    # torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
    # torch_model_ = BoundSequential.convert(torch_model, {"same-slope": False, "one-lb": True})
    # torch_model_ = BoundSequential.convert(torch_model, {})

    # for name, para in torch_model_.named_parameters():
    #     print('{}: {}'.format(name, para.shape))
    # import pdb; pdb.set_trace()
    ###########################
    ## Select larger/smaller initial set
    # # x0_min, x0_max, x1_min, x1_max = [-6.75, -5.5, 0, 1.5]
    # x0_min, x0_max, x1_min, x1_max = [-6.28125, -6.125, 0, 0.1875]
    # num_control_inputs = 2

    # lower_A, upper_A, lower_sum_b, upper_sum_b = torch_model_.full_backward_range(norm=np.inf,
    #     x_U=torch.Tensor([[x0_max, x1_max]]),
    #     x_L=torch.Tensor([[x0_min, x1_min]]),
    #     upper=True, lower=True,
    #     C=torch.eye(num_control_inputs).unsqueeze(0),
    #     return_matrices=True
    #     )
    # x_max = np.array([[x0_max, x1_max]])
    # x_min = np.array([[x0_min, x1_min]])
    # upper_A_numpy = upper_A.detach().numpy()
    # upper_sum_b_numpy = upper_sum_b.detach().numpy()
    # lower_A_numpy = lower_A.detach().numpy()
    # lower_sum_b_numpy = lower_sum_b.detach().numpy()
    # umax = np.maximum(upper_A_numpy@x_max+upper_sum_b_numpy, upper_A_numpy@x_min+upper_sum_b_numpy)
    # umin = np.minimum(lower_A_numpy@x_max+lower_sum_b_numpy, lower_A_numpy@x_min+lower_sum_b_numpy)

    # import pdb; pdb.set_trace()
    # x0_min, x0_max, x1_min, x1_max = [-8, -6.75, 0, 1.5]
    x0_min = -25
    x0_max = np.linspace(-25, 10, num=100) # [10]
    # x1_max = np.linspace(0.05, 0.5, num=100)
    umax_list = []
    umin_list = []
    a_list = []
    b_list = []

    ## Compute matrices for u as function of x
    for x0_maxi in x0_max:
        # test = torch.Tensor([[x_0min]])
        # import pdb; pdb.set_trace()
        lower_A, upper_A, lower_sum_b, upper_sum_b = torch_model_.full_backward_range(norm=np.inf,
                                    x_U=torch.Tensor([[x0_maxi]]),
                                    x_L=torch.Tensor([[x0_min]]),
                                    upper=True, lower=True,
                                    C=torch.eye(num_control_inputs).unsqueeze(0),
                                    return_matrices=True
                                    )
        # import pdb; pdb.set_trace()
        x_max = np.array([[x0_maxi]])
        x_min = np.array([[x0_min]])
        

        upper_A_numpy = upper_A.detach().numpy()
        upper_sum_b_numpy = upper_sum_b.detach().numpy()
        lower_A_numpy = lower_A.detach().numpy()
        lower_sum_b_numpy = lower_sum_b.detach().numpy()
        umax = np.maximum(upper_A_numpy@x_max+upper_sum_b_numpy, upper_A_numpy@x_min+upper_sum_b_numpy)
        umin = np.minimum(lower_A_numpy@x_max+lower_sum_b_numpy, lower_A_numpy@x_min+lower_sum_b_numpy)
        # import pdb; pdb.set_trace()
        a_list.append(lower_A_numpy[0,0,:])
        b_list.append(lower_sum_b_numpy[0,0])
        umax_list.append(umin)
        umin_list.append(umax)
    # print("upper: {}".format(umax))
    # print("lower: {}".format(umin))

    # plt.plot(x0_max, np.array(umax_list)[:,0,0,0])
    # plt.xlabel('x max')
    # plt.ylabel('u max')
    # plt.show()
    # # import pdb; pdb.set_trace()

    # fig, axs = plt.subplots(3,1)
    # axs[0].plot(x0_max, a_list)
    # axs[1].plot(x0_max, b_list)
    # axs[2].plot(x0_max, np.array(umax_list)[:,0,0,0])
    # plt.show()



    ######################################################
    # Old evaluation code from bound_layers.py

    # Sample a grid of pts from the input set, to get exact NN output polytope
    pts = np.linspace(x_min, x_max[-1], num=1000)
    sampled_outputs = torch_model.forward(torch.Tensor(pts))
    # import pdb; pdb.set_trace()
    # Print and compare the two bounds numerically    
    # sampled_output_min = np.min(sampled_outputs.data.numpy()[:,0])
    # sampled_output_max = np.max(sampled_outputs.data.numpy()[:,0])
    # crown_min = out_min_crown.data.numpy()[0,0]
    # crown_max = out_max_crown.data.numpy()[0,0]
    # print("The sampled outputs lie between: [{},{}]".format(
    #     sampled_output_min, sampled_output_max))
    # print("CROWN bounds are: [{},{}]".format(
    #     crown_min, crown_max))
    # conservatism_above = crown_max - sampled_output_max
    # conservatism_below = sampled_output_min - crown_min
    # print("Over-conservatism: [{},{}]".format(
    #     conservatism_below, conservatism_above))
    # print("^ These should both be positive! {}".format(
    #     "They are :)" if conservatism_above>=0 and conservatism_below>=0 else "*** THEY AREN'T ***"))
    
    # Plot vertical lines for CROWN bounds, x's for sampled outputs
    # plt.axvline(out_min_crown.data.numpy()[0,0], ls='--', label='CROWN Bounds')
    # plt.axvline(out_max_crown.data.numpy()[0,0], ls='--')


    #####################################################
    # Plot vx over state space (also uncomment block above)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(pts[:,0], pts[:,1], sampled_outputs.data.numpy()[:,0])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('vx')

    # print("Showing plot...")
    # plt.show()
    # print('---')
    # # import pdb; pdb.set_trace()
    # fig, ax = plt.subplots(1)
    # ax.plot(pts[:,0,0], sampled_outputs.data.numpy()[:,0,0])
    # ax.set_xlabel('y')
    # ax.set_ylabel('vx')
    # # ax.set_zlabel('vx')
    # print("Showing plot...")
    # plt.show()
    # print('---')

    # switch_lower = 0.21
    # switch_upper = 0.24
    # switch_lower = 0.168
    # switch_upper = 0.2
    switch_lower = -15
    switch_upper = 0
    switch_3 = 10

    index1 = np.where(pts<=switch_lower)[0][-1]
    index2 = np.where(x0_max<=switch_lower)[0][-1]
    xs = pts[0:index1,0,0]
    vxs = sampled_outputs.data.numpy()[0:index1,0,0]
    ubound = pts[0:index1,0,0]*a_list[index2]+b_list[index2]
    # import pdb; pdb.set_trace()

    xlim, ylim = [-30, 15], [-3,-1.7]
    fig, ax = plt.subplots(3,1)
    ax[0].plot(xs, vxs, label = 'u')
    ax[0].plot(pts[0:index1,0,0], ubound, label = 'lower_bound')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('u')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].legend()

    index1 = np.where(pts<=switch_upper)[0][-1]
    index2 = np.where(x0_max<=switch_upper)[0][-1]
    
    xs = pts[0:index1,0,0]
    vxs = sampled_outputs.data.numpy()[0:index1,0,0]
    ubound = pts[0:index1,0,0]*a_list[index2]+b_list[index2]

    ax[1].plot(xs, vxs)
    ax[1].plot(pts[0:index1,0,0], ubound)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('u')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)


    index1 = np.where(pts<=switch_3)[0][-1]
    index2 = np.where(x0_max<=switch_3)[0][-1]
    
    xs = pts[0:index1,0,0]
    vxs = sampled_outputs.data.numpy()[0:index1,0,0]
    ubound = pts[0:index1,0,0]*a_list[index2]+b_list[index2]

    # import pdb; pdb.set_trace()
    
    ax[2].plot(xs, vxs)
    ax[2].plot(pts[0:index1,0,0], ubound)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('u')
    ax[2].set_xlim(xlim)
    ax[2].set_ylim(ylim)

    
    print("Showing plot...")
    plt.show()
    print('---')

    switch = -24
    index1 = np.where(pts<=switch)[0][-1]
    index2 = np.where(x0_max<=switch)[0][-1]
    xs = pts[0:index1,0,0]
    vxs = sampled_outputs.data.numpy()[0:index1,0,0]
    ubound = pts[0:index1,0,0]*a_list[index2]+b_list[index2]
    # import pdb; pdb.set_trace()

    plt.ion()
    fig, ax = plt.subplots(1,1)
    data, = ax.plot(xs, vxs)
    line, = ax.plot(pts[0:index1,0,0], ubound)
    ax.set_xlabel('y')
    ax.set_ylabel('vx')

    ax.set_xlim([-30,30])
    ax.set_ylim([-4,-1])


    for i in range(-23,50):
        switch = i

        index1 = np.where(pts<=switch)[0][-1]
        index2 = np.where(x0_max<=switch)[0][-1]
        xs = pts[0:index1,0,0]
        vxs = sampled_outputs.data.numpy()[0:index1,0,0]
        ubound = pts[0:index1,0,0]*a_list[index2]+b_list[index2]
        # import pdb; pdb.set_trace()

        # fig, ax = plt.subplots(1,1)
        # ax.plot(xs, vxs)
        # line = ax.plot(pts[0:index1,0,0], ubound)
        # ax.set_xlabel('y')
        # ax.set_ylabel('vx')
        line.set_xdata(pts[0:index1,0,0])
        line.set_ydata(ubound)

        data.set_xdata(xs)
        data.set_ydata(vxs)

        fig.canvas.draw()
        

        fig.canvas.flush_events()
        import time
        time.sleep(0.1)
        



    #
    ###########################