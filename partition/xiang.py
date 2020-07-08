import torch
import numpy as np
from partition.models import model_xiang_2017, model_xiang_2020_robot_arm
from crown_ibp.bound_layers import BoundSequential

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product

def simulation_guided_partition(model, input_range, num_outputs, viz=False, bound_method="ibp"):
    # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
    tolerance_eps = 0.01
    sect_method = 'max'
    num_inputs = input_range.shape[0]
    
    # Get initial output reachable set (Line 3)
    output_range = get_output_range(model, input_range, num_outputs, bound_method=bound_method)
    
    M = [(input_range, output_range)] # (Line 4)
    interior_M = []
    
    # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
    # (Line 5)
    N = 1000
    sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
    sampled_outputs = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
    
    # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
    output_range_sim = np.empty((num_outputs, 2))
    output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
    output_range_sim[:,0] = np.min(sampled_outputs, axis=0)
    
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    while len(M) != 0:
        input_range_, output_range_ = M.pop(0) # Line 9
        
        if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0):
            # Line 11
            tmp = np.dstack([u_e, output_range_])
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)
            interior_M.append((input_range_, output_range_))
        else:
            # Line 14
            if np.max(input_range_[:,1] - input_range_[:,0]) > tolerance_eps:
                # Line 15
                input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                for input_range_ in input_ranges_:
                    output_range_ = get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
                    M.append((input_range_, output_range_)) # Line 18
            else: # Lines 19-20
                break
    
    # Line 24
    if len(M) > 0:
        # Squash all of M down to one range
        M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
        M_range = np.empty((num_outputs, 2))
        M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
        M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
    
        # Combine M (remaining ranges) with u_e (interior ranges)
        tmp = np.dstack([u_e, M_range])
        u_e[:,1] = np.max(tmp[:,1,:], axis=1)
        u_e[:,0] = np.min(tmp[:,0,:], axis=1)
    
    if viz:
        visualize_partitions(sampled_outputs, u_e, input_range, M=M, interior_M=interior_M, output_range_sim=output_range_sim)
    
    return u_e

def uniform_partition(model, input_range, num_outputs, viz=False, bound_method="ibp"):
    num_inputs = input_range.shape[0]
    num_partitions = 16*np.ones((num_inputs,), dtype=int)
    slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
    
    ranges = []
    
    u_e = np.empty((num_outputs, 2))
    u_e[:,0] = np.inf
    u_e[:,1] = -np.inf
    for element in product(*[range(num) for num in num_partitions]):
        input_range_ = np.empty_like(input_range)
        input_range_[:,0] = input_range[:,0]+np.multiply(element, slope)
        input_range_[:,1] = input_range[:,0]+np.multiply(np.array(element)+1, slope)
        output_range_ = get_output_range(model, input_range_, num_outputs, bound_method=bound_method)
        
        tmp = np.dstack([u_e, output_range_])
        u_e[:,1] = np.max(tmp[:,1,:], axis=1)
        u_e[:,0] = np.min(tmp[:,0,:], axis=1)
        
        ranges.append((input_range_, output_range_))
    
    if viz:
        N = 1000
        sampled_inputs = np.random.uniform(input_range[:,0], input_range[:,1], (N,num_inputs,))
        sampled_outputs = model(torch.Tensor(sampled_inputs), method_opt=None).data.numpy()
        visualize_partitions(sampled_outputs, u_e, input_range, interior_M=ranges)
    
    return u_e


def visualize_partitions(sampled_outputs, estimated_output_range, input_range, output_range_sim=None, interior_M=None, M=None):
    fig, axes = plt.subplots(1,2)
    if interior_M is not None and len(interior_M) > 0:
        for (input_range_, output_range_) in interior_M:
            rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            axes[1].add_patch(rect)

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='m')
            axes[0].add_patch(rect)
    if M is not None and len(M) > 0:
        for (input_range_, output_range_) in M:
            rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            axes[1].add_patch(rect)

            rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
                    fc='none', linewidth=1,edgecolor='b')
            axes[0].add_patch(rect)

    axes[1].scatter(sampled_outputs[:,0], sampled_outputs[:,1], c='k', zorder=2)

    rect = Rectangle(estimated_output_range[:,0], estimated_output_range[0,1]-estimated_output_range[0,0], estimated_output_range[1,1]-estimated_output_range[1,0],
                    fc='none', linewidth=2,edgecolor='g')
    axes[1].add_patch(rect)

    if output_range_sim is not None:
        rect = Rectangle(output_range_sim[:,0], output_range_sim[0,1]-output_range_sim[0,0], output_range_sim[1,1]-output_range_sim[1,0],
                        fc='none', linewidth=1,edgecolor='k')
        axes[1].add_patch(rect)

    axes[0].set_xlim(input_range[0,0], input_range[0,1])
    axes[0].set_ylim(input_range[1,0], input_range[1,1])

    plt.show()

def get_output_range(model, input_range, num_outputs, bound_method="ibp"):
    method_dict = {
        "crown": "full_backward_range",
        "ibp": "interval_range",
    }
    output_range = np.empty((num_outputs,2))
    for out_index in range(num_outputs):
        C = torch.zeros((1,1,num_outputs))
        C[0,0,out_index] = 1
        out_max, out_min = model(norm=np.inf,
                                    x_U=torch.Tensor([input_range[:,1]]),
                                    x_L=torch.Tensor([input_range[:,0]]),
                                    C=C,
                                    method_opt=method_dict[bound_method],
                                    )[:2]
        output_range[out_index,:] = [out_min, out_max]
    return output_range

def bisect(input_range):
    return sect(input_range, num_sects=2)

def sect(input_range, num_sects=3, select='random'):
    num_inputs = input_range.shape[0]
    if select == 'random':
        input_dim_to_sect = np.random.randint(0, num_inputs)
    else:
        input_dim_to_sect = np.argmax(input_range[:,1] - input_range[:,0])
    input_ranges = np.tile(input_range, (num_sects,1,1,))
    diff = (input_range[input_dim_to_sect,1]-input_range[input_dim_to_sect,0])/float(num_sects)
    for i in range(num_sects-1):
        new_endpt = input_range[input_dim_to_sect,0]+(i+1)*diff
        input_ranges[i,input_dim_to_sect,1] = new_endpt
        input_ranges[i+1,input_dim_to_sect,0] = new_endpt
    return input_ranges

def xiang2020example(bound_method="ibp"):
    
    torch_model = model_xiang_2020_robot_arm()
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})

    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3] # x1min, x1max
    ])
    num_outputs = 2
    
    output_range_simulation_guided = simulation_guided_partition(torch_model_, input_range, num_outputs, viz=True, bound_method=bound_method)
    output_range_uniform = uniform_partition(torch_model_, input_range, num_outputs, viz=True, bound_method=bound_method)


def xiang2017example():
    '''
    Goal: Confirm that IBP yields the same bounds as Xiang's method for propagating uncertainty via maximum sensitivity
    '''
    
    torch_model = model_xiang_2017()
    torch_model = model_xiang_2020_robot_arm()

    # Example querying a single pt
    x = torch.Tensor([0,0])
    # print(torch_model(x))

    # Define input set
    x0_min, x0_max, x1_min, x1_max = [np.pi/3, 2*np.pi/3, np.pi/3, 2*np.pi/3]
    # x0_min, x0_max, x1_min, x1_max = [0, 1, 0, 1]

    # Sample a grid of pts from the input set, to get exact NN output polytope
    x0 = np.linspace(x0_min, x0_max, num=100)
    x1 = np.linspace(x1_min, x1_max, num=100)
    xx,yy = np.meshgrid(x0, x1)
    pts = np.reshape(np.dstack([xx,yy]), (-1,2))
    sampled_outputs = torch_model.forward(torch.Tensor(pts)).data.numpy()

    plt.scatter(sampled_outputs[:,0], sampled_outputs[:,1], marker='.', label="Samples")

    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})

    # Evaluate IBP bounds
    out_max_ibp_x0, out_min_ibp_x0 = torch_model_.interval_range(norm=np.inf,
                                x_U=torch.Tensor([[x0_max, x1_max]]),
                                x_L=torch.Tensor([[x0_min, x1_min]]),
                                C=torch.Tensor([[[1, 0]]]),
                                )[:2]
    out_max_ibp_x1, out_min_ibp_x1 = torch_model_.interval_range(norm=np.inf,
                                x_U=torch.Tensor([[x0_max, x1_max]]),
                                x_L=torch.Tensor([[x0_min, x1_min]]),
                                C=torch.Tensor([[[0, 1]]]),
                                )[:2]
    # print(out_max_ibp_x0, out_max_ibp_x1)
    # print(out_min_ibp_x0, out_min_ibp_x1)

    plt.axvline(out_min_ibp_x0.data.numpy()[0,0], ls='--')
    plt.axvline(out_max_ibp_x0.data.numpy()[0,0], ls='--')
    plt.axhline(out_min_ibp_x1.data.numpy()[0,0], ls='--')
    plt.axhline(out_max_ibp_x1.data.numpy()[0,0], ls='--')

    num_x0 = 3; num_x1 = 3;
    for i in range(num_x0):
        for j in range(num_x1):
            x0_slope = (x0_max - x0_min)/num_x0
            x1_slope = (x1_max - x1_min)/num_x1
            init_state_range_ = np.array([[x0_min+x0_slope*i, x0_min+x0_slope*(i+1)],
                                        [x1_min+x1_slope*j, x1_min+x1_slope*(j+1)]])
            # Evaluate IBP bounds
            out_max_ibp_x0, out_min_ibp_x0 = torch_model_.interval_range(norm=np.inf,
                                        x_U=torch.Tensor([init_state_range_[:,1]]),
                                        x_L=torch.Tensor([init_state_range_[:,0]]),
                                        C=torch.Tensor([[[1, 0]]]),
                                        )[:2]
            out_max_ibp_x1, out_min_ibp_x1 = torch_model_.interval_range(norm=np.inf,
                                        x_U=torch.Tensor([init_state_range_[:,1]]),
                                        x_L=torch.Tensor([init_state_range_[:,0]]),
                                        C=torch.Tensor([[[0, 1]]]),
                                        )[:2]
            # plt.axvline(out_min_ibp_x0.data.numpy()[0,0])
            # plt.axvline(out_max_ibp_x0.data.numpy()[0,0])
            # plt.axhline(out_min_ibp_x1.data.numpy()[0,0])
            # plt.axhline(out_max_ibp_x1.data.numpy()[0,0])
            rect = Rectangle([out_min_ibp_x0, out_min_ibp_x1], out_max_ibp_x0-out_min_ibp_x0, out_max_ibp_x1-out_min_ibp_x1,
                linewidth=1,edgecolor='r',facecolor='none')
            plt.gca().add_patch(rect)

    plt.show()

if __name__ == '__main__':
	xiang2017example()

	print("CROWN...")
	xiang2020example(bound_method="crown")
	print("IBP...")
	xiang2020example(bound_method="ibp")

