import numpy as np
np.set_printoptions(precision=2)
from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
from reach_lp.nn_bounds import BoundClosedLoopController
import torch
import time
import pypoman

def init_state_range_to_polytope(init_state_range):

    pts = []
    for i in range(num_states):
        for j in range(num_states):
            pts.append([init_state_range[0,i], init_state_range[1,j]])
    vertices = np.array(pts)
    A_inputs, b_inputs = pypoman.compute_polytope_halfspaces(vertices)
    return A_inputs, b_inputs

def reachLP_1(torch_model_cl, A_inputs, b_inputs, At, bt, ct, A_out):
    # Get bounds on each state from A_inputs, b_inputs
    vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
    x_max = []
    x_min = []
    for state in range(num_states):
        x_max.append(np.max([v[state] for v in vertices]))
        x_min.append(np.min([v[state] for v in vertices]))
    
    num_facets = A_out.shape[0]
    bs = np.zeros((num_facets))
    for i in range(num_facets):
        xt1_max, _, xt1_min, _ = torch_model_cl.full_backward_range(norm=np.inf,
                                    x_U=torch.Tensor([x_max]),
                                    x_L=torch.Tensor([x_min]),
                                    upper=True, lower=True, C=torch.Tensor([[[1]]]),
                                    A_out=torch.Tensor([A_out[i,:]]),
                                    A_in=A_inputs, b_in=b_inputs)
        bs[i] = xt1_max
    return bs

def reachLP_n(n, keras_model, A_inputs, b_inputs, At, bt, ct, A_out):
    torch_model = keras2torch(keras_model, "torch_model")
    torch_model_cl = BoundClosedLoopController.convert(torch_model, {"same-slope": True},
        A_dyn=torch.Tensor([At]), b_dyn=torch.Tensor([bt]), c_dyn=[ct])
    
    all_bs = []
    bs = reachLP_1(torch_model_cl, A_inputs, b_inputs, At, bt, ct, A_out)
    all_bs.append(bs)
    for i in range(1,n):
        bs = reachLP_1(torch_model_cl, A_out, bs, At, bt, ct, A_out)
        all_bs.append(bs)
    return all_bs



