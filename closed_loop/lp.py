import numpy as np
np.set_printoptions(precision=2)
from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
from reach_lp.nn_bounds import BoundClosedLoopController
from reach_lp.nn import load_model
import torch
import time
import pypoman

def min_and_max_controls(pts, A_inputs, b_inputs, At, bt, ct, A_out, keras_model=None):
    if keras_model is None:
        keras_model = load_model()
    torch_model = keras2torch(keras_model, "torch_model")
    # crown_params = {"zero-lb": True}
    # crown_params = {"one-lb": True}
    crown_params = {"same-slope": True}
    torch_model_cl = BoundClosedLoopController.convert(torch_model, crown_params,
        A_dyn=torch.Tensor([At]), b_dyn=torch.Tensor([bt]), c_dyn=[ct])

    # Get bounds on each state from A_inputs, b_inputs
    num_states = At.shape[0]
    vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
    x_max = []
    x_min = []
    for state in range(num_states):
        x_max.append(np.max([v[state] for v in vertices]))
        x_min.append(np.min([v[state] for v in vertices]))

    lower_A, upper_A, lower_sum_b, upper_sum_b = torch_model_cl.full_backward_range(norm=np.inf,
                                    x_U=torch.Tensor([x_max]),
                                    x_L=torch.Tensor([x_min]),
                                    upper=True, lower=True, C=torch.Tensor([[[1]]]),
                                    A_out=torch.Tensor([1,0]),
                                    A_in=A_inputs, b_in=b_inputs,
                                    closed_loop=False)

    omega = lower_A.data.numpy()
    lamb = upper_A.data.numpy()
    nl = lower_sum_b.data.numpy().squeeze()
    ul = upper_sum_b.data.numpy().squeeze()
    pi_l = np.dot(omega, pts.T).squeeze()+nl
    pi_u = np.dot(lamb, pts.T).squeeze()+ul
    return np.expand_dims(pi_l, axis=-1), np.expand_dims(pi_u, axis=-1)
    # return pi_l, pi_u
    # return lower_A, upper_A, lower_sum_b, upper_sum_b


### This method is deprecated, lives in ClosedLoopAnalyzer now.
def reachLP_1(torch_model_cl, A_inputs, b_inputs, At, bt, ct, A_out, u_limits=None):
    # Get bounds on each state from A_inputs, b_inputs
    num_states = At.shape[0]
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
                                    A_in=A_inputs, b_in=b_inputs,
                                    u_limits=u_limits)
        bs[i] = xt1_max
    return bs

### This method is deprecated, lives in ClosedLoopAnalyzer now.
def reachLP_n(n, keras_model, A_inputs, b_inputs, At, bt, ct, A_out, u_limits=None):
    torch_model = keras2torch(keras_model, "torch_model")
    # crown_params = {"zero-lb": True}
    # crown_params = {"one-lb": True}
    crown_params = {"same-slope": True}
    torch_model_cl = BoundClosedLoopController.convert(torch_model, crown_params,
        A_dyn=torch.Tensor([At]), b_dyn=torch.Tensor([bt]), c_dyn=[ct])
    
    all_bs = []
    bs = reachLP_1(torch_model_cl, A_inputs, b_inputs, At, bt, ct, A_out, u_limits=u_limits)
    all_bs.append(bs)
    for i in range(1,n):
        bs = reachLP_1(torch_model_cl, A_out, bs, At, bt, ct, A_out, u_limits=u_limits)
        all_bs.append(bs)
    return all_bs



