import numpy as np
from partition.Propagator import Propagator
import torch
import pypoman
from closed_loop.reach_sdp import getE_in, getE_mid, getE_out, getInputConstraints, getOutputConstraints, getNNConstraints
import cvxpy as cp
from tqdm import tqdm
from closed_loop.ClosedLoopConstraints import PolytopeInputConstraint, LpInputConstraint, PolytopeOutputConstraint, LpOutputConstraint
from copy import deepcopy


class ClosedLoopPropagator(Propagator):
    def __init__(self, input_shape=None, dynamics=None):
        Propagator.__init__(self, input_shape=input_shape)
        self.dynamics = dynamics

    def get_one_step_reachable_set(self, input_constraint, output_constraint, u_limits=[-5., 5.]):
        raise NotImplementedError

    def get_reachable_set(self, input_constraint, output_constraint, t_max, u_limits=None):
        output_constraints = []
        output_constraint, _ = self.get_one_step_reachable_set(input_constraint, output_constraint, u_limits=u_limits)
        output_constraints.append(deepcopy(output_constraint))
        for i in range(1, t_max):
            next_input_constraint = output_constraint.to_input_constraint()
            next_output_constraint = deepcopy(output_constraint)
            output_constraint, _ = self.get_one_step_reachable_set(next_input_constraint, next_output_constraint, u_limits=u_limits)
            output_constraints.append(deepcopy(output_constraint))
        return output_constraints, {}

class ClosedLoopSDPPropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(self, input_shape=input_shape, dynamics=dynamics)

    def torch2network(self, torch_model):
        return torch_model

    def get_one_step_reachable_set(self, input_constraint, output_constraint, u_limits=[-5., 5.]):
        if isinstance(output_constraint, PolytopeOutputConstraint):
            A_out = output_constraint.A
        else:
            raise NotImplementedError
        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
        else:
            raise NotImplementedError

        # Count number of units in each layer, except last layer
        # For keras:
        # num_neurons = np.sum([layer.get_config()['units'] for layer in self.network.layers][:-1])
        # For torch:
        num_neurons = np.sum([layer.out_features for layer in self.network if isinstance(layer, torch.nn.Linear)][:-1])

        # Number of vertices in input polyhedron
        m = A_inputs.shape[0]
        num_states = self.dynamics.At.shape[0]
        num_inputs = self.dynamics.bt.shape[1]

        u_min, u_max = u_limits

        # Get change of basis matrices
        E_in = getE_in(num_states, num_neurons, num_inputs)
        E_mid = getE_mid(num_states, num_neurons, num_inputs, self.network, u_min, u_max)
        E_out = getE_out(num_states, num_neurons, num_inputs, self.dynamics.At, self.dynamics.bt, self.dynamics.ct)

        # Get P,Q,S and constraint lists
        P, input_set_constrs = getInputConstraints(num_states, m, A_inputs, b_inputs)
        Q, nn_constrs = getNNConstraints(num_neurons, num_inputs)

        # M_in describes the input set in NN coords
        M_in = cp.quad_form(E_in, P)
        M_mid = cp.quad_form(E_mid, Q)

        num_facets = A_inputs.shape[0]
        bs = np.zeros((num_facets))
        for i in tqdm(range(num_facets)):
            S_i, reachable_set_constrs, b_i = getOutputConstraints(num_states, A_inputs[i,:])
            M_out = cp.quad_form(E_out, S_i)

            constraints = input_set_constrs + nn_constrs + reachable_set_constrs

            constraints.append(M_in + M_mid + M_out << 0)

            objective = cp.Minimize(b_i)
            prob = cp.Problem(objective,
                              constraints)
            prob.solve()
            # print("status:", prob.status)
            bs[i] = b_i.value

        if isinstance(output_constraint, PolytopeOutputConstraint):
            output_constraint.b = bs
        else:
            raise NotImplementedError

        # print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("b_{i}: {val}".format(i=i, val=b_i.value))
        # print("S_{i}: {val}".format(i=i, val=S_i.value))

        return output_constraint, {}

class ClosedLoopCROWNIBPCodebasePropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(self, input_shape=input_shape, dynamics=dynamics)

    def torch2network(self, torch_model):
        from closed_loop.nn_bounds import BoundClosedLoopController
        torch_model_cl = BoundClosedLoopController.convert(torch_model, self.params,
            A_dyn=torch.Tensor([self.dynamics.At]), b_dyn=torch.Tensor([self.dynamics.bt]), c_dyn=[self.dynamics.ct])
        return torch_model_cl

    def forward_pass(self, input_data):
        return self.network(torch.Tensor(input_data), method_opt=None).data.numpy()

    def get_one_step_reachable_set(self, input_constraint, output_constraint, u_limits=None):
        if isinstance(output_constraint, PolytopeOutputConstraint):
            A_out = output_constraint.A
            num_facets = A_out.shape[0]
            bs = np.zeros((num_facets))
        elif isinstance(output_constraint, LpOutputConstraint):
            A_out = np.eye(2)
            num_facets = A_out.shape[0]
            ranges = np.zeros((num_facets,2))
        else:
            raise NotImplementedError

        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # Get bounds on each state from A_inputs, b_inputs
            num_states = self.dynamics.At.shape[0]
            vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
            x_max = []
            x_min = []
            for state in range(num_states):
                x_max.append(np.max([v[state] for v in vertices]))
                x_min.append(np.min([v[state] for v in vertices]))
            norm = np.inf
        elif isinstance(input_constraint, LpInputConstraint):
            x_min = input_constraint.range[...,0]
            x_max = input_constraint.range[...,1]
            norm = input_constraint.p
            A_inputs = None
            b_inputs = None
        else:
            raise NotImplementedError

        # Compute the NN output matrices (for the input constraints)
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(method_opt=self.method_opt,
                                    norm=norm,
                                    x_U=torch.Tensor([x_max]),
                                    x_L=torch.Tensor([x_min]),
                                    upper=True, lower=True, C=torch.Tensor([[[1]]]),
                                    A_out=None,
                                    A_in=A_inputs, b_in=b_inputs,
                                    u_limits=u_limits,
                                    return_matrices=True)

        for i in range(num_facets):
            # For each dimension of the output constraint (facet/lp-dimension):
            #  compute a bound of the NN output using the pre-computed matrices
            if A_out is None:
                A_out_torch = None
            else:
                A_out_torch = torch.Tensor([A_out[i,:]])
            
            xt1_max, xt1_min = self.network.compute_bound_from_matrices(lower_A, lower_sum_b, upper_A, upper_sum_b, torch.Tensor([x_max]), torch.Tensor([x_min]), norm, A_out=A_out_torch, A_dyn=torch.Tensor([self.dynamics.At]), b_dyn=torch.Tensor([self.dynamics.bt]))

            if isinstance(output_constraint, PolytopeOutputConstraint):
                bs[i] = xt1_max
            elif isinstance(output_constraint, LpOutputConstraint):
                ranges[i,0] = xt1_min[0][0]
                ranges[i,1] = xt1_max[0][0]
            else:
                raise NotImplementedError

        if isinstance(output_constraint, PolytopeOutputConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, LpOutputConstraint):
            output_constraint.range = ranges
        else:
            raise NotImplementedError

        return output_constraint, {}

class ClosedLoopIBPPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape, dynamics=dynamics)
        raise NotImplementedError
        # TODO: Write nn_bounds.py:BoundClosedLoopController:interval_range
        # (using bound_layers.py:BoundSequential:interval_range)
        self.method_opt = "interval_range"
        self.params = {}

class ClosedLoopCROWNPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape, dynamics=dynamics)
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": False}

class ClosedLoopFastLinPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape, dynamics=dynamics)
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": True}

