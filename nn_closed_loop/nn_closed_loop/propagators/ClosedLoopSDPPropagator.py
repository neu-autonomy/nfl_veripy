from .ClosedLoopPropagator import ClosedLoopPropagator
import numpy as np
import nn_closed_loop.constraints as constraints
from tqdm import tqdm
import cvxpy as cp
from nn_closed_loop.utils.reach_sdp import getE_in, getE_mid, getE_out, getInputConstraints, getOutputConstraints, getNNConstraints, getInputConstraintsEllipsoid, getOutputConstraintsEllipsoid
from nn_closed_loop.utils.utils import init_state_range_to_polytope
import torch

class ClosedLoopSDPPropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(self, input_shape=input_shape, dynamics=dynamics)

    def torch2network(self, torch_model):
        return torch_model

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            A_out = output_constraint.A
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            A_out = np.vstack([np.eye(self.dynamics.num_states), -np.eye(self.dynamics.num_states)])
        elif isinstance(output_constraint, constraints.EllipsoidOutputConstraint):
            A_out = np.empty((1,1)) # dummy so that num_facets is right...
        else:
            raise NotImplementedError

        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
        elif isinstance(input_constraint, constraints.LpInputConstraint):
            if input_constraint.p != np.inf:
                raise NotImplementedError
            else:
                A_inputs, b_inputs = init_state_range_to_polytope(input_constraint.range)
        elif isinstance(input_constraint, constraints.EllipsoidInputConstraint):
            A = input_constraint.shape
            b = input_constraint.center
        else:
            raise NotImplementedError

        # Count number of units in each layer, except last layer
        # For keras:
        # num_neurons = np.sum([layer.get_config()['units'] for layer in self.network.layers][:-1])
        # For torch:
        num_neurons = np.sum([layer.out_features for layer in self.network if isinstance(layer, torch.nn.Linear)][:-1])

        # Number of vertices in input polyhedron
        num_states = self.dynamics.At.shape[0]
        num_inputs = self.dynamics.bt.shape[1]

        # Get change of basis matrices
        E_in = getE_in(num_states, num_neurons, num_inputs)
        E_mid = getE_mid(num_states, num_neurons, num_inputs, self.network, self.dynamics.u_limits)
        E_out = getE_out(num_states, num_neurons, num_inputs, self.dynamics.At, self.dynamics.bt, self.dynamics.ct)

        # Get P,Q,S and constraint lists
        if isinstance(input_constraint, constraints.PolytopeInputConstraint) or isinstance(input_constraint, constraints.LpInputConstraint):
            P, input_set_constrs = getInputConstraints(num_states, A_inputs.shape[0], A_inputs, b_inputs)
        elif isinstance(input_constraint, constraints.EllipsoidInputConstraint):
            P, input_set_constrs = getInputConstraintsEllipsoid(num_states, A, b)
        
        Q, nn_constrs = getNNConstraints(num_neurons, num_inputs)

        # M_in describes the input set in NN coords
        M_in = cp.quad_form(E_in, P)
        M_mid = cp.quad_form(E_mid, Q)

        num_facets = A_out.shape[0]
        bs = np.zeros((num_facets))
        for i in tqdm(range(num_facets)):

            if isinstance(input_constraint, constraints.PolytopeInputConstraint) or isinstance(input_constraint, constraints.LpInputConstraint):
                S_i, reachable_set_constrs, b_i = getOutputConstraints(num_states, A_out[i,:])
            elif isinstance(input_constraint, constraints.EllipsoidInputConstraint):
                S_i, reachable_set_constrs, A, b = getOutputConstraintsEllipsoid(num_states)

            M_out = cp.quad_form(E_out, S_i)

            constrs = input_set_constrs + nn_constrs + reachable_set_constrs

            constrs.append(M_in + M_mid + M_out << 0)

            if isinstance(input_constraint, constraints.PolytopeInputConstraint) or isinstance(input_constraint, constraints.LpInputConstraint):
                objective = cp.Minimize(b_i)
            elif isinstance(input_constraint, constraints.EllipsoidInputConstraint):
                objective = cp.Minimize(-cp.log_det(A))

            prob = cp.Problem(objective, constrs)
            prob.solve(verbose=False, solver=cp.MOSEK)
            # print("status:", prob.status)
            bs[i] = b_i.value

        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            output_constraint.range = np.empty((num_states,2))
            output_constraint.range[:,0] = -bs[num_facets//2:]
            output_constraint.range[:,1] = bs[:num_facets//2]
        elif isinstance(output_constraint, constraints.EllipsoidOutputConstraint):
            output_constraint.center = b
            output_constraint.shape = A
        else:
            raise NotImplementedError

        # print("status:", prob.status)
        # print("optimal value", prob.value)
        # print("b_{i}: {val}".format(i=i, val=b_i.value))
        # print("S_{i}: {val}".format(i=i, val=S_i.value))

        return output_constraint, {}
