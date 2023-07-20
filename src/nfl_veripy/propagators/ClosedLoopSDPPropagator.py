import cvxpy as cp
import nfl_veripy.constraints as constraints
import numpy as np
import torch
from nfl_veripy.utils.reach_sdp import (
    getE_in,
    getE_mid,
    getE_out,
    getInputConstraints,
    getNNConstraints,
    getOutputConstraints,
)
from tqdm import tqdm

from .ClosedLoopPropagator import ClosedLoopPropagator


class ClosedLoopSDPPropagator(ClosedLoopPropagator):
    def __init__(
        self, input_shape=None, dynamics=None, cvxpy_solver="default"
    ):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.cvxpy_solver = cvxpy_solver

    def torch2network(self, torch_model):
        return torch_model

    def get_one_step_reachable_set(self, initial_set):
        A_inputs, b_inputs = initial_set.get_polytope()
        reachable_set = constraints.create_empty_constraint(
            self.boundary_type, num_facets=self.num_polytope_facets
        )
        if isinstance(reachable_set, constraints.PolytopeConstraint):
            A_out = reachable_set.A
        elif isinstance(reachable_set, constraints.LpConstraint):
            A_out = np.vstack(
                [
                    np.eye(self.dynamics.num_states),
                    -np.eye(self.dynamics.num_states),
                ]
            )

        # Count number of units in each layer, except last layer
        # For keras:
        # num_neurons = np.sum([layer.get_config()['units'] for layer in
        #  self.network.layers][:-1])
        # For torch:
        num_neurons = np.sum(
            [
                layer.out_features
                for layer in self.network
                if isinstance(layer, torch.nn.Linear)
            ][:-1]
        )

        # Number of vertices in input polyhedron
        num_states = self.dynamics.At.shape[0]
        num_inputs = self.dynamics.bt.shape[1]

        # Get change of basis matrices
        E_in = getE_in(num_states, num_neurons, num_inputs)
        E_mid = getE_mid(
            num_states,
            num_neurons,
            num_inputs,
            self.network,
            self.dynamics.u_limits,
        )
        E_out = getE_out(
            num_states,
            num_neurons,
            num_inputs,
            self.dynamics.At,
            self.dynamics.bt,
            self.dynamics.ct,
        )

        # Get P,Q,S and constraint lists
        P, input_set_constrs = getInputConstraints(
            num_states, A_inputs.shape[0], A_inputs, b_inputs
        )

        Q, nn_constrs = getNNConstraints(num_neurons, num_inputs)

        # M_in describes the input set in NN coords
        M_in = cp.quad_form(E_in, P)
        M_mid = cp.quad_form(E_mid, Q)

        num_facets = A_out.shape[0]
        bs = np.zeros((num_facets))
        for i in tqdm(range(num_facets)):
            S_i, reachable_set_constrs, b_i = getOutputConstraints(
                num_states, A_out[i, :]
            )

            M_out = cp.quad_form(E_out, S_i)

            constrs = input_set_constrs + nn_constrs + reachable_set_constrs

            constrs.append(M_in + M_mid + M_out << 0)

            objective = cp.Minimize(b_i)

            prob = cp.Problem(objective, constrs)

            solver_args = {"verbose": False}
            if self.cvxpy_solver == "MOSEK":
                solver_args["solver"] = cp.MOSEK
            else:
                solver_args["solver"] = cp.SCS
            prob.solve(**solver_args)
            bs[i] = b_i.value

        if isinstance(reachable_set, constraints.PolytopeConstraint):
            reachable_set.b = bs
        elif isinstance(reachable_set, constraints.LpConstraint):
            reachable_set.range = np.empty((num_states, 2))
            reachable_set.range[:, 0] = -bs[(num_facets // 2) :]
            reachable_set.range[:, 1] = bs[: (num_facets // 2)]

        return reachable_set, {}
