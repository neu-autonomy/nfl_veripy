import cvxpy as cp
import numpy as np
import pypoman
import torch

import nfl_veripy.analyzers.Analyzer as Analyzer
import nfl_veripy.constraints as constraints

from .ClosedLoopPropagator import ClosedLoopPropagator


class ClosedLoopSeparablePropagator(ClosedLoopPropagator):
    """
    Instead of relaxing NN then folding the relaxed NN into the optimization
    over next states, treat the dynamics and NN separately.
    ==> A generalization of the approach in [Xiang21TNNLS]
    First, find bounds on the set of controls that could be applied. Then, find
    bounds on the set of next states that could be achieved for any of those.
    """

    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)

        self.partitioner_hyperparams = {
            "num_simulations": 1,
            "type": "None",
            "termination_condition_type": None,
            "termination_condition_value": None,
            "interior_condition": "linf",
            "make_animation": False,
            "show_animation": False,
        }
        self.propagator_hyperparams = {
            "input_shape": self.input_shape,
        }

    def torch2network(self, torch_model):
        self.nn_analyzer = Analyzer(torch_model)
        self.nn_analyzer.partitioner = self.partitioner_hyperparams
        self.nn_analyzer.propagator = self.propagator_hyperparams
        return torch_model

    def get_min_max_controls(self, nn_input_max, nn_input_min, C):
        input_range = np.stack([nn_input_min[0], nn_input_max[0]]).T
        nn_output_range, analyzer_info = self.nn_analyzer.get_output_range(
            input_range
        )
        u_min = nn_output_range[:, 0]
        u_max = nn_output_range[:, 1]
        return u_min, u_max

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        if isinstance(input_constraint, constraints.PolytopeConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # Get bounds on each state from A_inputs, b_inputs
            try:
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(A_inputs, b_inputs)
                )
            except Exception:
                # Sometimes get arithmetic error... this may fix it
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(
                        A_inputs, b_inputs + 1e-6
                    )
                )
            x_max = np.max(vertices, 0)
            x_min = np.min(vertices, 0)
        elif isinstance(input_constraint, constraints.LpConstraint):
            x_min = input_constraint.range[..., 0]
            x_max = input_constraint.range[..., 1]
            A_inputs = None
            b_inputs = None
        else:
            raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeConstraint):
            A_out = output_constraint.A
            num_facets = A_out.shape[0]
            bs = np.zeros((num_facets))
        elif isinstance(output_constraint, constraints.LpConstraint):
            A_out = np.eye(x_min.shape[0])
            num_facets = A_out.shape[0]
            ranges = np.zeros((num_facets, 2))
        else:
            raise NotImplementedError

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        prev_state_max = torch.Tensor([x_max])
        prev_state_min = torch.Tensor([x_min])
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor([self.dynamics.sensor_noise[:, 1]])
            nn_input_min += torch.Tensor([self.dynamics.sensor_noise[:, 0]])

        # Compute the NN output matrices (for the input constraints)
        num_control_inputs = self.dynamics.bt.shape[1]
        C = torch.eye(num_control_inputs).unsqueeze(0)

        u_min, u_max = self.get_min_max_controls(nn_input_max, nn_input_min, C)

        # Sample a grid of pts from the input set, to get exact NN output
        # polytope
        # x0 = np.linspace(x_min[0], x_max[0], num=10)
        # x1 = np.linspace(x_min[1], x_max[1], num=10)
        # xx, yy = np.meshgrid(x0, x1)
        # pts = np.reshape(np.dstack([xx, yy]), (-1, 2))
        # sampled_outputs = self.network.forward(torch.Tensor(pts))

        # # Print and compare the two bounds numerically
        # sampled_output_min = np.min(sampled_outputs.data.numpy())
        # sampled_output_max = np.max(sampled_outputs.data.numpy())
        # print("(u_min, u_max): ({.4f}, {.4f})".format(u_min, u_max))
        # print("(sampled_min, sampled_max): ({.4f}, {.4f})".format(
        # sampled_output_min, sampled_output_max))

        for i in range(num_facets):
            # For each dimension of the output constraint (facet/lp-dimension):
            # compute a bound of the NN output using the pre-computed matrices

            (
                A_out_xt1_max,
                A_out_xt1_min,
            ) = self.compute_bound_cf(
                u_min,
                u_max,
                x_max,
                x_min,
                A_out[i, :],
                A_in=A_inputs,
                b_in=b_inputs,
            )

            if isinstance(output_constraint, constraints.PolytopeConstraint):
                bs[i] = A_out_xt1_max
            elif isinstance(output_constraint, constraints.LpConstraint):
                ranges[i, 0] = A_out_xt1_min
                ranges[i, 1] = A_out_xt1_max
            else:
                raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, constraints.LpConstraint):
            output_constraint.range = ranges
        else:
            raise NotImplementedError
        return output_constraint, {}

    def get_one_step_backprojection_set(
        self, output_constraint, input_constraint, num_partitions=None
    ):
        raise NotImplementedError

    def compute_bound(
        self, u_min, u_max, xt_max, xt_min, A_out, A_in=None, b_in=None
    ):
        if A_in is None or b_in is None:
            # Resort to cvxpy solver because polytope constraints
            (
                A_out_xt1_max,
                A_out_xt1_min,
            ) = self.compute_bound_lp(
                u_min,
                u_max,
                xt_max,
                xt_min,
                A_out,
                A_in=A_in,
                b_in=b_in,
            )
        else:
            # Can solve in closed-form
            (
                A_out_xt1_max,
                A_out_xt1_min,
            ) = self.compute_bound_cf(
                u_min,
                u_max,
                xt_max,
                xt_min,
                A_out,
            )

        return A_out_xt1_max, A_out_xt1_min

    def compute_bound_lp(
        self, u_min, u_max, xt_max, xt_min, A_out, A_in=None, b_in=None
    ):
        num_states = 2
        num_control_inputs = 1

        xt = cp.Variable(num_states)
        u = cp.Variable(num_control_inputs)
        cost = A_out.T @ (
            self.dynamics.At @ xt + self.dynamics.bt @ u + self.dynamics.ct
        )

        # Input set constraints
        constraints = []
        if A_in is not None and b_in is not None:
            constraints += [A_in @ xt <= b_in]
        else:
            constraints += [
                xt <= xt_max,
                xt >= xt_min,
            ]

        constraints += [
            u <= u_max,
            u >= u_min,
        ]

        objective = cp.Maximize(cost)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        ub = prob.value

        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        lb = prob.value

        return ub, lb

    def compute_bound_cf(
        self, u_min, u_max, xt_max, xt_min, A_out, A_in=None, b_in=None
    ):
        # max Aout(At*xt + bt*u + ct) s.t. xt\inXt, u\inU
        # = max Aout([At bt ct]@[xt; u; 1]) s.t. xt\inXt, u\inU
        # = max D@z s.t. z\inZ, where D = Aout([At; bt; ct], z = [xt; u; 1]
        # = (max (z_eps*D)@a s.t. a\inA) + D@z_ctr, where a = z - z_ctr / z_eps

        D = A_out @ np.hstack(
            [
                self.dynamics.At,
                self.dynamics.bt,
                np.expand_dims(self.dynamics.ct, axis=1),
            ]
        )
        z0 = np.hstack([(xt_max + xt_min) / 2.0, (u_max + u_min) / 2.0, 1])
        eps = np.hstack([(xt_max - xt_min) / 2.0, (u_max - u_min) / 2.0, 0])

        ub = np.sum(np.abs(np.multiply(eps, D))) + D @ z0
        lb = -np.sum(np.abs(np.multiply(eps, D))) + D @ z0

        return ub, lb


class ClosedLoopSeparableCROWNPropagator(ClosedLoopSeparablePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.propagator_hyperparams["type"] = "CROWN_LIRPA"


class ClosedLoopSeparableIBPPropagator(ClosedLoopSeparablePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.propagator_hyperparams["type"] = "IBP_LIRPA"


class ClosedLoopSeparableSGIBPPropagator(ClosedLoopSeparablePropagator):
    """
    This is roughly the approach from [Xiang21TNNLS],
    replacing reachODE with a closed-form/LP soln valid for DT plants
    """

    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.partitioner_hyperparams["type"] = "SimGuided"
        self.partitioner_hyperparams["num_simulations"] = 1e5
        self.partitioner_hyperparams["termination_condition_type"] = (
            "input_cell_size"
        )
        self.partitioner_hyperparams["termination_condition_value"] = 0.1
        # self.partitioner_hyperparams["termination_condition_type"] = (
        #     "time_budget"
        # )
        # self.partitioner_hyperparams["termination_condition_value"] = 2.0
        self.propagator_hyperparams["type"] = "IBP_LIRPA"
