import os
from copy import deepcopy

import nfl_veripy.constraints as constraints
import numpy as np
import torch
from nfl_veripy.utils.utils import create_cl_model

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from .ClosedLoopPropagator import ClosedLoopPropagator

dir_path = os.path.dirname(os.path.realpath(__file__))


class ClosedLoopAUTOLIRPAPropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )

    def torch2network(self, torch_model):
        return torch_model

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        output_constraints = []
        # import pdb; pdb.set_trace()
        info = {"per_timestep": []}
        output_constraint, this_info = self.get_one_step_reachable_set(
            input_constraint, output_constraint, 0
        )
        output_constraints.append(deepcopy(output_constraint))
        info["per_timestep"].append(this_info)
        step_num = 1
        for i in np.arange(
            0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
        ):
            next_output_constraint = deepcopy(output_constraint)
            output_constraint, this_info = self.get_one_step_reachable_set(
                input_constraint, next_output_constraint, step_num
            )
            output_constraints.append(deepcopy(output_constraint))
            info["per_timestep"].append(this_info)
            step_num += 1

        return output_constraints, info

    def get_one_step_reachable_set(
        self, input_constraint, output_constraint, t
    ):
        if isinstance(input_constraint, constraints.LpConstraint):
            x_range = input_constraint.range
        else:
            raise NotImplementedError

        nominal_input = (
            torch.Tensor([x_range[:, 1]]) + torch.Tensor([x_range[:, 0]])
        ) / 2.0
        eps = (
            torch.Tensor([x_range[:, 1]]) - torch.Tensor([x_range[:, 0]])
        ) / 2.0

        model = create_cl_model(self.dynamics, t + 1)

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        my_input = BoundedTensor(nominal_input, ptb)

        model = BoundedModule(model, nominal_input)

        lb, ub = model.compute_bounds(x=(my_input,), method="backward")
        output_constraint.range = np.vstack(
            (lb.detach().numpy(), ub.detach().numpy())
        ).T
        return output_constraint, {}
