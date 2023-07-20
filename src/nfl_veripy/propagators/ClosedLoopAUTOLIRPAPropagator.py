import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.utils import create_cl_model

from .ClosedLoopPropagator import ClosedLoopPropagator

dir_path = os.path.dirname(os.path.realpath(__file__))


class ClosedLoopAUTOLIRPAPropagator(ClosedLoopPropagator):
    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)

    def torch2network(
        self, torch_model: torch.nn.Sequential
    ) -> torch.nn.Sequential:
        return torch_model

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint, t_max: int
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        reachable_sets_list = []
        info = {"per_timestep": []}  # type: dict[str, Any]
        reachable_set, this_info = self.get_N_step_reachable_set(
            initial_set, 0
        )
        reachable_sets_list.append(deepcopy(reachable_set))
        info["per_timestep"].append(this_info)
        step_num = 1
        for i in np.arange(
            0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
        ):
            reachable_set, this_info = self.get_N_step_reachable_set(
                reachable_set, step_num
            )
            reachable_sets_list.append(deepcopy(reachable_set))
            info["per_timestep"].append(this_info)
            step_num += 1

        reachable_sets = constraints.list_to_constraint(reachable_sets_list)

        return reachable_sets, info

    def get_N_step_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        num_steps: int,
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        initial_set_range = initial_set.to_range()

        nominal_input = (
            torch.Tensor([initial_set_range[:, 1]])
            + torch.Tensor([initial_set_range[:, 0]])
        ) / 2.0
        eps = (
            torch.Tensor([initial_set_range[:, 1]])
            - torch.Tensor([initial_set_range[:, 0]])
        ) / 2.0

        model = create_cl_model(self.dynamics, num_steps + 1)

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        my_input = BoundedTensor(nominal_input, ptb)

        model = BoundedModule(model, nominal_input)

        lb, ub = model.compute_bounds(x=(my_input,), method="backward")

        reachable_set = constraints.LpConstraint(
            range=np.vstack((lb.detach().numpy(), ub.detach().numpy())).T
        )
        return reachable_set, {}
