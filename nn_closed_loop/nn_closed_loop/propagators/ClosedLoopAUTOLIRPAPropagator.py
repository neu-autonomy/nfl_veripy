from bisect import bisect
from hashlib import new
import multiprocessing
from posixpath import split
from tkinter.messagebox import NO
from tokenize import Hexnumber

import matplotlib
from .ClosedLoopPropagator import ClosedLoopPropagator
import nn_closed_loop.elements as elements
import numpy as np
import pypoman
import nn_closed_loop.constraints as constraints
import torch
from nn_closed_loop.utils.utils import range_to_polytope
import cvxpy as cp
from itertools import product
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

import time
import os

from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor

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
        info = {'per_timestep': []}
        output_constraint, this_info = self.get_one_step_reachable_set(
            input_constraint, output_constraint, 0
        )
        output_constraints.append(deepcopy(output_constraint))
        info['per_timestep'].append(this_info)
        for i in np.arange(0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt):
            next_input_constraint = deepcopy(output_constraint)
            next_output_constraint = deepcopy(output_constraint)
            
            output_constraint, this_info = self.get_one_step_reachable_set(
                next_input_constraint, next_output_constraint, i
            )
            output_constraints.append(deepcopy(output_constraint))
            info['per_timestep'].append(this_info)

        return output_constraints, info
    
    def get_one_step_reachable_set(self, input_constraint, output_constraint, t):
    
        if isinstance(input_constraint, constraints.LpConstraint):
            x_range = input_constraint.range
        else:
            raise NotImplementedError

        nominal_input = (torch.Tensor([x_range[:, 1]]) + torch.Tensor([x_range[:, 0]])) / 2.
        eps = (torch.Tensor([x_range[:, 1]]) - torch.Tensor([x_range[:, 0]]))/2.

        model = self.create_model(t+1)

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        my_input = BoundedTensor(nominal_input, ptb)

        import pdb; pdb.set_trace()
        model = BoundedModule(model, nominal_input)
        
        
        
        lb, ub = model.compute_bounds(x=(my_input,), method="backward")
        import pdb; pdb.set_trace()
        return super().get_one_step_reachable_set(input_constraint, output_constraint)
    

    def create_model(self, num_steps):
        path = "{}/../../models/{}/{}".format(dir_path, "Pendulum", "default/single_pendulum_small_controller.torch")
        controller = Controller()
        # controller.load_state_dict(self.network.state_dict(), strict=False)
        controller.load_state_dict(torch.load(path))
        dynamics = self.dynamics.nn_module
        model = ClosedLoopDynamics(controller, dynamics, num_steps=num_steps)
        return model


# Define computation as a nn.Module.
class ClosedLoopDynamics(nn.Module):
  def __init__(self, controller, dynamics, num_steps=1):
    super().__init__()
    self.controller = controller
    self.dynamics = dynamics
    self.num_steps = num_steps

  def forward(self, xt):

    xts = [xt]
    for i in range(self.num_steps):

      ut = self.controller(xts[-1])
      xt1 = self.dynamics(xts[-1], ut)

      xts.append(xt1)

    return xts[-1]
  

class Controller(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(2, 25)
    self.fc2 = nn.Linear(25, 25)
    self.fc3 = nn.Linear(25, 1)

  def forward(self, xt):

    # ut = F.relu(torch.matmul(xt, torch.Tensor([[1], [0]])))
    output = F.relu(self.fc1(xt))
    output = F.relu(self.fc2(output))
    output = self.fc3(output)

    return output