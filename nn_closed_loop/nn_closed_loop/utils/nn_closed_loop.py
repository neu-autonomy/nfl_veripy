from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def bound_closed_loop_dynamics(controller, dynamics, num_steps=1):
  model = ClosedLoopDynamics(controller, dynamics, num_steps=num_steps)
  bounded_model = BoundedModule(model, nominal_input)


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