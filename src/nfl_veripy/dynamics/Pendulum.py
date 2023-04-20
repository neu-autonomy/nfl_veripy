import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nfl_veripy.utils.mpc import control_mpc
from scipy.linalg import solve_discrete_are

from .Dynamics import DiscreteTimeDynamics


class Pendulum(DiscreteTimeDynamics):
    def __init__(self):
        self.continuous_time = False

        # dt = 0.0625

        dt = 0.1
        self.gravity = 1
        self.length = 0.5
        self.mass = 0.5

        self.dynamics_module = PendulumDynamics()
        self.controller_module = Controller()

        At = np.zeros((2, 2))
        bt = np.zeros((2, 1))

        # u_limits = None
        u_limits = np.array([[-10.0, 10.0]])  # (u0_min, u0_max)

        super().__init__(At=At, bt=bt, ct=None, dt=dt, u_limits=u_limits)

        self.cmap_name = "tab10"

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(2)
        if not hasattr(self, "R"):
            self.R = 1
        if not hasattr(self, "Pinf"):
            self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

        return control_mpc(
            x0,
            self.At,
            self.bt,
            self.ct,
            self.Q,
            self.R,
            self.Pinf,
            self.u_limits[:, 0],
            self.u_limits[:, 1],
            n_mpc=10,
            debug=False,
        )

    def dynamics_step(self, xs, us):
        dt = self.dt
        gravity = self.gravity
        length = self.length
        mass = self.mass

        x0_t1 = xs[:, 0] + dt * xs[:, 1]
        x1_t1 = (
            xs[:, 1]
            + dt * (gravity / length) * np.sin(xs[:, 0])
            + dt * us[:, 0] / (mass * length**2)
        )

        xs_t1 = np.vstack((x0_t1, x1_t1)).T

        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xs_t1 += noise

        return xs_t1


# Define computation as a nn.Module.
class PendulumDynamics(nn.Module):
    def forward(self, xt, ut):
        # got this from pg 15 of: https://arxiv.org/pdf/2108.01220.pdf
        # updated to values from page 21
        dt = 0.1
        gravity = 1
        length = 0.5
        mass = 0.5

        xt_0 = torch.matmul(xt, torch.Tensor([[1], [0]]))
        xt_1 = torch.matmul(xt, torch.Tensor([[0], [1]]))

        xt1_0 = torch.matmul(xt, torch.Tensor([[1.0], [dt]]))
        xt1_1 = (
            xt_1
            + dt * (gravity / length) * torch.sin(xt_0)
            + dt * ut / (mass * length**2)
        )

        xt1 = torch.cat([xt1_0, xt1_1], 1)

        return xt1


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
