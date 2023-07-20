import numpy as np
from nfl_veripy.utils.mpc import control_mpc
from scipy.linalg import solve_discrete_are

from .Dynamics import DiscreteTimeDynamics


class Taxinet(DiscreteTimeDynamics):
    def __init__(self):
        self.continuous_time = False

        # dt = 0.125
        dt = 1

        v = 5
        ll = 5

        At = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, np.pi / 180 * v * dt],
                [0, 0, 0, 1],
            ]
        )
        bt = (
            np.pi
            / 180
            * np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [-v * 0.74 / ll * dt, -v * 0.44 / ll * dt],
                ]
            )
        )

        ct = np.array([0.0, 0.0, 0.0, 0.0]).T

        # u_limits = None
        u_limits = np.array([[-12, 12], [-30, 30]])  # (u0_min, u0_max)
        x_limits = {0: [-0.8, 0.8], 1: [-0.8, 0.8]}

        super().__init__(
            At=At, bt=bt, ct=ct, dt=dt, u_limits=u_limits, x_limits=x_limits
        )

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
