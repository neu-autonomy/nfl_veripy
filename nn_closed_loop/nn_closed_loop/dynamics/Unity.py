from .Dynamics import DiscreteTimeDynamics
import numpy as np
from scipy.linalg import solve_discrete_are
from nn_closed_loop.utils.mpc import control_mpc


class Unity(DiscreteTimeDynamics):
    def __init__(self, nx=2, nu=2):

        self.continuous_time = False

        self.nx = nx
        self.nu = nu

        At = np.eye(nx)
        for i in range(nx):
            At[i, i+1:] = 0.1
        bt = np.zeros((nx, nu))
        bt[-1, -1] = 1.
        ct = np.zeros((nx,))

        # u_limits = None
        u_limits = np.vstack([-np.ones(nu), np.ones(nu)]).T
        # u_limits = np.array([[-1.0, 1.0]])  # (u0_min, u0_max)

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits)

        self.cmap_name = "tab10"

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(self.nx)
        if not hasattr(self, "R"):
            self.R = np.eye(self.nu)
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
            n_mpc=5,
            debug=False,
        )
