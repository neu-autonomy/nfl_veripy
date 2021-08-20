# from Dynamics import Dynamics
from .Dynamics import Dynamics
import numpy as np


class Duffing(Dynamics):
    def __init__(self):

        self.continuous_time = True

        # g = 9.8  # m/s^2
        dim = 2
        zeta = 0.3
        dt = 0.03
        # dt = 0.1
        At = np.array([[0, 1], [-1, -2*zeta]], dtype=float)

        bt = np.zeros((dim, 1), dtype=float)
        bt[1][0] = 1.0

        ct = np.zeros((dim,), dtype=float)
        # ct[-1] = -g
        # ct = np.array([0., 0., 0. ,0., 0., -g]).T

        u_limits = None
        # u_limits = np.array(
        #     [
        #         [-1, 1],
        #     ]
        # )
        # u_limits = np.array(
        #     [
        #         [-np.pi / 9, np.pi / 9],
        #         [-np.pi / 9, np.pi / 9],
        #         [0, 2 * g],
        #     ]
        # )


        self.dim = dim
        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt)

        self.cmap_name = "tab20"
        self.name = "duffing"

        # # LQR-MPC parameters
        # self.Q = np.eye(2)
        # self.R = 1
        # self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def dynamics_step(self, xs, us):
        return xs + self.dt * self.dynamics(xs, us)

    def dynamics(self, xs, us):
        xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        xdot[:,-1] = xdot[:,-1] - np.power(xs[:,0], 3)
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xdot += noise
        return xdot
