from .Dynamics import Dynamics
import numpy as np


class Quadrotor(Dynamics):
    def __init__(self):

        self.continuous_time = True

        g = 9.8  # m/s^2

        At = np.zeros((6, 6))
        At[0][3] = 1
        At[1][4] = 1
        At[2][5] = 1

        bt = np.zeros((6, 3))
        bt[3][0] = g
        bt[4][1] = -g
        bt[5][2] = 1

        ct = np.zeros((6,))
        ct[-1] = -g
        # ct = np.array([0., 0., 0. ,0., 0., -g]).T

        # u_limits = None
        u_limits = np.array(
            [
                [-np.pi / 9, np.pi / 9],
                [-np.pi / 9, np.pi / 9],
                [0, 2 * g],
            ]
        )

        dt = 0.1

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt)

        self.cmap_name = "tab20"

        # # LQR-MPC parameters
        # self.Q = np.eye(2)
        # self.R = 1
        # self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def dynamics_step(self, xs, us):
        return xs + self.dt * self.dynamics(xs, us)

    def dynamics(self, xs, us):
        xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xdot += noise
        return xdot
