from .Dynamics import Dynamics
import numpy as np
from nn_closed_loop.utils.mpc import control_mpc

import scipy.io as sio
import scipy
import os


class ISS(Dynamics):
    def __init__(self):

        self.continuous_time = True

        mat_fname = "{}/../../datasets/iss/iss.mat".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        mat_contents = sio.loadmat(mat_fname)

        At = scipy.sparse.csr_matrix.toarray(mat_contents['A'])
        bt = scipy.sparse.csr_matrix.toarray(mat_contents['B'])
        n = 270
        m = 3
        ct = np.zeros(n).T

        # At = np.array([[1, 1], [0, 1]])
        # bt = np.array([[0.5], [1]])
        # ct = np.array([0.0, 0.0]).T

        u_limits = None
        # u_limits = 5*np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])  # (u0_min, u0_max)

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits)

        self.cmap_name = "tab10"
        self.name = "iss"
        self.n = n
        self.m = m
        self.dt = 0.03

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(self.n)
        if not hasattr(self, "R"):
            self.R = 0.01*np.eye(self.m)
        if not hasattr(self, "Pinf"):
            # self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)
            # self.Pinf = np.zeros((self.n, self.n))
            self.Pinf = 10000*np.eye(self.n)

        # self.u_limits[:, 0],
        # self.u_limits[:, 1],

        return control_mpc(
            x0,
            self.At,
            self.bt,
            self.ct,
            self.dt,
            self.Q,
            self.R,
            self.Pinf,
            None,
            None,
            n_mpc=3,
            debug=False,
        )

    def dynamics_step(self, xs, us):
        return xs + self.dt * self.dynamics(xs, us)

    def dynamics(self, xs, us):
        xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        # xdot[:, -1] = xdot[:, -1] - np.power(xs[:, 0], 3)
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xdot += noise
        return xdot