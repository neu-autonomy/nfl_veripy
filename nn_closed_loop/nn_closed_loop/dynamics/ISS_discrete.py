# from Dynamics import Dynamics
from .Dynamics import Dynamics
import numpy as np
from scipy.linalg import solve_discrete_are
from nn_closed_loop.utils.mpc import control_mpc

from os.path import dirname, join as pjoin
import scipy.io as sio
import scipy, os

class ISS(Dynamics):
    def __init__(self):

        self.continuous_time = False

        # mat_fname = 'iss.mat'
        # data_dir = pjoin(dirname(sio.__file__), 'dynamics')
        mat_fname = pjoin(os.getcwd(), 'iss.mat')

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
        self.dt = 0.1

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
            self.Q,
            self.R,
            self.Pinf,
            None,
            None,
            n_mpc=3,
            debug=False,
        )

    def dynamics_step(self, xs, us):
        # Dynamics are already discretized:
        xs_t1 = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xs_t1 += noise
        return xs_t1
