import numpy as np
from nfl_veripy.utils.mpc import control_mpc
from scipy.linalg import solve_discrete_are

from .Dynamics import DiscreteTimeDynamics


class GroundRobotDI(DiscreteTimeDynamics):
    def __init__(self):
        self.continuous_time = False

        At = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        bt = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        ct = np.array([0.0, 0.0, 0.0, 0.0]).T

        # u_limits = None
        u_limits = 1 * np.array([[-4, 4], [-4, 4]])

        x_limits = np.array(
            [
                [-1e2, 1e2],
                [-1e2, 1e2],
                [-1, 1],
                [-1, 1],
            ]
        )
        x_limits = {2: [-1, 1], 3: [-1, 1]}

        # x_limits=None

        dt = 1

        super().__init__(
            At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt, x_limits=x_limits
        )

        self.cmap_name = "tab10"

    def control_mpc(self, x0):
        # LQR-MPC parameters
        if not hasattr(self, "Q"):
            self.Q = np.eye(2)
        if not hasattr(self, "R"):
            self.R = np.eye(2)
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

        # Control function for if model gives [v, w] command inputs
        # def control_nn(self, x, model):
        #     if x.ndim == 1:
        #         batch_x = np.expand_dims(x, axis=0)
        #     else:
        #         batch_x = x
        #     us = model.forward(torch.Tensor(batch_x)).data.numpy()
        #     if not hasattr(self, 'theta') or len(self.theta) != len(us):
        #         self.theta = np.zeros(len(us))

        # R = np.array(
        #     [
        #         [
        #             [np.cos(theta), -self.r * np.sin(theta)],
        #             [np.sin(theta), self.r * np.cos(theta)],
        #         ]
        #         for theta in self.theta
        #     ]
        # )

    #     us_transformed = np.array([R[i]@us[i] for i in range(len(us))])

    #     # print("theta: {}".format(self.theta[0]))
    #     # print("transformed u: {}".format(us_transformed[0]))
    #     # print("x-direction: {}".format(R[0][:,0]))
    #     self.theta = self.theta + us[:,1]
    #     return us_transformed
