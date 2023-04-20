import numpy as np

from .Dynamics import ContinuousTimeDynamics


class Duffing(ContinuousTimeDynamics):
    def __init__(self):
        dim = 2
        zeta = 0.3
        dt = 0.05
        At = np.array([[0, 1], [-1, -2 * zeta]], dtype=float)

        bt = np.zeros((dim, 1), dtype=float)
        bt[1][0] = 1.0

        ct = np.zeros((dim,), dtype=float)

        u_limits = None

        self.dim = dim
        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt)

        self.cmap_name = "tab20"

    def dynamics(self, xs, us):
        # Same as CT dynamics, but also add x1**3 term
        xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        xdot[:, 1] = xdot[:, 1] - np.power(xs[:, 0], 3)
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:, 0],
                high=self.process_noise[:, 1],
                size=xs.shape,
            )
            xdot += noise
        return xdot
