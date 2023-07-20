import numpy as np

from .Dynamics import Dynamics


class Unity(Dynamics):
    def __init__(self, nx=2, nu=2):
        self.continuous_time = False

        At = np.eye(nx)
        bt = np.ones((nx, nu))
        ct = np.zeros((nx,))

        # u_limits = None
        u_limits = np.vstack([-np.ones(nu), np.ones(nu)]).T
        # u_limits = np.array([[-1.0, 1.0]])  # (u0_min, u0_max)

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits)

        self.cmap_name = "tab10"

        self.name = "Unity_{}".format(str(nx).zfill(3))

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
