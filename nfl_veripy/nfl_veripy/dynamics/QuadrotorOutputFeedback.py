import numpy as np

from .Quadrotor import Quadrotor


class QuadrotorOutputFeedback(Quadrotor):
    def __init__(self, process_noise=None, sensing_noise=None):
        super().__init__()
        if process_noise is None:
            self.process_noise = (
                0.005
                * np.dstack(
                    [-np.ones(self.num_states), np.ones(self.num_states)]
                )[0]
            )
        else:
            self.process_noise = (
                process_noise
                * 0.05
                * np.dstack(
                    [-np.ones(self.num_states), np.ones(self.num_states)]
                )[0]
            )

        if sensing_noise is None:
            self.sensor_noise = (
                0.001
                * np.dstack(
                    [-np.ones(self.num_outputs), np.ones(self.num_outputs)]
                )[0]
            )
        else:
            self.process_noise = (
                sensing_noise
                * np.dstack(
                    [-np.ones(self.num_states), np.ones(self.num_states)]
                )[0]
            )
