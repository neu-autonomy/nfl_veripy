import numpy as np

from .DoubleIntegrator import DoubleIntegrator


class DoubleIntegratorOutputFeedback(DoubleIntegrator):
    def __init__(self, process_noise=None, sensing_noise=None):
        super().__init__()
        # self.process_noise = np.array(
        #     [
        #         [-0.5, 0.5],
        #         [-0.01, 0.01],
        #     ]
        # )
        if process_noise is None:
            self.process_noise = (
                0.1
                * np.dstack(
                    [-np.ones(self.num_states), np.ones(self.num_states)]
                )[0]
            )
        else:
            self.process_noise = (
                process_noise
                * np.dstack(
                    [-np.ones(self.num_states), np.ones(self.num_states)]
                )[0]
            )

        # self.sensor_noise = np.array(
        #     [
        #         [-0.8, 0.8],
        #         [-0.0, 0.0],
        #     ]
        # )

        if sensing_noise is None:
            self.sensor_noise = (
                0.1
                * np.dstack(
                    [-np.ones(self.num_outputs), np.ones(self.num_outputs)]
                )[0]
            )
        else:
            self.sensor_noise = (
                sensing_noise
                * np.dstack(
                    [-np.ones(self.num_outputs), np.ones(self.num_outputs)]
                )[0]
            )
