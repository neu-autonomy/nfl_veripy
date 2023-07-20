import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

import nfl_veripy.partitioners as partitioners
import nfl_veripy.propagators as propagators
import nfl_veripy.visualizers as visualizers
from nfl_veripy.utils.utils import get_sampled_outputs, samples_to_range

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = "20"


class Analyzer:
    def __init__(self, torch_model):
        self.torch_model = torch_model

        self.partitioner = None
        self.propagator = None
        self.visualizer = None

    @property
    def partitioner_dict(self):
        return partitioners.partitioner_dict

    @property
    def propagator_dict(self):
        return propagators.propagator_dict

    @property
    def partitioner(self):
        return self._partitioner

    @partitioner.setter
    def partitioner(self, hyperparams):
        if hyperparams is None:
            return

        hyperparams_ = hyperparams.copy()
        partitioner = hyperparams_.pop("type", None)

        self._partitioner = self.instantiate_partitioner(
            partitioner, hyperparams_
        )

    def instantiate_partitioner(self, partitioner, hyperparams):
        return self.partitioner_dict[partitioner](**hyperparams)

    @property
    def propagator(self):
        return self._propagator

    @propagator.setter
    def propagator(self, hyperparams):
        if hyperparams is None:
            return
        hyperparams_ = hyperparams.copy()
        propagator = hyperparams_.pop("type", None)

        self._propagator = self.instantiate_propagator(
            propagator, hyperparams_
        )
        if propagator is not None:
            self._propagator.network = self.torch_model

    def instantiate_propagator(self, propagator, hyperparams):
        return self.propagator_dict[propagator](**hyperparams)

    @property
    def visualizer(self):
        return self._visualizer

    @visualizer.setter
    def visualizer(self, hyperparams):
        if hyperparams is None:
            return

        hyperparams_ = hyperparams.copy()
        visualizer = hyperparams_.pop("type", None)

        self._visualizer = self.instantiate_visualizer(
            visualizer, hyperparams_
        )

    def get_output_range(self, input_range, verbose=False):
        output_range, info = self.partitioner.get_output_range(
            input_range, self.propagator
        )
        return output_range, info

    def get_sampled_outputs(
        self, input_range: np.ndarray, num_samples: int = 1000
    ):
        return get_sampled_outputs(
            input_range, self.propagator, num_samples=num_samples
        )

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def get_exact_output_range(
        self, input_range: np.ndarray, num_samples: int = int(1e4)
    ) -> np.ndarray:
        sampled_outputs = self.get_sampled_outputs(
            input_range, num_samples=num_samples
        )
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

    def get_exact_hull(self, input_range, num_samples=int(1e4)):
        sampled_outputs = self.get_sampled_outputs(
            input_range, num_samples=num_samples
        )
        return ConvexHull(sampled_outputs)

    def get_error_np(self, input_range, output_range, **analyzer_info):
        if self.partitioner.interior_condition == "convex_hull":
            exact_hull = self.get_exact_hull(input_range)

            error = self.partitioner.get_error(
                exact_hull, analyzer_info["estimated_hull"]
            )
        elif self.partitioner.interior_condition in ["lower_bnds", "linf"]:
            output_range_exact = self.get_exact_output_range(input_range)

            error = self.partitioner.get_error(
                output_range_exact, output_range
            )

        return error
