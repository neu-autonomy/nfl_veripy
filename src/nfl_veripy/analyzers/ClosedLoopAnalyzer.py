from typing import Any

import numpy as np
import torch

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.partitioners as partitioners
import nfl_veripy.propagators as propagators
import nfl_veripy.visualizers as visualizers
from nfl_veripy.utils.utils import get_sampled_outputs, samples_to_range

from .Analyzer import Analyzer


class ClosedLoopAnalyzer(Analyzer):
    def __init__(
        self, torch_model: torch.nn.Sequential, dynamics: dynamics.Dynamics
    ):
        self.torch_model = torch_model
        self.dynamics = dynamics
        super().__init__(torch_model=torch_model)

    @property
    def partitioner_dict(self) -> dict:
        return partitioners.partitioner_dict

    @property
    def propagator_dict(self) -> dict:
        return propagators.propagator_dict

    def instantiate_partitioner(
        self, partitioner_name: str, hyperparams: dict[str, Any]
    ) -> partitioners.ClosedLoopPartitioner:
        partitioner = partitioners.partitioner_dict[partitioner_name](
            self.dynamics
        )
        for key, value in hyperparams.items():
            if hasattr(partitioner, key):
                setattr(partitioner, key, value)
        return partitioner

    def instantiate_propagator(
        self, propagator_name: str, hyperparams: dict[str, Any]
    ) -> propagators.ClosedLoopPropagator:
        propagator = propagators.propagator_dict[propagator_name](
            self.dynamics
        )
        for key, value in hyperparams.items():
            if hasattr(propagator, key):
                setattr(propagator, key, value)
        return propagator

    def instantiate_visualizer(
        self, visualizer_name: str, hyperparams: dict[str, Any]
    ) -> visualizers.ForwardVisualizer:
        visualizer = visualizers.ForwardVisualizer(self.dynamics)
        for key, value in hyperparams.items():
            if hasattr(visualizer, key):
                setattr(visualizer, key, value)
        return visualizer

    def get_one_step_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        reachable_set, info = self.partitioner.get_one_step_reachable_set(
            initial_set, self.propagator
        )
        return reachable_set, info

    def get_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint, t_max: float
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        reachable_set, info = self.partitioner.get_reachable_set(
            initial_set, self.propagator, t_max
        )
        return reachable_set, info

    def get_sampled_outputs(
        self, input_range: np.ndarray, num_samples: int = 1000
    ) -> np.ndarray:
        return get_sampled_outputs(
            input_range, self.propagator, num_samples=num_samples
        )

    def get_sampled_output_range(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        t_max: int = 5,
        num_samples: int = 1000,
    ) -> np.ndarray:
        return self.partitioner.get_sampled_out_range(
            initial_set, self.propagator, t_max, num_samples
        )

    def get_exact_output_range(
        self,
        input_range: np.ndarray,
        num_samples: int = int(1e4),
    ) -> np.ndarray:
        sampled_outputs = self.get_sampled_outputs(
            input_range, num_samples=num_samples
        )
        output_range = samples_to_range(sampled_outputs)
        return output_range

    def get_error(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        reachable_sets: constraints.MultiTimestepConstraint,
        t_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.partitioner.get_error(
            initial_set, reachable_sets, self.propagator, t_max
        )

    def visualize(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        reachable_sets: constraints.MultiTimestepConstraint,
        network: torch.nn.Sequential,
        **kwargs,
    ) -> None:
        self.visualizer.visualize(
            initial_set, reachable_sets, network, **kwargs
        )
