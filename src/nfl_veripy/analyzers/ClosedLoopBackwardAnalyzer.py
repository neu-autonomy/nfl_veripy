from typing import Any

import numpy as np
import torch

import nfl_veripy.analyzers as analyzers
import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.partitioners as partitioners
import nfl_veripy.propagators as propagators
import nfl_veripy.visualizers as visualizers
from nfl_veripy.utils.utils import get_sampled_outputs, samples_to_range

from .Analyzer import Analyzer


class ClosedLoopBackwardAnalyzer(Analyzer):
    def __init__(
        self, torch_model: torch.nn.Sequential, dynamics: dynamics.Dynamics
    ):
        self.torch_model = torch_model
        self.dynamics = dynamics
        analyzers.Analyzer.__init__(self, torch_model=torch_model)

    @property
    def partitioner_dict(self):
        return partitioners.partitioner_dict

    @property
    def propagator_dict(self):
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
    ) -> visualizers.BackwardVisualizer:
        visualizer = visualizers.BackwardVisualizer(self.dynamics)
        for key, value in hyperparams.items():
            if hasattr(visualizer, key):
                setattr(visualizer, key, value)
        return visualizer

    def get_one_step_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        backprojection_set, info = (
            self.partitioner.get_one_step_backprojection_set(
                target_set, self.propagator, overapprox=overapprox
            )
        )
        return backprojection_set, info

    def get_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        backprojection_set, info = self.partitioner.get_backprojection_set(
            target_set, self.propagator, t_max, overapprox=overapprox
        )
        return backprojection_set, info

    def get_N_step_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
        num_partitions=None,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        backprojection_set, info = (
            self.partitioner.get_N_step_backprojection_set(
                target_set,
                self.propagator,
                t_max,
                num_partitions=num_partitions,
                overapprox=overapprox,
            )
        )
        return backprojection_set, info

    def get_backprojection_error(
        self,
        target_set: constraints.SingleTimestepConstraint,
        backprojection_sets: constraints.MultiTimestepConstraint,
        t_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.partitioner.get_backprojection_error(
            target_set,
            backprojection_sets,
            self.propagator,
            t_max,
            backreachable_sets=None,
        )

    def get_sampled_outputs(self, input_range, N=1000):
        return get_sampled_outputs(input_range, self.propagator, N=N)

    def get_sampled_output_range(
        self, input_constraint, t_max=5, num_samples=1000
    ):
        return self.partitioner.get_sampled_out_range(
            input_constraint, self.propagator, t_max, num_samples
        )

    def get_output_range(self, input_constraint, output_constraint):
        return self.partitioner.get_output_range(
            input_constraint, output_constraint
        )

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

    def get_error(self, input_constraint, output_constraint, t_max):
        return self.partitioner.get_error(
            input_constraint, output_constraint, self.propagator, t_max
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
