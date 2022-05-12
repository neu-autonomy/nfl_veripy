from .Propagator import Propagator
import nn_partition.utils.utils
import numpy as np
import torch

#######################
# Auto-LIRPA Codebase
#######################

# Suppress autoLIRPA logging info...
import logging
from auto_LiRPA.utils import logger

logger.setLevel(logging.WARNING)

import time


class AutoLIRPAPropagator(Propagator):
    def __init__(self, input_shape=None, bound_opts={}):
        Propagator.__init__(self, input_shape=input_shape)
        self.bound_opts = bound_opts

    def torch2network(self, torch_model):
        from auto_LiRPA import BoundedModule

        my_input = torch.empty((1,) + self.input_shape)
        if hasattr(torch_model, "core"):
            torch_model = torch_model.core

        model = BoundedModule(
            torch_model, my_input, bound_opts=self.bound_opts
        )
        return model

    def forward_pass(self, input_data):
        return self.network(torch.Tensor(input_data)).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        from auto_LiRPA import PerturbationLpNorm, BoundedTensor

        center = (input_range[..., 1] + input_range[..., 0]) / 2.0
        radius = ((input_range[..., 1] - input_range[..., 0]) / 2.0).astype(
            np.float32
        )

        # Define perturbation
        ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
        # Make the input a BoundedTensor with perturbation
        my_input = BoundedTensor(torch.Tensor(np.expand_dims(center, 0)), ptb)
        # Forward propagation using BoundedTensor
        prediction = self.network(my_input)
        # Compute LiRPA bounds
        lb, ub = self.compute_bounds()

        num_outputs = lb.shape[-1]
        output_range = np.empty((num_outputs, 2))
        output_range[:, 0] = lb.data.numpy().squeeze()
        output_range[:, 1] = ub.data.numpy().squeeze()

        return output_range, {}


class CROWNAutoLIRPAPropagator(AutoLIRPAPropagator):
    def __init__(self, input_shape=None, bound_opts={}):
        AutoLIRPAPropagator.__init__(
            self, input_shape=input_shape, bound_opts=bound_opts
        )
        self.method = "CROWN"

    def compute_bounds(self):
        lb, ub = self.network.compute_bounds(IBP=False, method="backward")
        return lb, ub


class FastLinAutoLIRPAPropagator(CROWNAutoLIRPAPropagator):
    def __init__(self, input_shape=None):
        CROWNAutoLIRPAPropagator.__init__(
            self, input_shape=input_shape, bound_opts={"relu": "same-slope"}
        )
        self.method = "Fast-Lin"


class IBPAutoLIRPAPropagator(AutoLIRPAPropagator):
    def __init__(self, input_shape=None):
        AutoLIRPAPropagator.__init__(self, input_shape=input_shape)
        self.method = "IBP"

    def compute_bounds(self):
        lb, ub = self.network.compute_bounds(IBP=True, method=None)
        return lb, ub


class CROWNIBPAutoLIRPAPropagator(AutoLIRPAPropagator):
    def __init__(self, input_shape=None):
        AutoLIRPAPropagator.__init__(self, input_shape=input_shape)
        self.method = "CROWN-IBP"

    def compute_bounds(self):
        # not completely sure how to blend CROWN and IBP here
        # see L96 on https://github.com/KaidiXu/auto_LiRPA/blob/master/examples/vision/simple_training.py
        raise NotImplementedError


class ExhaustiveAutoLIRPAPropagator(AutoLIRPAPropagator):
    def __init__(self, input_shape=None):
        AutoLIRPAPropagator.__init__(self, input_shape=input_shape)
        self.method = "exhaustive"

    def get_sampled_outputs(self, input_range, N=1000):
        return nn_partition.utils.utils.get_sampled_outputs(
            input_range, self, N=N
        )

    def samples_to_range(self, sampled_outputs):
        return nn_partition.utils.utils.samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range, {}

    def get_output_range(self, input_range):
        return self.get_exact_output_range(input_range)
