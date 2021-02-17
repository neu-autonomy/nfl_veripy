from .Propagator import Propagator
import torch
import numpy as np

#######################
# CROWN-IBP Codebase
#######################


class CROWNIBPCodebasePropagator(Propagator):
    def __init__(self, input_shape=None):
        Propagator.__init__(self, input_shape=input_shape)
        self.method_opt = None

    def torch2network(self, torch_model):
        from crown_ibp.bound_layers import BoundSequential

        torch_model_ = BoundSequential.convert(
            torch_model, {"same-slope": True}
        )
        return torch_model_

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        print(
            "[WARNING] CROWNIBPCodebasePropagator.get_output_range is hard-coded for NNs with only 2 outputs..."
        )
        num_outputs = 2
        output_range = np.empty((num_outputs, 2))
        for out_index in range(num_outputs):
            C = torch.zeros((1, 1, num_outputs))
            C[0, 0, out_index] = 1
            x_U = torch.Tensor([input_range[:, 1]])
            x_L = torch.Tensor([input_range[:, 0]])
            out_max, out_min = self.network(
                norm=np.inf, x_U=x_U, x_L=x_L, C=C, method_opt=self.method_opt
            )[:2]
            output_range[out_index, :] = [out_min, out_max]
        return output_range, {}


class IBPPropagator(CROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None):
        CROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape)
        self.method_opt = "interval_range"


class CROWNPropagator(CROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None):
        CROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape)
        self.method_opt = "full_backward_range"
