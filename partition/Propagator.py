
import torch
import numpy as np

class Propagator:
    def __init__(self):
        return

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = self.torch2network(network)

class CROWNIBPCodebasePropagator(Propagator):
    def __init__(self):
        Propagator.__init__(self)
        self.method_opt = None

    def torch2network(self, torch_model):
        from crown_ibp.bound_layers import BoundSequential
        torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
        return torch_model_

    def forward_pass(self, input_data):
        return self.network(torch.Tensor(input_data), method_opt=None).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        num_outputs = 2
        output_range = np.empty((num_outputs,2))
        for out_index in range(num_outputs):
            C = torch.zeros((1,1,num_outputs))
            C[0,0,out_index] = 1
            out_max, out_min = self.network(norm=np.inf,
                                        x_U=torch.Tensor([input_range[:,1]]),
                                        x_L=torch.Tensor([input_range[:,0]]),
                                        C=C,
                                        method_opt=self.method_opt,
                                        )[:2]
            output_range[out_index,:] = [out_min, out_max]
        return output_range, {}

class IBPPropagator(CROWNIBPCodebasePropagator):
    def __init__(self):
        CROWNIBPCodebasePropagator.__init__(self)
        self.method_opt = "interval_range"

class CROWNPropagator(CROWNIBPCodebasePropagator):
    def __init__(self):
        CROWNIBPCodebasePropagator.__init__(self)
        self.method_opt = "full_backward_range"