import torch
import numpy as np
import partition.network_utils

class Propagator:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = self.torch2network(network)

#######################
# CROWN-IBP Codebase
#######################

class CROWNIBPCodebasePropagator(Propagator):
    def __init__(self, input_shape=None):
        Propagator.__init__(self, input_shape=input_shape)
        self.method_opt = None

    def torch2network(self, torch_model):
        from crown_ibp.bound_layers import BoundSequential
        torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
        return torch_model_

    def forward_pass(self, input_data):
        return self.network(torch.Tensor(input_data), method_opt=None).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        print("[WARNING] CROWNIBPCodebasePropagator.get_output_range is hard-coded for NNs with only 2 outputs...")
        num_outputs = 2
        output_range = np.empty((num_outputs,2))
        for out_index in range(num_outputs):
            C = torch.zeros((1,1,num_outputs))
            C[0,0,out_index] = 1
            x_U = torch.Tensor([input_range[:,1]])
            x_L = torch.Tensor([input_range[:,0]])
            out_max, out_min = self.network(norm=np.inf, x_U=x_U, x_L=x_L, C=C, method_opt=self.method_opt)[:2]
            output_range[out_index,:] = [out_min, out_max]
        return output_range, {}

class IBPPropagator(CROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None):
        CROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape)
        self.method_opt = "interval_range"

class CROWNPropagator(CROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None):
        CROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape)
        self.method_opt = "full_backward_range"



#######################
# Auto-LIRPA Codebase
#######################


class AutoLIRPAPropagator(Propagator):
    def __init__(self, input_shape=None, bound_opts={}):
        Propagator.__init__(self, input_shape=input_shape)
        self.bound_opts = bound_opts

    def torch2network(self, torch_model):
        from auto_LiRPA import BoundedModule

        my_input = torch.empty((1,)+self.input_shape)
        if hasattr(torch_model, "core"):
            torch_model = torch_model.core

        model = BoundedModule(torch_model, my_input, bound_opts=self.bound_opts)
        return model

    def forward_pass(self, input_data):
        return self.network(torch.Tensor(input_data)).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        from auto_LiRPA import PerturbationLpNorm, BoundedTensor

        center = (input_range[...,1] + input_range[...,0]) / 2.
        radius = ((input_range[...,1] - input_range[...,0]) / 2.).astype(np.float32)

        # Define perturbation
        ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
        # Make the input a BoundedTensor with perturbation
        my_input = BoundedTensor(torch.Tensor([center]), ptb)
        # Forward propagation using BoundedTensor
        # prediction = self.network(my_input)
        # Compute LiRPA bounds
        lb, ub = self.compute_bounds()

        num_outputs = lb.shape[-1]
        output_range = np.empty((num_outputs,2))
        output_range[:,0] = lb.data.numpy().squeeze()
        output_range[:,1] = ub.data.numpy().squeeze()

        return output_range, {}

class CROWNAutoLIRPAPropagator(AutoLIRPAPropagator):
    def __init__(self, input_shape=None, bound_opts={}):
        AutoLIRPAPropagator.__init__(self, input_shape=input_shape, bound_opts=bound_opts)
        self.method = "CROWN"

    def compute_bounds(self):
        lb, ub = self.network.compute_bounds(IBP=False, method="backward")
        return lb, ub

class FastLinAutoLIRPAPropagator(CROWNAutoLIRPAPropagator):
    def __init__(self, input_shape=None):
        CROWNAutoLIRPAPropagator.__init__(self, input_shape=input_shape, bound_opts={"relu": "same-slope"})
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
        return partition.network_utils.get_sampled_outputs(input_range, self, N=N)

    def samples_to_range(self, sampled_outputs):
        return partition.network_utils.samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range, {}

    def get_output_range(self, input_range):
        return self.get_exact_output_range(input_range)

#######################
# SDP Codebase
#######################

class SDPPropagator(Propagator):
    def __init__(self, input_shape=None):
        Propagator.__init__(self, input_shape=input_shape)

    def torch2network(self, torch_model):
        from robust_sdp.network import Network
        self.torch_model = torch_model
        dims = []
        act = None
        for idx, m in enumerate(torch_model.modules()):
            if isinstance(m, torch.nn.Sequential): continue
            elif isinstance(m, torch.nn.ReLU):
                if act is None or act == 'relu':
                    act = 'relu'
                else:
                    print('Multiple types of activations in your model --- unsuported by robust_sdp.')
                    assert(0)
            elif isinstance(m, torch.nn.Linear):
                dims.append(m.in_features)
            else:
                print("That layer isn't supported.")
                assert(0)
        dims.append(m.out_features)
        if len(dims) != 3:
            print("robust sdp only supports 1 hidden layer (for now).")
            assert(0)
        net = Network(dims, act, 'rand')

        for name, param in torch_model.named_parameters():
            layer, typ = name.split('.')
            layer = int(int(layer)/2)
            if typ == 'weight':
                net._W[layer] = param.data.numpy()
            elif typ == 'bias':
                net._b[layer] = np.expand_dims(param.data.numpy(), axis=-1)
            else:
                print('this layer isnt a weight or bias ???')

        return net

    def forward_pass(self, input_data):
        return self.torch_model(torch.Tensor(input_data)).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        from partition.expt import robust_sdp
        output_range = robust_sdp(net=self.network, input_range=input_range, verbose=verbose, viz=False)
        return output_range, {}        
