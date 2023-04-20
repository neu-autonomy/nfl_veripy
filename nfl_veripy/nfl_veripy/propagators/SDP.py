from .Propagator import Propagator
import numpy as np
import torch.nn

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
            if isinstance(m, torch.nn.Sequential):
                continue
            elif isinstance(m, torch.nn.ReLU):
                if act is None or act == "relu":
                    act = "relu"
                else:
                    print(
                        "Multiple types of activations in your model --- unsuported by robust_sdp."
                    )
                    assert 0
            elif isinstance(m, torch.nn.Linear):
                dims.append(m.in_features)
            else:
                print("That layer isn't supported.")
                assert 0
        dims.append(m.out_features)
        if len(dims) != 3:
            print("robust sdp only supports 1 hidden layer (for now).")
            assert 0
        net = Network(dims, act, "rand")

        for name, param in torch_model.named_parameters():
            layer, typ = name.split(".")
            layer = int(int(layer) / 2)
            if typ == "weight":
                net._W[layer] = param.data.numpy()
            elif typ == "bias":
                net._b[layer] = np.expand_dims(param.data.numpy(), axis=-1)
            else:
                print("this layer isnt a weight or bias ???")

        return net

    def forward_pass(self, input_data):
        return self.torch_model(torch.Tensor(input_data)).data.numpy()

    def get_output_range(self, input_range, verbose=False):
        from partition.expt import robust_sdp

        output_range = robust_sdp(
            net=self.network,
            input_range=input_range,
            verbose=verbose,
            viz=False,
        )
        return output_range, {}
