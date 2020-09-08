import numpy as np
from partition.Propagator import Propagator

class ClosedLoopPropagator(Propagator):
    def __init__(self, input_shape=None):
        Propagator.__init__(self, input_shape=input_shape)

    @property
    def cl_network(self):
        return self._cl_network

    @cl_network.setter
    def cl_network(self, cl_network):
        self._cl_network = self.torch2cl_network(network)

class ClosedLoopCROWNIBPCodebasePropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None):
        ClosedLoopPropagator.__init__(self, input_shape=input_shape)

    def torch2network(self, torch_model):
        from crown_ibp.bound_layers import BoundSequential
        torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
        return torch_model_

    def torch2cl_network(self, torch_model):
        from closed_loop.nn_bounds import BoundClosedLoopController
        # torch_model = keras2torch(keras_model, "torch_model")
        # crown_params = {"zero-lb": True}
        # crown_params = {"one-lb": True}
        crown_params = {"same-slope": True}
        torch_model_cl = BoundClosedLoopController.convert(torch_model, crown_params,
            A_dyn=torch.Tensor([At]), b_dyn=torch.Tensor([bt]), c_dyn=[ct])
        return torch_model_cl

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

    def reachLP_1(self, torch_model_cl, A_inputs, b_inputs, A_out, u_limits=None):
        # Get bounds on each state from A_inputs, b_inputs
        num_states = At.shape[0]
        vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
        x_max = []
        x_min = []
        for state in range(num_states):
            x_max.append(np.max([v[state] for v in vertices]))
            x_min.append(np.min([v[state] for v in vertices]))
        
        num_facets = A_out.shape[0]
        bs = np.zeros((num_facets))
        for i in range(num_facets):
            xt1_max, _, xt1_min, _ = self.cl_network.full_backward_range(norm=np.inf,
                                        x_U=torch.Tensor([x_max]),
                                        x_L=torch.Tensor([x_min]),
                                        upper=True, lower=True, C=torch.Tensor([[[1]]]),
                                        A_out=torch.Tensor([A_out[i,:]]),
                                        A_in=A_inputs, b_in=b_inputs,
                                        u_limits=u_limits)
            bs[i] = xt1_max
        return bs

    def reachLP_n(self, n, A_inputs, b_inputs, A_out, u_limits=None):
        # Call reach_LP_1 sequentially n times for the n-step reachable set.
        all_bs = []
        bs = self.reachLP_1(A_inputs, b_inputs, A_out, u_limits=u_limits)
        all_bs.append(bs)
        for i in range(1,n):
            bs = self.reachLP_1(A_out, bs, A_out, u_limits=u_limits)
            all_bs.append(bs)
        return all_bs

class ClosedLoopIBPPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape)
        self.method_opt = "interval_range"

class ClosedLoopCROWNPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(self, input_shape=input_shape)
        self.method_opt = "full_backward_range"
