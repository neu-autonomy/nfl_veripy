from .ClosedLoopPropagator import ClosedLoopPropagator
import numpy as np
import pypoman
import nn_closed_loop.constraints as constraints
import torch
from nn_closed_loop.utils.utils import range_to_polytope
import requests
import json
from copy import deepcopy


class ClosedLoopOVERTPropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.url = 'http://localhost:8000/overt'

    def torch2network(self, torch_model):

        import pdb; pdb.set_trace()

        with open("test.txt", "ab") as f:
            for layer in torch_model:
                w = torch_model[layer].weight.data.numpy()
                b = torch_model[layer].bias.data.numpy()
                numpy.savetxt(f, a)



        return torch_model
        # from nn_closed_loop.utils.nn_bounds import BoundClosedLoopController

        # torch_model_cl = BoundClosedLoopController.convert(
        #     torch_model, dynamics=self.dynamics, bound_opts=self.params
        # )
        # return torch_model_cl

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_reachable_set(self, input_constraint, output_constraint, t_max):

        num_timesteps = len(np.arange(0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt)) + 1

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            raise NotImplementedError
        elif isinstance(input_constraint, constraints.LpConstraint):
            data = {
                'input_set': {
                    'low': input_constraint.range[..., 0].tolist(),
                    'high': input_constraint.range[..., 1].tolist(),
                },
                'num_timesteps': num_timesteps,
            }
            response = requests.post(self.url, data = json.dumps(data))
            ranges_list = json.loads(response.text)["result"]
            ranges = [np.array(r) for r in ranges_list]

            output_constraints = []
            for r in ranges_list:
                o = deepcopy(output_constraint)
                o.range = np.array(r)
                output_constraints.append(o)

        return output_constraints, {}

    def get_one_step_backprojection_set(self, output_constraint, input_constraint, num_partitions=None):
        raise NotImplementedError