import json
import os
from copy import deepcopy

import nfl_veripy.constraints as constraints
import numpy as np
import requests
import torch

from .ClosedLoopPropagator import ClosedLoopPropagator

model_dir = "{}/../../models/nnet/".format(
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(model_dir, exist_ok=True)


class ClosedLoopOVERTPropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.url = "http://localhost:8000/overt"
        self.model_filename = model_dir + "/tmp_model.nnet"

    def torch2network(self, torch_model):
        # Write a .nnet file based on the torch_model
        act = None
        dims = []
        with open(self.model_filename, "a") as f:
            f.truncate(0)

            for idx, m in enumerate(torch_model.modules()):
                if isinstance(m, torch.nn.Sequential):
                    continue
                elif isinstance(m, torch.nn.ReLU):
                    if act is None or act == "relu":
                        act = "relu"
                    else:
                        print(
                            "Multiple types of activations in your model ---"
                            " unsuported by robust_sdp."
                        )
                        assert 0
                elif isinstance(m, torch.nn.Linear):
                    dims.append(m.in_features)
                else:
                    print("That layer isn't supported.")
                    assert 0
            dims.append(m.out_features)
            f.write(str(len(dims) - 1) + "\n")
            f.write(", ".join([str(d) for d in dims]) + "\n")
            for i in range(5):
                f.write("0\n")

            for idx, m in enumerate(torch_model.modules()):
                if isinstance(m, torch.nn.Sequential):
                    continue
                elif isinstance(m, torch.nn.ReLU):
                    if act is None or act == "relu":
                        act = "relu"
                    else:
                        print(
                            "Multiple types of activations in your model ---"
                            " unsuported by robust_sdp."
                        )
                        assert 0
                elif isinstance(m, torch.nn.Linear):
                    w = m.weight.data.numpy()
                    np.savetxt(f, w, fmt="%s", delimiter=", ")
                    b = m.bias.data.numpy()
                    np.savetxt(f, b, fmt="%s", delimiter=", ")
                else:
                    print("That layer isn't supported.")
                    assert 0

        # Internally, we'll just use the typical torch stuff
        return torch_model

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        num_timesteps = (
            len(
                np.arange(
                    0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
                )
            )
            + 1
        )

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            raise NotImplementedError
        elif isinstance(input_constraint, constraints.LpConstraint):
            data = {
                "input_set": {
                    "low": input_constraint.range[..., 0].tolist(),
                    "high": input_constraint.range[..., 1].tolist(),
                },
                "num_timesteps": num_timesteps,
                "controller": self.model_filename,
                "dt": self.dynamics.dt,
                "system": self.dynamics.__class__.__name__,
            }
            response = requests.post(self.url, data=json.dumps(data))
            ranges_list = json.loads(response.text)["result"]

            output_constraints = []
            for r in ranges_list:
                o = deepcopy(output_constraint)
                o.range = np.array(r)
                output_constraints.append(o)

        return output_constraints, {}

    def get_one_step_backprojection_set(
        self, output_constraint, input_constraint, num_partitions=None
    ):
        raise NotImplementedError
