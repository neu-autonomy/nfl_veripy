import numpy as np
import torch
from nfl_veripy.utils.nn import load_controller

if __name__ == "__main__":
    torch_model = load_controller(
        system="GroundRobotSI", model_name="complex_potential_field"
    )
    num_control_inputs = 2

    ###########################
    # To get NN output bounds:
    print("---")
    print("Example of bounding the NN output associated with an input set.")

    ###########################
    # 'same_slope: True' makes U of smaller set to be contained by U of larger
    # set, but I don't think this is the answer
    from nfl_veripy.dynamics.GroundRobotSI import GroundRobotSI
    from nfl_veripy.utils.nn_bounds import BoundClosedLoopController

    torch_model_ = BoundClosedLoopController.convert(
        torch_model, {"same-slope": False, "zero-lb": True}, GroundRobotSI()
    )

    x0_min, x0_max, x1_min, x1_max = [-6, -5, 1, 2]

    # Compute matrices for u as function of x
    lower_A, upper_A, lower_sum_b, upper_sum_b = (
        torch_model_.full_backward_range(
            norm=np.inf,
            x_U=torch.Tensor([[x0_max, x1_max]]),
            x_L=torch.Tensor([[x0_min, x1_min]]),
            upper=True,
            lower=True,
            C=torch.eye(num_control_inputs).unsqueeze(0),
            return_matrices=True,
        )
    )
    upper_A = upper_A.detach().numpy()
    upper_sum_b = upper_sum_b.detach().numpy()
    lower_A = lower_A.detach().numpy()
    lower_sum_b = lower_sum_b.detach().numpy()

    x_max = np.array([x0_max, x1_max])
    x_min = np.array([x0_min, x1_min])

    print("U: {}x+{}".format(upper_A, upper_sum_b))
    print("L: {}x+{}".format(lower_A, lower_sum_b))

    from auto_LiRPA.bound_general import (
        BoundedModule,
        BoundedTensor,
        PerturbationLpNorm,
    )

    data_ub = torch.Tensor([[x0_max, x1_max]]).to("cuda:0")
    data_lb = torch.Tensor([[x0_min, x1_min]]).to("cuda:0")
    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=data_lb, x_U=data_ub)
    net = BoundedModule(
        torch_model,
        torch.zeros(2),
    )

    x = torch.Tensor([[-5.5, 1.5]]).to("cuda:0")
    data = BoundedTensor(x, ptb)
    A_dict = net.compute_bounds(
        x=(data,),
        C=torch.Tensor([[[1.0, -1.0]]]),
        method="backward",
    )

    #
    ###########################
