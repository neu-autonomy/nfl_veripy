import logging

import cvxpy as cp
import numpy as np
import torch

from crown_ibp.bound_layers import BoundSequential

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BoundClosedLoopController(BoundSequential):
    def __init__(self, dynamics, layers):
        super(BoundClosedLoopController, self).__init__(*layers)
        self.dynamics = dynamics
        self.try_to_use_closed_form = True

    # Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model, bound_opts=None, dynamics=None):
        layers = BoundClosedLoopController.sequential_model_to_layers(
            sequential_model, bound_opts=bound_opts
        )

        if dynamics.u_limits is not None:
            # Borrowing brilliant idea from Reach-SDP paper to add 2 ReLUs
            # to handle control limits

            from copy import deepcopy

            from crown_ibp.bound_layers import BoundLinear, BoundReLU

            # To ensure CROWN doesn't exceed the control limits below
            if bound_opts is None:
                bound_opts = {}
            else:
                control_limit_bound_opts = deepcopy(bound_opts)
            control_limit_bound_opts["zero-lb"] = True
            control_limit_bound_opts["same-slope"] = False

            num_inputs = dynamics.bt.shape[-1]

            # Last b += u_min
            layers[-1].bias.data += -dynamics.u_limits[:, 0].astype(np.float32)

            # ReLU
            layers.append(
                BoundReLU(layers[-1], bound_opts=control_limit_bound_opts)
            )

            # W = -I, b = u_max - u_min
            layer = BoundLinear(
                num_inputs,
                num_inputs,
                True,
                bound_opts=control_limit_bound_opts,
            )
            layer.weight.data = -torch.eye(num_inputs)
            layer.bias.data = torch.Tensor(
                dynamics.u_limits[:, 1] - dynamics.u_limits[:, 0]
            )
            layers.append(layer)

            # ReLU
            layers.append(
                BoundReLU(layers[-1], bound_opts=control_limit_bound_opts)
            )

            # W = -I, b = u_max
            layer = BoundLinear(
                num_inputs,
                num_inputs,
                True,
                bound_opts=control_limit_bound_opts,
            )
            layer.weight.data = -torch.eye(num_inputs)
            layer.bias.data = torch.Tensor(dynamics.u_limits[:, 1])
            layers.append(layer)

        b = BoundClosedLoopController(dynamics=dynamics, layers=layers)

        if "try_to_use_closed_form" in bound_opts:
            b.try_to_use_closed_form = bound_opts["try_to_use_closed_form"]

        return b

    def _add_dynamics(
        self, lower_A, upper_A, lower_sum_b, upper_sum_b, A_out, dynamics
    ):
        """
        From CROWN paper:
        Lambda = upper_A
        Omega = lower_A
        Delta = upper_sum_b
        Theta = lower_sum_b
        --> These are the slopes and intercepts of the control input lin bnds
        i.e., Omega x + Theta <= u <= Lambda x + Delta (forall x in X0)
        """

        # Determine which control inputs should use max vs. min for biggest
        # next state
        flip = torch.matmul(A_out, torch.Tensor(np.array([dynamics.bt]))) < 0
        flip_A = flip.repeat(1, dynamics.num_states, 1).transpose(2, 1)
        flip_b = flip.squeeze(1)

        # Populate replacements for Lambda, Omega, Delta, Theta using flip
        upsilon = torch.where(flip_A, lower_A, upper_A)
        psi = torch.where(flip_b, lower_sum_b, upper_sum_b)
        xi = torch.where(flip_A, upper_A, lower_A)
        gamma = torch.where(flip_b, upper_sum_b, lower_sum_b)

        # Fill in best/worst-case process noise realizations
        if dynamics.process_noise is None:
            process_noise_low = process_noise_high = torch.zeros_like(
                torch.Tensor(np.array([dynamics.ct]))
            )
        else:
            flip_process_noise = A_out < 0
            process_noise_low = torch.where(
                flip_process_noise,
                torch.Tensor(dynamics.process_noise[:, 1]),
                torch.Tensor(dynamics.process_noise[:, 0]),
            )
            process_noise_high = torch.where(
                flip_process_noise,
                torch.Tensor(dynamics.process_noise[:, 0]),
                torch.Tensor(dynamics.process_noise[:, 1]),
            )

        # Fill in best/worst-case sensor noise realizations
        if dynamics.sensor_noise is None:
            sensor_noise_low = sensor_noise_high = torch.zeros_like(
                torch.Tensor(np.array([dynamics.ct]))
            ).unsqueeze(-1)
        else:
            flip_sensor_noise_low = (
                torch.matmul(A_out, torch.Tensor(np.array([dynamics.bt]))).bmm(
                    xi
                )
                < 0
            )
            flip_sensor_noise_high = (
                torch.matmul(A_out, torch.Tensor(np.array([dynamics.bt]))).bmm(
                    upsilon
                )
                < 0
            )
            sensor_noise_low = torch.where(
                flip_sensor_noise_low,
                torch.Tensor(dynamics.sensor_noise[:, 0]),
                torch.Tensor(dynamics.sensor_noise[:, 1]),
            ).transpose(2, 1)
            sensor_noise_high = torch.where(
                flip_sensor_noise_high,
                torch.Tensor(dynamics.sensor_noise[:, 0]),
                torch.Tensor(dynamics.sensor_noise[:, 1]),
            ).transpose(2, 1)

        # Compute slope and intercept of closed-loop system linear bounds
        if dynamics.continuous_time:
            # x_{t+1} = (
            #     x_t+dt*x_dot = x_t+dt*(Ax+bu+c) <= (I+dt*(A+bY))x + dt*(bG)
            # )
            lower_A_with_dyn = torch.matmul(
                A_out,
                torch.eye(dynamics.num_states).unsqueeze(0)
                + dynamics.dt
                * (
                    torch.Tensor(np.array([dynamics.At]))
                    + torch.Tensor(np.array([dynamics.bt])).bmm(xi)
                ),
            )
            upper_A_with_dyn = torch.matmul(
                A_out,
                torch.eye(dynamics.num_states).unsqueeze(0)
                + dynamics.dt
                * (
                    torch.Tensor(np.array([dynamics.At]))
                    + torch.Tensor(np.array([dynamics.bt])).bmm(upsilon)
                ),
            )
            lower_sum_b_with_dyn = torch.matmul(
                A_out,
                dynamics.dt
                * (
                    torch.Tensor(np.array([dynamics.bt])).bmm(
                        gamma.unsqueeze(-1)
                    )
                    + torch.Tensor(np.array([dynamics.ct])).unsqueeze(-1)
                    + process_noise_low.unsqueeze(-1)
                    + torch.Tensor(np.array([dynamics.bt]))
                    .bmm(xi)
                    .bmm(sensor_noise_low)
                ),
            )
            upper_sum_b_with_dyn = torch.matmul(
                A_out,
                dynamics.dt
                * (
                    torch.Tensor(np.array([dynamics.bt])).bmm(
                        psi.unsqueeze(-1)
                    )
                    + torch.Tensor(np.array([dynamics.ct])).unsqueeze(-1)
                    + process_noise_high.unsqueeze(-1)
                    + torch.Tensor(np.array([dynamics.bt]))
                    .bmm(upsilon)
                    .bmm(sensor_noise_high)
                ),
            )
        else:
            # x_{t+1} = Ax+bu+c <= (A+bY)x+bG
            lower_A_with_dyn = torch.matmul(
                A_out,
                torch.Tensor(np.array([dynamics.At]))
                + torch.Tensor(np.array([dynamics.bt])).bmm(xi),
            )
            upper_A_with_dyn = torch.matmul(
                A_out,
                torch.Tensor(np.array([dynamics.At]))
                + torch.Tensor(np.array([dynamics.bt])).bmm(upsilon),
            )
            lower_sum_b_with_dyn = torch.matmul(
                A_out,
                torch.Tensor(np.array([dynamics.bt])).bmm(gamma.unsqueeze(-1))
                + torch.Tensor(np.array([dynamics.ct])).unsqueeze(-1)
                + process_noise_low.unsqueeze(-1)
                + torch.Tensor(np.array([dynamics.bt]))
                .bmm(xi)
                .bmm(sensor_noise_low),
            )
            upper_sum_b_with_dyn = torch.matmul(
                A_out,
                torch.Tensor(np.array([dynamics.bt])).bmm(psi.unsqueeze(-1))
                + torch.Tensor(np.array([dynamics.ct])).unsqueeze(-1)
                + process_noise_high.unsqueeze(-1)
                + torch.Tensor(np.array([dynamics.bt]))
                .bmm(upsilon)
                .bmm(sensor_noise_high),
            )

        return (
            lower_A_with_dyn,
            upper_A_with_dyn,
            lower_sum_b_with_dyn,
            upper_sum_b_with_dyn,
        )

    # High level function, will be called outside
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, spec_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def backward_range(
        self,
        norm=np.inf,
        x_U=None,
        x_L=None,
        eps=None,
        C=None,
        upper=False,
        lower=True,
        modules=None,
        return_matrices=False,
        final_layer=False,
    ):
        if not final_layer:
            # For non-final layers, don't worry about dynamics yet
            return super().backward_range(
                norm=norm,
                x_U=x_U,
                x_L=x_L,
                eps=eps,
                C=C,
                upper=upper,
                lower=lower,
                modules=modules,
            )
        else:
            (
                lower_A,
                upper_A,
                lower_sum_b,
                upper_sum_b,
            ) = self._prop_from_last_layer(
                C=C, x_U=x_U, modules=modules, upper=upper, lower=lower
            )

            if return_matrices:
                # The caller doesn't care about the actual NN bounds,
                # just the matrices (slope, intercept) used to compute them
                return lower_A, upper_A, lower_sum_b, upper_sum_b

            ub, lb = self.compute_bound_from_matrices(
                lower_A, lower_sum_b, upper_A, upper_sum_b, x_U, x_L, norm
            )
            return ub, lb

    def nonlinear_compute_bound_from_matrices(
        self,
        lower_A_with_dyn,
        lower_sum_b_with_dyn,
        upper_A_with_dyn,
        upper_sum_b_with_dyn,
        x_U,
        x_L,
        norm,
        A_out,
        dt,
    ):
        if (
            A_out[0][1] == 0.0
        ):  # first dimension does not have non-linear terms
            lb = self._get_concrete_bound_lpball(
                lower_A_with_dyn,
                lower_sum_b_with_dyn,
                sign=-1,
                x_U=x_U,
                x_L=x_L,
                norm=norm,
            )[0][0]
            ub = self._get_concrete_bound_lpball(
                upper_A_with_dyn,
                upper_sum_b_with_dyn,
                sign=+1,
                x_U=x_U,
                x_L=x_L,
                norm=norm,
            )[0][0]
        else:
            c_Y = -dt
            lb = self._get_concrete_bound_cvx_rlx(
                lower_A_with_dyn,
                lower_sum_b_with_dyn,
                sign=-1.0,
                x_U=x_U,
                x_L=x_L,
                norm=norm,
                c_Y=c_Y,
            )
            # lb = -5.
            ub = self._get_concrete_bound_cvx_rlx(
                upper_A_with_dyn,
                upper_sum_b_with_dyn,
                sign=+1.0,
                x_U=x_U,
                x_L=x_L,
                norm=norm,
                c_Y=c_Y,
            )
        return ub, lb

    def compute_bound_from_matrices(
        self,
        lower_A,
        lower_sum_b,
        upper_A,
        upper_sum_b,
        x_U,
        x_L,
        norm,
        A_out,
        dynamics=None,
        A_in=None,
        b_in=None,
    ):
        if dynamics is None:
            dynamics = self.dynamics

        (
            lower_A_with_dyn,
            upper_A_with_dyn,
            lower_sum_b_with_dyn,
            upper_sum_b_with_dyn,
        ) = self._add_dynamics(
            lower_A, upper_A, lower_sum_b, upper_sum_b, A_out, dynamics
        )

        if dynamics.__class__.__name__ == "Duffing":
            ub, lb = self.nonlinear_compute_bound_from_matrices(
                lower_A_with_dyn,
                lower_sum_b_with_dyn,
                upper_A_with_dyn,
                upper_sum_b_with_dyn,
                x_U,
                x_L,
                norm,
                A_out,
                dynamics.dt,
            )
            return ub, lb

        if (A_in is None or b_in is None) and self.try_to_use_closed_form:
            # Can solve in closed-form using lq-norm
            lb = self._get_concrete_bound_lpball(
                lower_A_with_dyn,
                lower_sum_b_with_dyn,
                sign=-1,
                x_U=x_U,
                x_L=x_L,
                norm=norm,
            )[0][0]
            ub = self._get_concrete_bound_lpball(
                upper_A_with_dyn,
                upper_sum_b_with_dyn,
                sign=+1,
                x_U=x_U,
                x_L=x_L,
                norm=norm,
            )[0][0]
        elif A_in is None or b_in is None:
            # Use linear program even though you could solve in closed form
            lb = self._get_concrete_bound_linearprog(
                lower_A_with_dyn,
                lower_sum_b_with_dyn,
                sign=-1,
                x_U=x_U,
                x_L=x_L,
            )
            ub = self._get_concrete_bound_linearprog(
                upper_A_with_dyn,
                upper_sum_b_with_dyn,
                sign=+1,
                x_U=x_U,
                x_L=x_L,
            )
        else:
            # Use linear program because initial set is a polytope
            # ==> can't solve in closed-form.
            lb = None
            ub = self._get_concrete_bound_linearprog(
                upper_A_with_dyn, upper_sum_b_with_dyn, A_in, b_in
            )

        ub, lb = self._check_if_bnds_exist(ub=ub, lb=lb, x_U=x_U, x_L=x_L)
        return ub, lb

    def _get_concrete_bound_linearprog(
        self, A, sum_b, A_in=None, b_in=None, x_U=None, x_L=None, sign=1
    ):
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug("Final A: %s", A.size())

        c = A.data.numpy().squeeze()
        n = c.shape[0]

        x = cp.Variable(n)
        cost = c.T @ x

        # Input set constraints
        constraints = []
        if A_in is not None and b_in is not None:
            constraints += [A_in @ x <= b_in]
        else:
            constraints += [
                x <= x_U.data.numpy().squeeze(0),
                x >= x_L.data.numpy().squeeze(0),
            ]

        if sign == 1:
            objective = cp.Maximize(cost)
        elif sign == -1:
            objective = cp.Minimize(cost)

        prob = cp.Problem(objective, constraints)
        prob.solve()
        bound = prob.value

        bound = bound + sum_b
        return bound

    # sign = +1: upper bound, sign = -1: lower bound
    def _get_concrete_bound_lpball(
        self, A, sum_b, x_U=None, x_L=None, norm=np.inf, sign=-1
    ):
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug("Final A: %s", A.size())
        if norm == np.inf:
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            logger.debug("A_0 shape: %s", A.size())
            logger.debug("sum_b shape: %s", sum_b.size())
            # we only need the lower bound

            # First 2 terms in Eq. 20 of: https://arxiv.org/pdf/1811.00866.pdf
            # eps*q_norm(Lamb) + Lamb*x0 <==> x0=center, eps=diff, Lamb=A
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            logger.debug("bound shape: %s", bound.size())
        else:
            print("Haven't handled non-linf norm yet.")
            raise NotImplementedError
            # x = x_U.view(x_U.size(0), -1, 1)
            # dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            # deviation = A.norm(dual_norm, -1) * eps
            # bound = A.bmm(x) + sign * deviation.unsqueeze(-1)

        # Add big summation term from Eq. 20 of:
        # https://arxiv.org/pdf/1811.00866.pdf
        if sum_b.ndim == 3:
            sum_b = sum_b.squeeze(-1)
        bound = bound.squeeze(-1) + sum_b

        return bound.data.numpy()

    def _get_concrete_bound_cvx_rlx(
        self, A, sum_b, x_U=None, x_L=None, norm=np.inf, sign=-1, c_Y=-0.05
    ):
        # self, A, sum_b, x_U = None, x_L = None, norm = np.inf, sign = -1
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug("Final A: %s", A.size())

        c = A.data.numpy().squeeze()
        n = c.shape[0]

        x = cp.Variable(n)
        Y = cp.Variable((n + 2, n + 2), symmetric=True)
        # The operator >> denotes matrix inequality.

        cost = sign * (c.T @ x + c_Y * Y[2][0])
        # constraints = [A_in @ x <= b_in]
        x_1_bound = np.array(
            [x_L.data.numpy().squeeze(0)[0], x_U.data.numpy().squeeze(0)[0]]
        )
        constraints = []
        constraints += [
            x <= x_U.data.numpy().squeeze(0),
            x >= x_L.data.numpy().squeeze(0),
        ]

        constraints += [Y[0][0] == 1]
        constraints += [Y[2][0] == Y[1][3]]
        constraints += [Y[3][0] == Y[1][1]]
        constraints += [Y >> 0]

        constraints += [x[0] == Y[1][0]]

        constraints += [Y[2][0] >= np.power(x_1_bound[0], 3)]
        constraints += [Y[2][0] <= np.power(x_1_bound[1], 3)]
        constraints += [Y[3][0] >= 0]
        constraints += [Y[3][0] <= np.square(np.max(np.abs(x_1_bound)))]

        objective = cp.Maximize(cost)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        bound = sign * prob.value

        bound = bound + sum_b
        return bound


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import model_from_json

    from crown_ibp.conversions.keras2torch import keras2torch

    # load json and create model
    json_file = open("/Users/mfe/Downloads/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/Users/mfe/Downloads/model.h5")
    print("Loaded model from disk")

    torch_model = keras2torch(model, "torch_model")

    ###########################
    # To get NN nominal prediction:
    print("---")
    print("Example of a simple forward pass for a single point input.")
    x = [2.5, 0.2]
    out = torch_model.forward(torch.Tensor([[x]]))
    print("For x={}, NN output={}.".format(x, out))
    print("---")
    #
    ###########################

    ###########################
    # To get NN output bounds:
    print("---")
    print("Example of bounding the NN output associated with an input set.")
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
    x0_min, x0_max, x1_min, x1_max = [2.5, 3.0, -0.25, 0.25]

    # Evaluate CROWN bounds
    out_max_crown, _, out_min_crown, _ = torch_model_.full_backward_range(
        norm=np.inf,
        x_U=torch.Tensor([[x0_max, x1_max]]),
        x_L=torch.Tensor([[x0_min, x1_min]]),
        upper=True,
        lower=True,
        C=torch.Tensor([[[1]]]),
    )

    # Sample a grid of pts from the input set, to get exact NN output polytope
    x0 = np.linspace(x0_min, x0_max, num=10)
    x1 = np.linspace(x1_min, x1_max, num=10)
    xx, yy = np.meshgrid(x0, x1)
    pts = np.reshape(np.dstack([xx, yy]), (-1, 2))
    sampled_outputs = torch_model.forward(torch.Tensor(pts))

    # Print and compare the two bounds numerically
    sampled_output_min = np.min(sampled_outputs.data.numpy())
    sampled_output_max = np.max(sampled_outputs.data.numpy())
    crown_min = out_min_crown.data.numpy()[0, 0]
    crown_max = out_max_crown.data.numpy()[0, 0]
    print(
        "The sampled outputs lie between: [{},{}]".format(
            sampled_output_min, sampled_output_max
        )
    )
    print("CROWN bounds are: [{},{}]".format(crown_min, crown_max))
    conservatism_above = crown_max - sampled_output_max
    conservatism_below = sampled_output_min - crown_min
    print(
        "Over-conservatism: [{},{}]".format(
            conservatism_below, conservatism_above
        )
    )
    print(
        "^ These should both be positive! {}".format(
            "They are :)"
            if conservatism_above >= 0 and conservatism_below >= 0
            else "*** THEY AREN'T ***"
        )
    )

    # Plot vertical lines for CROWN bounds, x's for sampled outputs
    plt.axvline(
        out_min_crown.data.numpy()[0, 0], ls="--", label="CROWN Bounds"
    )
    plt.axvline(out_max_crown.data.numpy()[0, 0], ls="--")
    plt.scatter(
        sampled_outputs.data.numpy(),
        np.zeros(pts.shape[0]),
        marker="x",
        label="Samples",
    )

    plt.legend()
    print("Showing plot...")
    plt.show()
    print("---")

    #
    ###########################

    ###########################
    # To get closed loop system bounds (using dynamics):
    # torch_model__ = BoundClosedLoopController.convert(
    #     torch_model,
    #     {"same-slope": True},
    #     A_dyn=torch.Tensor([[[1.0, 1.0], [0.0, 1.0]]]),
    #     b_dyn=torch.Tensor([[[0.5], [1.0]]]),
    #     c_dyn=[],
    # )
    # x0_min, x0_max, x1_min, x1_max = [2.5, 3.0, -0.25, 0.25]
    # x0_t1_max, _, x0_t1_min, _ = torch_model__.full_backward_range(
    #     norm=np.inf,
    #     x_U=torch.Tensor([[x0_max, x1_max]]),
    #     x_L=torch.Tensor([[x0_min, x1_min]]),
    #     upper=True,
    #     lower=True,
    #     C=torch.Tensor([[[1]]]),
    #     A_out=torch.Tensor([[-1, 0]]),
    #     A_in=np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]),
    #     b_in=np.array([-2.5, 3.0, 0.25, 0.25]),
    # )
    # print("x0:", x0_t1_min.data.numpy()[0, 0], x0_t1_max.data.numpy()[0, 0])

    #
    ###########################
