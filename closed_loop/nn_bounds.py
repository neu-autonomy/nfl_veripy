import torch
import numpy as np
from torch.nn import Sequential, Conv2d, Linear, ReLU
from crown_ibp.bound_layers import BoundSequential, BoundLinear, BoundReLU
from crown_ibp.model_defs import Flatten
import cvxpy as cp

import logging
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BoundClosedLoopController(BoundSequential):
    def __init__(self, A_dyn, b_dyn, c_dyn, layers):
        super(BoundClosedLoopController, self).__init__(*layers)
        self.A_dyn = A_dyn
        self.b_dyn = b_dyn
        self.c_dyn = c_dyn

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model, bound_opts=None, A_dyn=None, b_dyn=None, c_dyn=None):
        layers = BoundClosedLoopController.sequential_model_to_layers(sequential_model, bound_opts=bound_opts)
        b = BoundClosedLoopController(A_dyn=A_dyn, b_dyn=b_dyn, c_dyn=c_dyn, layers=layers)
        return b

    def _add_dynamics(self, lower_A, upper_A, lower_sum_b, upper_sum_b):

        '''

        Original:
        Lambda = upper_A
        Omega = lower_A
        Delta = upper_sum_b
        Theta = lower_sum_b

        No Flip:
        Upsilon = Lambda = upper_A
        Psi = Delta = upper_sum_b
        Xi = Omega = lower_A
        Gamma = Theta = lower_sum_b

        Flip:
        Upsilon = Omega = lower_A
        Psi = Theta = lower_sum_b
        Xi = Lambda = upper_A
        Gamma = Delta = upper_sum_b

        '''

        flip = torch.matmul(self.A_out, self.b_dyn) < 0
        if flip:
            # print("flip")
            upsilon = lower_A
            psi = lower_sum_b
            xi = upper_A
            gamma = upper_sum_b
        else:
            # print('dont flip')
            upsilon = upper_A
            psi = upper_sum_b
            xi = lower_A
            gamma = lower_sum_b

        lower_A_with_dyn = torch.matmul(self.A_out, self.A_dyn+self.b_dyn.bmm(xi))
        upper_A_with_dyn = torch.matmul(self.A_out, self.A_dyn+self.b_dyn.bmm(upsilon))
        lower_sum_b_with_dyn = torch.matmul(self.A_out, self.b_dyn * gamma)
        upper_sum_b_with_dyn = torch.matmul(self.A_out, self.b_dyn * psi)
        return lower_A_with_dyn, upper_A_with_dyn, lower_sum_b_with_dyn, upper_sum_b_with_dyn

    def full_backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=True, lower=True, A_out=None, A_in=None, b_in=None, closed_loop=True, u_limits=None):
        self.A_out = A_out # output constraints (facet of polytope)
        self.A_in = A_in # input polytope constraints (alternative to x \in eps_ball)
        self.b_in = b_in # input polytope constraints (alternative to x \in eps_ball)
        self.closed_loop = closed_loop
        self.u_limits = u_limits
        return super().full_backward_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C, upper=upper, lower=lower)

    ## High level function, will be called outside
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=False, lower=True, modules=None, return_matrices=False):
        if C.size() != (1,1,1):
            # TODO: Find better mechanism to check C
            # Don't consider dynamics yet, just call parent method
            # This gets called the first passes thru the network (not the final though)
            return super().backward_range(norm=norm, x_U=x_U, x_L=x_L, eps=eps, C=C, upper=upper, lower=lower, modules=modules)
        else:
            lower_A, upper_A, lower_sum_b, upper_sum_b = self._prop_from_last_layer(C=C, x_U=x_U, modules=modules, upper=upper, lower=lower)
            ###########
            # MAJOR HACK to extract lambda matrices from calculations
            if self.closed_loop == False:
                return lower_A, upper_A, lower_sum_b, upper_sum_b
            ##########

            if self.u_limits is not None:
                raise NotImplementedError
                # lb = self._get_concrete_bound3(lower_A, upper_A, lower_sum_b, upper_sum_b, sign = -1)
                lb = -5.
                ub = self._get_concrete_bound3(lower_A, upper_A, lower_sum_b, upper_sum_b)
            else:
                # lower_A_with_dyn, upper_A_with_dyn, lower_sum_b_with_dyn, upper_sum_b_with_dyn = lower_A, upper_A, lower_sum_b, upper_sum_b 
                lower_A_with_dyn, upper_A_with_dyn, lower_sum_b_with_dyn, upper_sum_b_with_dyn = self._add_dynamics(lower_A, upper_A, lower_sum_b, upper_sum_b)
            
                if self.A_in is None or self.b_in is None:
                    lb = self._get_concrete_bound(lower_A_with_dyn, lower_sum_b_with_dyn, sign = -1, x_U=x_U, x_L=x_L, norm=norm)
                    ub = self._get_concrete_bound(upper_A_with_dyn, upper_sum_b_with_dyn, sign = +1, x_U=x_U, x_L=x_L, norm=norm)
                else:
                    lb = self._get_concrete_bound2(lower_A_with_dyn, lower_sum_b_with_dyn, sign = -1, x_U=x_U, x_L=x_L, norm=norm)
                    ub = self._get_concrete_bound2(upper_A_with_dyn, upper_sum_b_with_dyn, sign = +1, x_U=x_U, x_L=x_L, norm=norm)

            ub, lb = self._check_if_bnds_exist(ub=ub, lb=lb, x_U=x_U, x_L=x_L)
            return ub, upper_sum_b, lb, lower_sum_b

    # sign = +1: upper bound, sign = -1: lower bound
    def _get_concrete_bound2(self, A, sum_b, x_U=None, x_L=None, norm=np.inf, sign = -1):
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug('Final A: %s', A.size())

        A_constr = self.A_in
        b = self.b_in
        c = A.data.numpy().squeeze()
        n = c.shape[0]

        x = cp.Variable(n)
        cost = c.T@x
        constraints = [A_constr @ x <= b]
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
    def _get_concrete_bound3(self, lower_A, upper_A, lower_sum_b, upper_sum_b):

        u_min, u_max = self.u_limits

        num_inputs = 1
        num_states = 2
        x = cp.Variable(num_states, name='x')

        constraints = []
        constraints += [self.A_in @ x <= self.b_in]

        upper_A_np = upper_A.data.numpy().squeeze()
        lower_A_np = lower_A.data.numpy().squeeze()
        upper_sum_b_np = upper_sum_b.data.numpy().squeeze()
        lower_sum_b_np = lower_sum_b.data.numpy().squeeze()

        A_dyn_np = self.A_dyn.data.numpy().squeeze()
        b_dyn_np = self.b_dyn.data.numpy().squeeze()
        A_out_np = self.A_out.data.numpy().squeeze()

        pi_l = lower_A_np@x+lower_sum_b_np
        pi_u = upper_A_np@x+upper_sum_b_np

        state_cost = A_out_np@(A_dyn_np@x)

        if np.dot(A_out_np, b_dyn_np) >= 0:
            
            u = cp.minimum(u_max, pi_u)
            u2 = u_min
        else:
            u = cp.maximum(u_min, pi_l)
            u2 = u_max
        u_cost = (A_out_np@b_dyn_np)*u
        u2_cost = (A_out_np@b_dyn_np)*u2
        cost = state_cost + u_cost
        cost2 = state_cost + u2_cost

        objective = cp.Maximize(cost)

        # Solve problem respecting one bound on u
        prob = cp.Problem(objective, constraints)
        prob.solve()
        bound = prob.value

        # Solve problem respecting other bound on u
        # (if pi_u or pi_l exceeds other bound everywhere)
        objective = cp.Maximize(cost2)
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.value > bound:
            bound = prob.value

        return bound

    # sign = +1: upper bound, sign = -1: lower bound
    def _get_concrete_bound(self, A, sum_b, x_U=None, x_L=None, norm=np.inf, sign = -1):
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        # A has shape (batch, specification_size, flattened_input_size)
        logger.debug('Final A: %s', A.size())
        if norm == np.inf:
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            logger.debug('A_0 shape: %s', A.size())
            logger.debug('sum_b shape: %s', sum_b.size())
            # we only need the lower bound

            # First 2 terms in Eq. 20 of: https://arxiv.org/pdf/1811.00866.pdf
            # eps*q_norm(Lamb) + Lamb*x0 <==> x0=center, eps=diff, Lamb=A
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            logger.debug('bound shape: %s', bound.size())
        else:
            x = x_U.view(x_U.size(0), -1, 1)
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            deviation = A.norm(dual_norm, -1) * eps
            bound = A.bmm(x) + sign * deviation.unsqueeze(-1)

        # Add big summation term from Eq. 20 of: https://arxiv.org/pdf/1811.00866.pdf
        if sum_b.ndim == 3:
            sum_b = sum_b.squeeze(-1)
        bound = bound.squeeze(-1) + sum_b

        return bound

if __name__ == '__main__':
    from keras.models import model_from_json
    from crown_ibp.conversions.keras2torch import keras2torch, get_keras_model
    import matplotlib.pyplot as plt
    import cvxpy as cp

    # load json and create model
    json_file = open('/Users/mfe/Downloads/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("/Users/mfe/Downloads/model.h5")
    print("Loaded model from disk")

    torch_model = keras2torch(model, "torch_model")
    
    ###########################
    # To get NN nominal prediction:
    print('---')
    print("Example of a simple forward pass for a single point input.")
    x = [2.5, 0.2]
    out = torch_model.forward(torch.Tensor([[x]]))
    print("For x={}, NN output={}.".format(x,out))
    print('---')
    #
    ###########################


    ###########################
    # To get NN output bounds:
    print('---')
    print("Example of bounding the NN output associated with an input set.")
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
    x0_min, x0_max, x1_min, x1_max = [2.5, 3.0, -0.25, 0.25]

    # Evaluate CROWN bounds
    out_max_crown, _, out_min_crown, _ = torch_model_.full_backward_range(norm=np.inf,
                                x_U=torch.Tensor([[x0_max, x1_max]]),
                                x_L=torch.Tensor([[x0_min, x1_min]]),
                                upper=True, lower=True,
                                C=torch.Tensor([[[1]]]),
                                )

    # Sample a grid of pts from the input set, to get exact NN output polytope
    x0 = np.linspace(x0_min, x0_max, num=10)
    x1 = np.linspace(x1_min, x1_max, num=10)
    xx,yy = np.meshgrid(x0, x1)
    pts = np.reshape(np.dstack([xx,yy]), (-1,2))
    sampled_outputs = torch_model.forward(torch.Tensor(pts))

    # Print and compare the two bounds numerically    
    sampled_output_min = np.min(sampled_outputs.data.numpy())
    sampled_output_max = np.max(sampled_outputs.data.numpy())
    crown_min = out_min_crown.data.numpy()[0,0]
    crown_max = out_max_crown.data.numpy()[0,0]
    print("The sampled outputs lie between: [{},{}]".format(
        sampled_output_min, sampled_output_max))
    print("CROWN bounds are: [{},{}]".format(
        crown_min, crown_max))
    conservatism_above = crown_max - sampled_output_max
    conservatism_below = sampled_output_min - crown_min
    print("Over-conservatism: [{},{}]".format(
        conservatism_below, conservatism_above))
    print("^ These should both be positive! {}".format(
        "They are :)" if conservatism_above>=0 and conservatism_below>=0 else "*** THEY AREN'T ***"))

    # Plot vertical lines for CROWN bounds, x's for sampled outputs
    plt.axvline(out_min_crown.data.numpy()[0,0], ls='--', label='CROWN Bounds')
    plt.axvline(out_max_crown.data.numpy()[0,0], ls='--')
    plt.scatter(sampled_outputs.data.numpy(), np.zeros(pts.shape[0]), marker='x', label="Samples")

    plt.legend()
    print("Showing plot...")
    plt.show()
    print('---')

    #
    ###########################


    ###########################
    # # To get closed loop system bounds (using dynamics):
    # torch_model__ = BoundClosedLoopController.convert(torch_model, {"same-slope": True},
    #     A_dyn=torch.Tensor([[[1., 1.], [0., 1.]]]), b_dyn=torch.Tensor([[[0.5], [1.0]]]), c_dyn=[])
    # x0_min, x0_max, x1_min, x1_max = [2.5, 3.0, -0.25, 0.25]
    # x0_t1_max, _, x0_t1_min, _ = torch_model__.full_backward_range(norm=np.inf,
    #                             x_U=torch.Tensor([[x0_max, x1_max]]),
    #                             x_L=torch.Tensor([[x0_min, x1_min]]),
    #                             upper=True, lower=True, C=torch.Tensor([[[1]]]),
    #                             A_out=torch.Tensor([[-1,0]]),
    #                             A_in=np.array([[-1,  0],
    #                                         [ 1,  0],
    #                                         [ 0, -1],
    #                                         [ 0,  1]]),
    #                             b_in=np.array([-2.5 ,  3.  ,  0.25,  0.25]))
    # print("x0:", x0_t1_min.data.numpy()[0,0], x0_t1_max.data.numpy()[0,0])

    #
    ###########################
