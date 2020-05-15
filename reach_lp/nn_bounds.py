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
    def __init__(self, *args):
        super(BoundClosedLoopController, self).__init__(*args) 

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(sequential_model, bound_opts=None, A_dyn=None, b_dyn=None, c_dyn=None):
        # TODO: properly inherit this from BoundSequential
        # b = BoundSequential.convert(sequential_model, bound_opts=bound_opts)
        layers = []
        if isinstance(sequential_model, Sequential):
            seq_model = sequential_model
        else:
            seq_model = sequential_model.module
        for l in seq_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l, bound_opts))
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l, bound_opts))
            if isinstance(l, ReLU):
                layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten(bound_opts))
        b = BoundClosedLoopController(*layers)
        b.A_dyn = A_dyn
        b.b_dyn = b_dyn
        b.c_dyn = c_dyn
        return b

    def _add_dynamics(self, lower_A, upper_A, lower_sum_b, upper_sum_b):
        # return lower_A, upper_A, lower_sum_b, upper_sum_b
        # print("lower_A:", lower_A)
        # print("self.b_dyn:", self.b_dyn)
        # print('A_0 shape: %s', lower_A.size())
        # print('self.b_dyn shape: %s', self.b_dyn.size())

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
    def backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=False, lower=True, modules=None):
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
                lower_A_with_dyn, upper_A_with_dyn, lower_sum_b_with_dyn, upper_sum_b_with_dyn = self._add_dynamics(lower_A, upper_A, lower_sum_b, upper_sum_b)
                lb = self._get_concrete_bound3(lower_A, upper_A, lower_sum_b, upper_sum_b, lower_A_with_dyn, lower_sum_b_with_dyn, sign = -1, x_U=x_U, x_L=x_L, norm=norm)
                ub = self._get_concrete_bound3(lower_A, upper_A, lower_sum_b, upper_sum_b, upper_A_with_dyn, upper_sum_b_with_dyn, sign = +1, x_U=x_U, x_L=x_L, norm=norm)
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

        # print(self.A_in.shape)
        # print(self.b_in.shape)
        # print(A.shape)

        u_min = -1; u_max = 1

        A_constr = self.A_in
        b = self.b_in
        c = A.data.numpy().squeeze()
        n = c.shape[0]

        # print("A_constr:", A_constr)
        # print("b:", b)
        # print("c:", c)
        # print("sign:", sign)

        # print(c.shape)
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

        # print("bound:", bound)
        # print("sum_b:", sum_b)
        bound = bound + sum_b
        return bound

    # sign = +1: upper bound, sign = -1: lower bound
    def _get_concrete_bound3(self, lower_A, upper_A, lower_sum_b, upper_sum_b, A, sum_b, x_U=None, x_L=None, norm=np.inf, sign = -1):
        if A is None:
            return None
        A = A.view(A.size(0), A.size(1), -1)
        logger.debug('Final A: %s', A.size())

        A_constr = self.A_in
        b = self.b_in
        c = A.data.numpy().squeeze()
        n = c.shape[0]

        u_min, u_max = self.u_limits

        num_inputs = 1
        u = cp.Variable(num_inputs, name='u')
        x = cp.Variable(n, name='x')

        constraints = []

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

        upper_A_np = upper_A.data.numpy().squeeze()
        lower_A_np = lower_A.data.numpy().squeeze()
        upper_sum_b_np = upper_sum_b.data.numpy().squeeze()
        lower_sum_b_np = lower_sum_b.data.numpy().squeeze()

        upsilon_np = upsilon.data.numpy().squeeze()
        psi_np = psi.data.numpy().squeeze()
        xi_np = xi.data.numpy().squeeze()
        gamma_np = gamma.data.numpy().squeeze()

        # print(upper_A_np.shape)
        # print(x.shape)
        # print(upper_sum_b_np.shape)
        # print(u.shape)

        # print((upper_A_np@x).shape)

        # flip = torch.matmul(self.A_out, self.b_dyn) < 0
        # if flip:
        #     constraints += [u <= lower_A_np*x+lower_sum_b_np]
        #     constraints += [u >= upper_A_np*x+upper_sum_b_np]
        # else:
        #     constraints += [u <= upper_A_np*x+upper_sum_b_np]
        #     constraints += [u >= lower_A_np*x+lower_sum_b_np]

        # np.array([upper_A_np@x+upper_sum_b_np, 0.01])

        pi_u = upsilon_np@x+psi_np
        pi_l = xi_np@x+gamma_np
        # pi_u = upper_A_np@x+upper_sum_b_np
        # pi_l = lower_A_np@x+lower_sum_b_np

        # u_upper = cp.Variable(2)
        # u_lower = cp.Variable(2)
        # constraints += [u_upper[0] == pi_u]
        # constraints += [u_upper[1] == u_max]
        # constraints += [u_lower[0] == pi_l]
        # constraints += [u_lower[1] == u_min]

        # if flip:
        #     constraints += [u >= pi_l]

        # else:
        #     constraints += [u <= pi_u]

        if sign == 1:
            constraints += [u <= pi_u]
            # constraints += [u <= u_max]
            # constraints += [u == pi_u]
            # constraints += [u == cp.minimum(u_max, pi_u)]
        elif sign == -1:
            constraints += [u >= pi_l]
            # constraints += [u >= u_min]
            # constraints += [u == cp.maximum(u_min, pi_l)]

        # if sign == -1:
        #     constraints += [u <= pi_u]
        #     # constraints += [u >= pi_l]

        # elif sign == 1:
        #     constraints += [u >= pi_l]
        #     # constraints += [u <= pi_u]

        # constraints += [u <= pi_u]
        # constraints += [u >= pi_l]

        # constraints += [u <= u_max]
        # constraints += [u >= u_min]

        A_dyn_np = self.A_dyn.data.numpy().squeeze()
        b_dyn_np = self.b_dyn.data.numpy().squeeze()
        A_out_np = self.A_out.data.numpy().squeeze()

        cost = A_out_np@(A_dyn_np@x+b_dyn_np@u)

        constraints += [A_constr @ x <= b]
        if sign == 1:
            objective = cp.Maximize(cost)
        elif sign == -1:
            objective = cp.Minimize(cost)

        prob = cp.Problem(objective, constraints)
        prob.solve()
        bound = prob.value

        bound = bound
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

            # print("A.shape:", A.shape)
            # print("center: ", center)
            # print("diff: ", diff)
            # print("A.bmm(center):", A.bmm(center))

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
        # print("bound.size(): {}".format(bound.size()))
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
    torch_model_ = BoundSequential.convert(torch_model, {"same-slope": True})
    torch_model__ = BoundClosedLoopController.convert(torch_model, {"same-slope": True},
        A_dyn=torch.Tensor([[[1., 1.], [0., 1.]]]), b_dyn=torch.Tensor([[[0.5], [1.0]]]), c_dyn=[])

    # x = [2.5, 0.2]
    # print("keras:", model.predict(np.expand_dims(np.array(x), axis=0)))
    # print("torch:", torch_model.forward(torch.Tensor([[x]])))
    # print("torch:", torch_model_.forward(torch.Tensor([[x]])))

    # np.random.seed(0)
    # all_eps = np.linspace(0, 1.0, num=10)
    # for eps in all_eps:
    #     for i in range(10):
    #         x0_min = np.random.uniform(2.5, 3.0)
    #         x0_max = x0_min + eps
    #         x1_min = np.random.uniform(-0.5, 0.5)
    #         x1_max = x1_min + eps

    x0_min, x0_max, x1_min, x1_max = [2.5, 3.0, -0.25, 0.25]

    # x0_t1_max, tmp, x0_t1_min, tmp2 = torch_model__.full_backward_range(norm=np.inf,
    #                             x_U=torch.Tensor([[x0_max, x1_max]]),
    #                             x_L=torch.Tensor([[x0_min, x1_min]]),
    #                             upper=True, lower=True, C=torch.Tensor([[[1]]]),
    #                             A_out=torch.Tensor([[1,0]]))
    # x0_t1_max_, tmp, x0_t1_min_, tmp2 = torch_model__.full_backward_range(norm=np.inf,
    #                             x_U=torch.Tensor([[x0_max, x1_max]]),
    #                             x_L=torch.Tensor([[x0_min, x1_min]]),
    #                             upper=True, lower=True, C=torch.Tensor([[[1]]]),
    #                             A_out=torch.Tensor([[-1,0]]))

    x0_t1_max__, tmp, x0_t1_min__, tmp2 = torch_model__.full_backward_range(norm=np.inf,
                                x_U=torch.Tensor([[x0_max, x1_max]]),
                                x_L=torch.Tensor([[x0_min, x1_min]]),
                                upper=True, lower=True, C=torch.Tensor([[[1]]]),
                                A_out=torch.Tensor([[1,0]]),
                                A_in=np.array([[-1,  0],
                                            [ 1,  0],
                                            [ 0, -1],
                                            [ 0,  1]]),
                                b_in=np.array([-2.5 ,  3.  ,  0.25,  0.25]))

    print('-------')

    x0_t1_max___, tmp, x0_t1_min___, tmp2 = torch_model__.full_backward_range(norm=np.inf,
                                x_U=torch.Tensor([[x0_max, x1_max]]),
                                x_L=torch.Tensor([[x0_min, x1_min]]),
                                upper=True, lower=True, C=torch.Tensor([[[1]]]),
                                A_out=torch.Tensor([[-1,0]]),
                                A_in=np.array([[-1,  0],
                                            [ 1,  0],
                                            [ 0, -1],
                                            [ 0,  1]]),
                                b_in=np.array([-2.5 ,  3.  ,  0.25,  0.25]))

    # x1_t1_max, tmp, x1_t1_min, tmp2 = torch_model__.full_backward_range(norm=np.inf,
    #                             x_U=torch.Tensor([[x0_max, x1_max]]),
    #                             x_L=torch.Tensor([[x0_min, x1_min]]),
    #                             upper=True, lower=True, C=torch.Tensor([[[1]]]),
    #                             A_out=torch.Tensor([[0., 1.]]))
    # print("x0:", x0_t1_min.data.numpy()[0,0], x0_t1_max.data.numpy()[0,0])
    # print("x0:", x0_t1_min_.data.numpy()[0,0], x0_t1_max_.data.numpy()[0,0])
    print("x0:", x0_t1_min__.data.numpy()[0,0], x0_t1_max__.data.numpy()[0,0])
    print('-------')
    print("x0:", x0_t1_min___.data.numpy()[0,0], x0_t1_max___.data.numpy()[0,0])
    # print("x0:", x0_t1_min__.data.numpy()[0,0], x0_t1_max__.data.numpy()[0,0])
    # print("x1:", x1_t1_min.data.numpy()[0,0], x1_t1_max.data.numpy()[0,0])




            # # print(out_min2.data.numpy())
            # u_min_crown = out_min2.data.numpy()[0,0]
            # u_max_crown = out_max2.data.numpy()[0,0]

            # x0 = np.linspace(x0_min, x0_max, num=10)
            # x1 = np.linspace(x1_min, x1_max, num=10)
            # xx,yy = np.meshgrid(x0, x1)
            # pts = np.reshape(np.dstack([xx,yy]), (-1,2))
            # sampled_outputs_keras = model.predict(pts)
            # sampled_outputs = torch_model_.forward(torch.Tensor(pts))
            # u_max_true = np.max(sampled_outputs.data.numpy())
            # u_min_true = np.min(sampled_outputs.data.numpy())
            # u_max_keras = np.max(sampled_outputs_keras)
            # u_min_keras = np.min(sampled_outputs_keras)

            # print("True: [{},{}]".format(u_min_true, u_max_true))
            # print("Kera: [{},{}]".format(u_min_keras, u_max_keras))
            # print("CROW: [{},{}]".format(u_min_crown, u_max_crown))
            # print('---')

            # if u_min_true < u_min_crown-1e-5 or u_max_true > u_max_crown+1e-5:
            #     assert(0)


    # x0_min, x0_max, x1_min, x1_max = [0.2, 0.3, 0.1, 0.2]
    # x0_min, x0_max, x1_min, x1_max = [0.79, 0.8, 1.19, 1.2]
    

    # plt.scatter(out_min.data.numpy()[0], [0], marker='x')
    # plt.scatter(out_max.data.numpy()[0], [0], marker='x')
    # plt.scatter(out_min2.data.numpy()[0], [0], marker='o')
    # plt.scatter(out_max2.data.numpy()[0], [0], marker='o')
    # plt.scatter(sampled_outputs.data.numpy(), np.zeros(pts.shape[0]), marker='x')
    # plt.show()