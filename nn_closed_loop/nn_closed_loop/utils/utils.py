import pickle
import numpy as np
import os
import nn_closed_loop.constraints as constraints
import cvxpy as cp
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def save_dataset(xs, us, system="DoubleIntegrator", dataset_name="default"):
    path = "{}/../../datasets/{}/{}".format(dir_path, system, dataset_name)
    os.makedirs(path, exist_ok=True)
    with open(path+"/dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)


def load_dataset(system="DoubleIntegrator", dataset_name="default"):
    path = "{}/../../datasets/{}/{}".format(dir_path, system, dataset_name)
    with open(path+"/dataset.pkl", "rb") as f:
        xs, us = pickle.load(f)
    return xs, us


def range_to_polytope(state_range):
    num_states = state_range.shape[0]
    A = np.vstack([np.eye(num_states), -np.eye(num_states)])
    b = np.hstack([state_range[:, 1], -state_range[:, 0]])
    return A, b


def get_polytope_A(num):
    theta = np.linspace(0, 2 * np.pi, num=num + 1)
    A_out = np.dstack([np.cos(theta), np.sin(theta)])[0][:-1]
    return A_out


def get_next_state(xt, ut, At, bt, ct):
    return np.dot(At, xt.T) + np.dot(bt, ut.T)


def plot_polytope_facets(A, b, ls='-', show=True):
    import matplotlib.pyplot as plt
    cs = ['r','g','b','brown','tab:red', 'tab:green', 'tab:blue', 'tab:brown']
    ls = ['-', '-', '-', '-', '--', '--', '--', '--']
    num_facets = b.shape[0]
    x = np.linspace(1, 5, 2000)
    for i in range(num_facets):
        alpha = 0.2
        if A[i, 1] == 0:
            offset = -0.1*np.sign(A[i, 0])
            plt.axvline(x=b[i]/A[i, 0], ls=ls[i], c=cs[i])
            plt.fill_betweenx(y=np.linspace(-2, 2, 2000), x1=b[i]/A[i, 0], x2=offset+b[i]/A[i, 0], fc=cs[i], alpha=alpha)
        else:
            offset = -0.1*np.sign(A[i, 1])
            y = (b[i] - A[i, 0]*x)/A[i, 1]
            plt.plot(x, y, ls=ls[i], c=cs[i])
            plt.fill_between(x, y, y+offset, fc=cs[i], alpha=alpha)
    if show:
        plt.show()


def get_polytope_verts(A, b):
    import pypoman
    # vertices = pypoman.duality.compute_polytope_vertices(A, b)
    vertices = pypoman.polygon.compute_polygon_hull(A, b)
    print(vertices)


def over_approximate_constraint(constraint):

    if isinstance(constraint, constraints.LpConstraint):
        return constraint
    elif isinstance(constraint, constraints.PolytopeConstraint):

        # Note: this is a super sketchy implementation that only works under certain cases
        # specifically when all the contraints have the same A matrix

        # TODO: Add an assert
        # TODO: implement a more general version

        constraint.A = constraint.A[0]
        constraint.b = [np.max(np.array(constraint.b), axis=0)]

        return constraint


def optimize_over_all_states(num_states, xt, constrs):

    # Solve optimization problem (min and max) for each state
    # We define A and solve for b, according to:
    # --> b = min A[i, :] @ xt s.t. dynamics, NN control, target set, etc.
    A = np.vstack([np.eye(num_states), -np.eye(num_states)])
    num_facets = A.shape[0]
    A_i = cp.Parameter(num_states)
    obj = A_i@xt
    prob = cp.Problem(cp.Maximize(obj), constrs)
    b = np.empty(num_facets)
    for i in range(num_facets):
        A_i.value = A[i, :]
        prob.solve()
        b[i] = prob.value

    return b, prob.status


def optimization_results_to_backprojection_set(
    status, b, backreachable_set
):

    # It appears there are no states in that backreachable_set
    # that could lead to the target_set using that relaxed NN
    if status == 'infeasible':
        return None

    # xt_min/max_cvxpy holds the results of the optimization
    xt_max_cvxpy = b[:int(len(b)/2)]
    xt_min_cvxpy = -b[int(len(b)/2):]

    # make sure we don't return a state beyond where we relaxed the NN
    # i.e., backprojection set \subseteq backreachable_set
    xt_max = np.minimum(xt_max_cvxpy, backreachable_set.range[:, 1])
    xt_min = np.maximum(xt_min_cvxpy, backreachable_set.range[:, 0])

    # Note: Probably need to do above 2 lines slightly differently
    # if we're using a PolytopeConstraint
    # From before...This cell of the backprojection set is upper-bounded by
    # cell of the backreachable set that we used in the NN relaxation
    # ==> the polytope is the intersection (i.e., concatenation)
    # of the polytope used for relaxing the NN and the soln to the LP

    # dump those results into an LpConstraint
    backprojection_set = constraints.LpConstraint(
        range=np.vstack([xt_min, xt_max]).T,
        p=np.inf
    )

    return backprojection_set


def get_crown_matrices(propagator, state_set, num_control_inputs, sensor_noise):
    nn_input_max = torch.Tensor(np.expand_dims(state_set.range[:, 1], axis=0))
    nn_input_min = torch.Tensor(np.expand_dims(state_set.range[:, 0], axis=0))
    if sensor_noise is not None:
        # Because there might be sensor noise, the NN could see a different
        # set of states than the system is actually in
        raise NotImplementedError

    # Compute the NN output matrices (for this backreachable_set)
    C = torch.eye(num_control_inputs).unsqueeze(0)
    return CROWNMatrices(
        *propagator.network(
            method_opt=propagator.method_opt,
            norm=np.inf,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )
    )


class CROWNMatrices:
    def __init__(self, lower_A, upper_A, lower_sum_b, upper_sum_b):
        self.upper_A = upper_A
        self.lower_A = lower_A
        self.upper_sum_b = upper_sum_b
        self.lower_sum_b = lower_sum_b

    def to_numpy(self):
        return self.upper_A_numpy, self.lower_A_numpy, self.upper_sum_b_numpy, self.lower_sum_b_numpy

    @property
    def lower_A_numpy(self):
        return self.lower_A.detach().numpy()[0]

    @property
    def upper_A_numpy(self):
        return self.upper_A.detach().numpy()[0]

    @property
    def lower_sum_b_numpy(self):
        return self.lower_sum_b.detach().numpy()[0]

    @property
    def upper_sum_b_numpy(self):
        return self.upper_sum_b.detach().numpy()[0]


if __name__ == "__main__":
    # save_dataset(xs, us)
    # xs, us = load_dataset()

    import matplotlib.pyplot as plt

    A = np.array([
              [1, 1],
              [0, 1],
              [-1, -1],
              [0, -1]
    ])
    b = np.array([2.8, 0.41, -2.7, -0.39])

    A2 = np.array([
                  [1, 1],
                  [0, 1],
                  [-0.97300157, -0.95230697],
                  [0.05399687, -0.90461393]
    ])
    b2 = np.array([2.74723146, 0.30446292, -2.64723146, -0.28446292])

    # get_polytope_verts(A, b)
    plot_polytope_facets(A, b)
    # get_polytope_verts(A2, b2)
    plot_polytope_facets(A2, b2, ls='--')
    plt.show()
