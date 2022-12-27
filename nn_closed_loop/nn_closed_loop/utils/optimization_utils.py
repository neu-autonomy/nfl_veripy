import numpy as np
import cvxpy as cp
import nn_closed_loop.constraints as constraints


def optimize_over_all_states(xt, constrs):
    num_states = xt.shape[0]
    obj_facets = np.vstack([np.eye(num_states), -np.eye(num_states)])
    obj_facets_i = cp.Parameter(num_states)
    obj = obj_facets_i@xt
    prob = cp.Problem(cp.Maximize(obj), constrs)
    b = np.empty((2*num_states,))
    num_facets = obj_facets.shape[0]
    for i in range(num_facets):
        obj_facets_i.value = obj_facets[i, :]
        prob.solve()
        b[i] = prob.value
    return b, prob.status


def optimization_results_to_backprojection_set(status, b, backreachable_set):

    num_states = b.shape[0] // 2
    ranges = backreachable_set.range
    ranges[:, 0] = np.maximum(backreachable_set.range[:, 0], -b[num_states:])
    ranges[:, 1] = np.minimum(backreachable_set.range[:, 1], b[:num_states])
    backprojection_set = constraints.LpConstraint(range=ranges)

    return backprojection_set


