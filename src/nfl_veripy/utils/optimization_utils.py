import cvxpy as cp
import nfl_veripy.constraints as constraints
import numpy as np


def optimize_over_all_states(xt, constrs, facet_inds_to_optimize=None):
    num_states = xt.shape[0]
    obj_facets = np.vstack([np.eye(num_states), -np.eye(num_states)])
    obj_facets_i = cp.Parameter(num_states)
    obj = obj_facets_i @ xt
    prob = cp.Problem(cp.Maximize(obj), constrs)
    b = np.hstack(
        [np.inf * np.ones((num_states,)), -np.inf * np.ones((num_states,))]
    )
    # b = np.empty((2*num_states,))
    if facet_inds_to_optimize is None:
        num_facets = obj_facets.shape[0]
        facet_inds_to_optimize = range(num_facets)
    for i in facet_inds_to_optimize:
        obj_facets_i.value = obj_facets[i, :]
        prob.solve()
        b[i] = prob.value
    return b, prob.status


def optimization_results_to_backprojection_set(status, b, backreachable_set):
    if status == "infeasible" or status == "optimal_inaccurate":
        return None

    num_states = b.shape[0] // 2
    ranges = np.empty_like(backreachable_set.range)
    ranges[:, 0] = np.maximum(backreachable_set.range[:, 0], -b[num_states:])
    ranges[:, 1] = np.minimum(backreachable_set.range[:, 1], b[:num_states])
    backprojection_set = constraints.LpConstraint(range=ranges)

    return backprojection_set
