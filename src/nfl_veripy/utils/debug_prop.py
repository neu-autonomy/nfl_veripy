from copy import deepcopy

import cvxpy as cp
import nfl_veripy.constraints as constraints
import numpy as np
from torch import tensor


def new():
    ### 2-step debugging
    backreachable_set = constraints.LpConstraint(
        range=np.array([[2.0, 2.625], [1.625, 2.25]])
    )
    target_sets = constraints.MultiTimestepLpConstraint(
        range=np.array(
            [[[4.5, 5.0], [-0.25, 0.25]], [[3.75, 4.75], [0.75, 1.25]]]
        )
    )
    all_crown_matrices = [
        {
            "upper_A": tensor([[[-0.24625215, 0.11958137]]]),
            "lower_A": tensor([[[0.0, 0.0]]]),
            "upper_sum_b": tensor([[-0.45236018]]),
            "lower_sum_b": tensor([[-1.0]]),
        },
        {
            "upper_A": tensor([[[0.0, 0.0]]]),
            "lower_A": tensor([[[0.0, 0.0]]]),
            "upper_sum_b": tensor([[-1.0]]),
            "lower_sum_b": tensor([[-1.0]]),
        },
    ]
    num_steps = 2

    ### 3-step debugging
    # backreachable_set = constraints.LpConstraint(range=np.array([[0.028, 0.8], [1.28, 1.94]]))
    # target_sets = constraints.MultiTimestepLpConstraint(range=np.array([[[4.5, 5.], [-0.25, 0.25]], [[3.75, 4.75], [0.75, 1.25]], [[2., 3.5], [1.63, 2.25]]]))
    # all_crown_matrices = [
    #     {'upper_A': tensor([[[-0.6481,  0.4175]]]), 'lower_A': tensor([[[-0.6696,  0.4235]]]), 'upper_sum_b': tensor([[-0.3419]]), 'lower_sum_b': tensor([[-0.3529]])},
    #     {'upper_A': tensor([[[-0.1707,  0.0388]]]), 'lower_A': tensor([[[0., 0.]]]), 'upper_sum_b': tensor([[-0.4218]]), 'lower_sum_b': tensor([[-1.]])},
    #     {'upper_A': tensor([[[0., 0.]]]), 'lower_A': tensor([[[0., 0.]]]), 'upper_sum_b': tensor([[-1.]]), 'lower_sum_b': tensor([[-1.]])},
    # ]
    # num_steps = 3

    dt = 1
    At = np.array([[1, dt], [0, 1]])
    bt = np.array([[0.5 * dt * dt], [dt]])
    ct = np.array([0.0, 0.0]).T

    num_states = 2
    num_control_inputs = 1
    num_steps = target_sets.get_t_max()

    xt = cp.Variable((num_states, num_steps + 1))
    ut = cp.Variable((num_control_inputs, num_steps))
    constrs = []

    # Constraints to ensure that xt stays within the backreachable set
    A_backreach, b_backreach = backreachable_set.get_polytope()
    constrs += [A_backreach @ xt[:, 0] <= b_backreach]
    # constrs += [xt[:, 0] >= backreachable_set.range[:, 0]]
    # constrs += [xt[:, 0] <= backreachable_set.range[:, 1]]

    # if IMPORTANT_RUN:
    #     print('------------')
    #     print(A_backreach)
    #     print("@ xt[:, 0] <=")
    #     print(b_backreach)
    #     print('--')

    print("------------")
    print("***** BR *****")
    print("xt[:, 0] >=")
    print(backreachable_set.range[:, 0])
    print("--")
    print("xt[:, 0] <=")
    print(backreachable_set.range[:, 1])
    print("--")

    # Each ut must not exceed CROWN bounds
    for t in range(num_steps):
        # if t == 0:
        #     set_t = backreachable_set
        # else:
        #     set_t = target_sets.get_constraint_at_time_index(-t)

        # if set_t.crown_matrices is None:
        # TODO: these matrices have probably been computed already elsewhere, grab those instead of re-calculating
        # set_t.crown_matrices = get_crown_matrices(
        #     self,
        #     set_t,
        #     num_inputs,
        #     None
        # )
        crown_matrices = all_crown_matrices[t]

        lower_A = deepcopy(crown_matrices["lower_A"].detach().numpy()[0])
        lower_sum_b = deepcopy(
            crown_matrices["lower_sum_b"].detach().numpy()[0]
        )
        upper_A = deepcopy(crown_matrices["upper_A"].detach().numpy()[0])
        upper_sum_b = deepcopy(
            crown_matrices["upper_sum_b"].detach().numpy()[0]
        )

        constrs += [lower_A @ xt[:, t] + lower_sum_b <= ut[:, t]]
        constrs += [ut[:, t] <= upper_A @ xt[:, t] + upper_sum_b]

        # constrs += [ut[:, t] >= self.dynamics.u_limits[:, 0]]
        # constrs += [ut[:, t] <= self.dynamics.u_limits[:, 1]]

        print("***** CROWN *****")
        print(lower_A)
        print("@ xt[:, " + str(t) + "] + ")
        print(lower_sum_b)
        print("<= ut[:, " + str(t) + "]")
        print("--")
        print("ut[:, " + str(t) + "] <=")
        print(upper_A)
        print("@ xt[:, " + str(t) + "] + ")
        print(upper_sum_b)
        print("--")

    for t in range(num_steps):
        constrs += [At @ xt[:, t] + bt @ ut[:, t] + ct == xt[:, t + 1]]

        print(
            "self.dynamics.dynamics_step(xt[:, "
            + str(t)
            + "], ut[:, "
            + str(t)
            + "]) == xt[:, "
            + str(t + 1)
            + "]"
        )
        print("--")

    # x_t and x_{t+1} connected through system dynamics, have to ensure xt reaches the "target set" given ut
    for t in range(num_steps):
        A_set_t, b_target = target_sets.get_constraint_at_time_index(
            -t - 1
        ).get_polytope()
        constrs += [A_set_t @ xt[:, t + 1] <= b_target]

        # lo = deepcopy(target_sets.get_constraint_at_time_index(-t-1).range[:, 0])
        # up = deepcopy(target_sets.get_constraint_at_time_index(-t-1).range[:, 1])

        # constrs += [xt[:, t+1] >= lo]
        # constrs += [xt[:, t+1] <= up]

        # print("*** target set ***")
        # print("xt[:, " + str(t+1) + "] >=")
        # print(lo)
        # print('--')
        # print("xt[:, " + str(t+1) + "] <=")
        # print(up)
        # print('--')
        # print("******")

        print("*** option 2 ***")
        print(A_set_t)
        print("@ xt[:, " + str(t + 1) + "] <= ")
        print(b_target)
        print("--")
        print("******")

    facet_inds_to_optimize = None
    b, status = optimize_over_all_states(
        xt[:, 0], constrs, facet_inds_to_optimize=facet_inds_to_optimize
    )

    backprojection_set = optimization_results_to_backprojection_set(
        status, b, backreachable_set
    )

    return backprojection_set


def old():
    ## 2-step
    backreachable_set = constraints.LpConstraint(
        range=np.array([[2.0, 2.625], [1.625, 2.25]])
    )
    collected_input_constraints = [
        np.array([[4.5, 5.0], [-0.25, 0.25]]),
        np.array([[3.75, 4.75], [0.75, 1.25]]),
    ]
    all_crown_matrices = [
        {
            "upper_A": tensor([[[-0.24625215, 0.11958137]]]),
            "lower_A": tensor([[[0.0, 0.0]]]),
            "upper_sum_b": tensor([[-0.45236018]]),
            "lower_sum_b": tensor([[-1.0]]),
        },
        {
            "upper_A": tensor([[[0.0, 0.0]]]),
            "lower_A": tensor([[[0.0, 0.0]]]),
            "upper_sum_b": tensor([[-1.0]]),
            "lower_sum_b": tensor([[-1.0]]),
        },
    ]
    num_steps = 2

    ## 3-step
    # backreachable_set = constraints.LpConstraint(range=np.array([[0.028, 0.8], [1.28, 1.94]]))
    # collected_input_constraints = [
    #     np.array([[4.5, 5.], [-0.25, 0.25]]),
    #     np.array([[3.75, 4.75], [0.75, 1.25]]),
    #     np.array([[2., 3.5], [1.63, 2.25]]),
    # ]
    # all_crown_matrices = [
    #     {'upper_A': tensor([[[-0.6481,  0.4175]]]), 'lower_A': tensor([[[-0.6696,  0.4235]]]), 'upper_sum_b': tensor([[-0.3419]]), 'lower_sum_b': tensor([[-0.3529]])},
    #     {'upper_A': tensor([[[0., 0.]]]), 'lower_A': tensor([[[0., 0.]]]), 'upper_sum_b': tensor([[-1.]]), 'lower_sum_b': tensor([[-1.]])},
    #     {'upper_A': tensor([[[-0.1707,  0.0388]]]), 'lower_A': tensor([[[0., 0.]]]), 'upper_sum_b': tensor([[-0.4218]]), 'lower_sum_b': tensor([[-1.]])},
    # ]
    # num_steps = 3

    dt = 1
    At = np.array([[1, dt], [0, 1]])
    bt = np.array([[0.5 * dt * dt], [dt]])
    ct = np.array([0.0, 0.0]).T

    num_states = 2
    num_control_inputs = 1
    xt_min = backreachable_set.range[:, 0]
    xt_max = backreachable_set.range[:, 1]

    xt = cp.Variable((num_states, num_steps + 1))
    ut = cp.Variable((num_control_inputs, num_steps))
    constrs = []

    # x_{t=0} \in this partition of 0-th backreachable set
    constrs += [xt_min <= xt[:, 0]]
    constrs += [xt[:, 0] <= xt_max]

    print("-------------")
    print("*** BR ***")
    print(xt_min)
    print("<= xt[:, 0]")
    print("--")
    print("xt[:, 0] <=")
    print(xt_max)
    print("--")

    for t in range(num_steps):
        if t > 0:
            # Each xt must fall in the original backprojection
            constrs += [collected_input_constraints[-t][:, 0] <= xt[:, t]]
            constrs += [xt[:, t] <= collected_input_constraints[-t][:, 1]]

            print("*** Targets ***")
            print(collected_input_constraints[-t][:, 0])
            print("<= xt[:, " + str(t) + "]")
            print("--")
            print("xt[:, " + str(t) + "] <=")
            print(collected_input_constraints[-t][:, 1])
            print("--")

        # Gather CROWN bounds and previous BP bounds
        upper_A = all_crown_matrices[-t]["upper_A"].detach().numpy()[0]
        lower_A = all_crown_matrices[-t]["lower_A"].detach().numpy()[0]
        upper_sum_b = all_crown_matrices[-t]["upper_sum_b"].detach().numpy()[0]
        lower_sum_b = all_crown_matrices[-t]["lower_sum_b"].detach().numpy()[0]

        # u_t bounded by CROWN bounds
        constrs += [lower_A @ xt[:, t] + lower_sum_b <= ut[:, t]]
        constrs += [ut[:, t] <= upper_A @ xt[:, t] + upper_sum_b]

        print("*** CROWN ***")
        print(lower_A)
        print("@ xt[:, " + str(t) + "] +")
        print(lower_sum_b)
        print("<= ut[:, " + str(t) + "]")
        print("--")
        print("ut[:, " + str(t) + "] <=")
        print(upper_A)
        print("@ xt[:, " + str(t) + "] +")
        print(upper_sum_b)
        print("--")

        # x_t and x_{t+1} connected through system dynamics
        constrs += [At @ xt[:, t] + bt @ ut[:, t] + ct == xt[:, t + 1]]

        print(
            "self.dynamics.dynamics_step(xt[:, "
            + str(t)
            + "], ut[:, "
            + str(t)
            + "]) == xt[:, "
            + str(t + 1)
            + "]"
        )
        print("--")

    A_t_i = cp.Parameter(num_states)
    obj = A_t_i @ xt[:, 0]
    prob = cp.Problem(cp.Maximize(obj), constrs)

    # Solve optimization problem (min and max) for each state
    A_t_ = np.vstack([np.eye(num_states), -np.eye(num_states)])

    A_ = deepcopy(A_t_)
    b_ = np.empty((4,))

    for i, row in enumerate(A_t_):
        A_t_i.value = A_t_[i, :]
        prob.solve()
        b_[i] = prob.value

    return A_, b_


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


if __name__ == "__main__":
    ## 2 step. new is giving [[2.         2.625     ] [1.84601711 2.25      ]]
    # but should be giving array([ 2.625     ,  2.25      , -2.        , -1.63631528])

    bp_new = new()
    # print(bp_new.range)

    # bp_old = old()

    # print('new')
    # print(bp_new.range)
    # print('old')
    # print(bp_old)
    import pdb

    pdb.set_trace()
