import numpy as np
import cvxpy as cp

def control_mpc(x0, A, b, Q, R, P, u_min, u_max, n_mpc=10, debug=False):
    # TODO: account for final state constraint using O_inf

    u = cp.Variable(n_mpc)
    x = cp.Variable((n_mpc+1, x0.shape[0]))

    cost = 0

    constrs = []
    constrs.append(x[0,:] == x0)
    step = 0
    while step < n_mpc:
        constr = x[step+1,:] == A*x[step, :] + (b*u[step])[:,0]
        constrs.append(constr)

        # Input constraints
        constrs.append(u[step] <= u_max)
        constrs.append(u[step] >= -u_max)

        # State constraints
        constrs.append(x[step+1,0] >= -5)
        constrs.append(x[step+1,0] <= 5)
        constrs.append(x[step+1,1] >= -1)
        constrs.append(x[step+1,1] <= 1)

        # Control cost
        cost += cp.square(u[step])*R

        # State stage cost
        cost += cp.quad_form(x[step,:], Q)

        step += 1

    # Terminal state constraint
    # constrs.append()

    # Terminal state cost
    cost += cp.quad_form(x[n_mpc,:], P)

    prob = cp.Problem(cp.Minimize(cost), constrs)

    prob.solve()

    if debug:
        print(x.value)
    return u.value[0]


def control_linear(x):
    # Controller
    k = np.array([-0.5, -0.5])
    return np.dot(k,x)


