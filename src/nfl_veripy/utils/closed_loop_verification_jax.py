"""Compute reachable set of neural feedback loop in Jax (Reach-LP)."""

import functools

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# from jax_verify.src.types import Nest, SpecFn, Tensor
from matplotlib.patches import Rectangle
from nfl_veripy import dynamics as nfl_dynamics
from nfl_veripy.utils.nn_jax import (
    predict_future_states,
    predict_mlp,
    predict_next_state,
    setup_nn,
)
from nfl_veripy.utils.utils import colors
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

import jax_verify
from jax_verify.src import (
    bound_propagation,
    bound_utils,
    concretization,
    synthetic_primitives,
)
from jax_verify.src.linear import backward_crown, linear_relaxations


class LinFunExtractionConcretizer(concretization.BackwardConcretizer):
    """Linear function extractor.

    Given an objective over an output, extract the corresponding linear
    function over a target node.
    The relation between the output node and the target node are obtained by
    propagating backward the `base_transform`.
    """

    def __init__(self, base_transform, target_index, obj):
        self._base_transform = base_transform
        self._target_index = target_index
        self._obj = obj

    def should_handle_as_subgraph(self, primitive):
        return self._base_transform.should_handle_as_subgraph(primitive)

    def concretize_args(self, primitive):
        return self._base_transform.concretize_args(primitive)

    def concrete_bound(self, graph, inputs, env, node_ref):
        initial_lin_expression = linear_relaxations.LinearExpression(
            self._obj, jnp.zeros(self._obj.shape[:1])
        )
        target_linfun, _ = graph.backward_propagation(
            self._base_transform,
            env,
            {node_ref: initial_lin_expression},
            [self._target_index],
        )
        return target_linfun


def backward_crown_bound_propagation_linfun(
    function,
    *bounds,
    obj=None,
):
    """Run CROWN but return linfuns rather than concretized IntervalBounds.

    Args:
      function: Function performing computation to obtain bounds for. Takes as
        only argument the network inputs.
      *bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
      obj: Tensor describing facets of function's output to bound
    Returns:
      output_bound: Bounds on the output of the function obtained by FastLin
    """

    # As we want to extract some linfuns that are in the middle of two linear
    # layers, we want to avoid the linear operator fusion.
    simplifier_composition = synthetic_primitives.simplifier_composition
    default_simplifier_without_linear = simplifier_composition(
        synthetic_primitives.activation_simplifier,
        synthetic_primitives.hoist_constant_computations,
    )

    # We are first going to obtain intermediate bounds for all the activation
    # of the network, so that the backward propagation of the extraction can be
    # done.
    crown_algorithm = concretization.BackwardConcretizingAlgorithm(
        backward_crown.backward_crown_concretizer
    )
    # BoundRetrieverAlgorithm wraps an existing algorithm and captures all of
    # the intermediate bound it generates.
    bound_retriever_algorithm = bound_utils.BoundRetrieverAlgorithm(
        crown_algorithm
    )
    bound_propagation.bound_propagation(
        bound_retriever_algorithm,
        function,
        *bounds,
        graph_simplifier=default_simplifier_without_linear,
    )
    intermediate_bounds = bound_retriever_algorithm.concrete_bounds
    # Now that we have extracted all intermediate bounds, we create a
    # FixedBoundApplier. This is a forward transform that pretends to compute
    # bounds, but actually just look them up in a dict of precomputed bounds.
    fwd_bound_applier = bound_utils.FixedBoundApplier(intermediate_bounds)

    # Let's define what node we are interested in capturing linear functions
    # for. If needed, this could be extracted and given as argument to this
    # function, or as a callback that would compute which nodes to target.
    input_indices = [(i,) for i, _ in enumerate(bounds)]
    if False:  # TODO: add proper class type
        # TODO: add proper class type
        # if isinstance(bounds[0], 'simplex_bound'):
        # In the case we're using a Simplex bound, we also added a linear
        # layer to the front of the NN. So, stop backpropagating when you reach
        # that new linear layer, and you'll get the linfun relating polytope
        # coordinates to output.
        # We're identifying it because it's the first node that is not an input
        # bound
        for node_key in intermediate_bounds:
            if node_key not in input_indices:
                target_index = node_key
                break
    else:
        # We're propagating to the first input.
        target_index = input_indices[0]

    # Create the concretizer. See the class definition above. The definition
    # of a "concretized_bound" for this one is "Obj linear function
    # reformulated as a linear function of target index".
    # Note: If there is a need to handle a network with multiple output, it
    # should be easy to extend by making obj here a dict mapping output node to
    # objective on that output node.
    extracting_concretizer = LinFunExtractionConcretizer(
        backward_crown.backward_crown_transform, target_index, obj
    )

    # BackwardAlgorithmForwardConcretization uses:
    #  - A forward transform to compute all intermediate bounds (here a bound
    #    applier that just look them up).
    #  - A backward concretizer to compute all final bounds (which we have here
    #    defined as the linear function of the target index).
    fwd_bwd_alg = concretization.BackwardAlgorithmForwardConcretization
    lin_fun_extractor_algorithm = fwd_bwd_alg(
        fwd_bound_applier, extracting_concretizer
    )
    # We get one target_linfuns per output.
    target_linfuns, _ = bound_propagation.bound_propagation(
        lin_fun_extractor_algorithm,
        function,
        *bounds,
        graph_simplifier=default_simplifier_without_linear,
    )

    return target_linfuns


def get_multi_step_reachable_sets_unrolled(
    params,
    xt_bounds: jax_verify.IntervalBound,
    dynamics: nfl_dynamics.Dynamics,
    num_steps: int,
    verif_fn=jax_verify.backward_crown_bound_propagation,
):
    """Compute all N-step reachable sets, by using
    dynamics(control(...(dynamics(control(init_state))...))
    as function to verify."""
    logits_fn = functools.partial(
        predict_future_states, params, dynamics, num_steps
    )
    xt_bounds = verif_fn(logits_fn, xt_bounds)
    return xt_bounds


def get_multi_step_reachable_set(
    params,
    xt_bounds: jax_verify.IntervalBound,
    dynamics: nfl_dynamics.Dynamics,
    num_steps: int,
    verif_fn=jax_verify.backward_crown_bound_propagation,
):
    """Compute all N-step reachable sets but just return last one."""
    return get_multi_step_reachable_sets(
        params, xt_bounds, dynamics, num_steps, verif_fn
    )[-1]


def get_multi_step_reachable_sets(
    params,
    xt_bounds: jax_verify.IntervalBound,
    dynamics: nfl_dynamics.Dynamics,
    num_steps: int,
    verif_fn=jax_verify.backward_crown_bound_propagation,
):
    """Compute all N-step reachable sets and return them as list,
    by appending dynamics to end of NN."""
    bounds = [xt_bounds]
    for _ in range(num_steps):
        xt_bounds = get_one_step_reachable_set(
            params, xt_bounds, dynamics, verif_fn=verif_fn
        )
        bounds.append(xt_bounds)
    return bounds


def get_one_step_reachable_set(
    params,
    xt_bounds: jax_verify.IntervalBound,
    dynamics: nfl_dynamics.Dynamics,
    verif_fn=jax_verify.backward_crown_bound_propagation,
):
    """Computes min/max that could be achieved at next timestep,
    subject to NN relaxation, prev state bounds, and dynamics,
    by appending dynamics to end of NN.

    Args:
      params: list of weights and biases for the NN control policy
      xt_bounds: jax_verify.IntervalBound describing rectangular set of
        possible states x_t at current timestep
      dynamics: nfl_veripy.dynamics.Dynamics instance, describing state
        transition dynamics of system being controlled by NN policy
      verif_fn: jax_verify method to propagate initial state set thru system

    Returns:
      xt1_bounds: jax_verify.IntervalBound describing 1-step reachable set
    """

    logits_fn = functools.partial(predict_next_state, params, dynamics)
    xt1_bounds = verif_fn(logits_fn, xt_bounds)

    return xt1_bounds


def example_get_one_step_reachable_set():
    """Runs get_one_step_reachable_set on a simple NN, DoubleIntegrator,
    initial state set.

    Returns:
      output_bounds: jax_verify.IntervalBound rectangle of possible states at
      next timestep
    """
    dynamics = nfl_dynamics.DoubleIntegrator()
    initial_state_nominal = jnp.array([[0.8, 0.4]])
    input_bounds = jax_verify.IntervalBound(
        initial_state_nominal - 1.0, initial_state_nominal + 1.0
    )
    params, _, _ = setup_nn()
    output_bounds = get_one_step_reachable_set(params, input_bounds, dynamics)
    return output_bounds


def check_closed_loop_bounds_are_valid(
    params,
    dynamics,
    xt_bounds,
    seed=1701,
    num_timesteps=1,
    num_samples=100,
    reach_fn=get_multi_step_reachable_sets_unrolled,
    verif_fn=jax_verify.backward_crown_bound_propagation,
):
    """Confirms that bounds on next state (computed by ReachLP-Jax) are indeed
    outer bounds on next states via MC sampling.

    Also plots the given xt_bounds rectangle, the computed xt1_bounds
    rectangle, and the MC samples of x_{t} and their corresponding states at
    the next timestep, x_{t+1}.

    Args:
      params: list of weights and biases for the NN control policy
      dynamics: nfl_veripy.dynamics.Dynamics instance, describing state
        transition dynamics of system being controlled by NN policy
      xt_bounds: jax_verify.IntervalBound describing rectangular set of
        possible states x_t at current timestep
      seed: random seed for MC sampling of initial state set
      num_timesteps: number of timesteps to check reachable sets for
      num_samples: number of samples for MC sampling of initial state set
      reach_fn: function to use to compute N-step reachable sets
      verif_fn: jax_verify propagation function to get output bounds on a fn
        (used by reach_fn)

    Returns:
      True if computed output_bounds are indeed outside of all the MC sampled
      pts, otherwise False
    """
    key = jax.random.PRNGKey(seed)
    num_states = xt_bounds.lower.shape[1]
    xt_mc = jax.random.uniform(
        key,
        (num_samples, num_states),
        minval=xt_bounds.lower,
        maxval=xt_bounds.upper,
    )

    color = colors(0)
    plt.scatter(xt_mc[:, 0], xt_mc[:, 1], color=color)
    plt.gca().add_patch(
        Rectangle(
            xt_bounds.lower[0],
            xt_bounds.upper[0, 0] - xt_bounds.lower[0, 0],
            xt_bounds.upper[0, 1] - xt_bounds.lower[0, 1],
            facecolor="None",
            edgecolor=color,
        )
    )

    reachable_sets = reach_fn(
        params, xt_bounds, dynamics, num_timesteps, verif_fn=verif_fn
    )

    valid_bnds = True
    for t in range(1, num_timesteps + 1):
        u_mc = predict_mlp(params, dynamics.u_limits, xt_mc)
        xt1_mc = dynamics.dynamics_step_jnp(xt_mc, u_mc)

        xt1_bounds_mc = jax_verify.IntervalBound(
            xt1_mc.min(axis=0), xt1_mc.max(axis=0)
        )

        color = colors(t)
        plt.scatter(xt1_mc[:, 0], xt1_mc[:, 1], color=color)

        xt1_bounds = reachable_sets[t]
        plt.gca().add_patch(
            Rectangle(
                xt1_bounds.lower[0],
                xt1_bounds.upper[0, 0] - xt1_bounds.lower[0, 0],
                xt1_bounds.upper[0, 1] - xt1_bounds.lower[0, 1],
                facecolor="None",
                edgecolor=color,
            )
        )

        lower_bnds_valid = xt1_bounds.lower[0] <= xt1_bounds_mc.lower
        upper_bnds_valid = xt1_bounds.upper[0] >= xt1_bounds_mc.upper
        all_bnds_valid = jax.numpy.all(
            jax.numpy.logical_and(lower_bnds_valid, upper_bnds_valid)
        )
        valid_bnds = all_bnds_valid and valid_bnds

        xt_mc = xt1_mc.copy()

    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.legend()
    plt.show()

    return valid_bnds


def train(
    params_init,
    xs,
    us,
    dynamics,
    loss_weight,
    num_epochs=int(1e4),
    loss_freq=100,
    step_size=0.01,
    num_steps=1,
    xt0_bounds=jax_verify.IntervalBound(
        jnp.array([[2.5, -0.25]]), jnp.array([[3.0, 0.25]])
    ),
    plot=True,
):
    """Trains NN policy based on MPC imitation + safety (e.g., reach set size).

    Args:
      params_init: initial parameters of the NN control policy
      xs: jnp array (num_training_pts, num_states) of states (NN inputs) to be
        used in MPC imitation
      us: jnp array of (num_training_pts, num_control_inputs) of controls (NN
        outputs) to be used in MPC imitation
      dynamics: nfl_veripy.dynamics.Dynamics instance, describing state
        transition dynamics of system being controlled by NN policy
      loss_weight: coefficient for linear combination of MPC, ReachLP losses.
        0=MPC, 1=ReachLP, should be btwn [0,1].
      num_epochs: number of epochs to train for
      loss_freq: number of epochs between loss calculations for plotting (slows
        down training)
      step_size: how far to adjust params in direction of loss gradient
      num_steps: number of timesteps to use in reachable set calculation
      xt0_bounds: jax_verify.IntervalBound with min/max values of initial state
        set
      plot: bool to plot the loss function vs. epochs (false to skip plotting)

    Returns:
      params: final parameters of the NN control policy after training
    """
    params = params_init.copy()

    batched_predict = jax.vmap(predict_mlp, in_axes=(None, None, 0))
    losses = np.zeros((num_epochs // loss_freq, 2))

    def loss_multi_step_reachable_set_volume(params, input_bounds, dynamics):
        reachable_set = get_multi_step_reachable_set(
            params, input_bounds, dynamics, num_steps
        )
        return jnp.product(reachable_set.upper - reachable_set.lower)

    def loss_multi_step_reachable_set_origin(params, input_bounds, dynamics):
        reach_fn = get_multi_step_reachable_set

        reachable_set = reach_fn(params, input_bounds, dynamics, num_steps)
        return jnp.sum(jnp.abs(reachable_set.lower)) + jnp.sum(
            jnp.abs(reachable_set.upper)
        )

    def loss_multi_step_reachable_set_origin_radial(
        params, input_bounds, dynamics
    ):
        """Compute loss as 'area' of rectangle (each pt has area z=x^2+y^2)."""

        reachable_set = get_multi_step_reachable_set(
            params, input_bounds, dynamics, num_steps
        )
        x0, y0 = reachable_set.lower[0]
        x1, y1 = reachable_set.upper[0]
        # Double integral of (x**2 + y**2) from x=[x0, x1] and y=[y0, y1]
        loss = (x1**3 - x0**3) * (y1 - y0) + (y1**3 - y0**3) * (
            x1 - x0
        )
        return loss

    def loss_multi_step_reachable_set_fwd_bwd_similar(
        params, input_bounds, dynamics
    ):
        reachable_sets_bwd = get_multi_step_reachable_sets_unrolled(
            params,
            input_bounds,
            dynamics,
            num_steps,
            verif_fn=jax_verify.backward_crown_bound_propagation,
        )
        reachable_sets_fwd = get_multi_step_reachable_sets_unrolled(
            params,
            input_bounds,
            dynamics,
            num_steps,
            verif_fn=jax_verify.forward_crown_bound_propagation,
        )

        area_bwd = jnp.product(
            reachable_sets_bwd[-1].upper - reachable_sets_bwd[-1].lower
        )
        area_fwd = jnp.product(
            reachable_sets_fwd[-1].upper - reachable_sets_fwd[-1].lower
        )

        loss = area_fwd - area_bwd

        return loss

    def loss_mpc_mse(params, xs, us):
        us_pred = batched_predict(params, dynamics.u_limits, xs)
        return jnp.mean(jnp.square(us_pred - us))

    def loss(params, input_bounds, dynamics, xs, us, loss_weight):
        """Linear combination of MPC loss and Reachable Set loss
        (loss_weight: 0 <- MPC --------- Reach -> 1)."""

        loss_reach_fn = loss_multi_step_reachable_set_fwd_bwd_similar
        # loss_reach_fn = loss_multi_step_reachable_set_origin
        # loss_reach_fn = loss_multi_step_reachable_set_volume

        l_reach = loss_reach_fn(params, input_bounds, dynamics)
        l_mpc = loss_mpc_mse(params, xs, us)
        l_total = (loss_weight) * l_reach + (1 - loss_weight) * l_mpc
        return l_total

    @jax.jit
    def update(params, xs, us):
        grads = jax.grad(loss)(
            params, xt0_bounds, dynamics, xs, us, loss_weight
        )
        return [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)
        ]

    for i in tqdm(range(num_epochs)):
        if i % loss_freq == 0:
            l_total = loss(params, xt0_bounds, dynamics, xs, us, loss_weight)
            losses[i // loss_freq, 0] = i
            losses[i // loss_freq, 1] = l_total
        params = update(params, xs, us)

    if plot:
        plt.plot(losses[:, 0], losses[:, 1])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    return params


def get_next_action_with_mpc(
    x0s,
    dynamics,
    state_cost_matrix,
    control_cost_matrix,
    terminal_cost_matrix,
    n_mpc=10,
):
    """Starting from initial state, compute optimal trajectory with MPC,
    return the first action.

    Args:
      x0s: np array of initial states
      dynamics: nfl_veripy.dynamics.Dynamics instance, describing state
        transition dynamics of system being controlled by NN policy
      state_cost_matrix: LQR cost matrix to encourage state error to be
        minimized in optimal trajectory
      control_cost_matrix: LQR cost matrix to encourage usage of control input
        to be minimized in optimal trajectory
      terminal_cost_matrix: LQR terminal cost matrix (via solving discrete ARE)
      n_mpc: number of steps in MPC horizon

    Returns:
      us: np array of first timestep's control inputs for each state in x0s
    """

    num_initial_pts = x0s.shape[0]

    # Corresponding first control for each state in x0s
    us = np.empty((num_initial_pts, dynamics.num_inputs))

    # Control and state sequence, starting from a single state x0 \in x0s
    u = cp.Variable((n_mpc, dynamics.num_inputs))
    x = cp.Variable((n_mpc + 1, dynamics.num_states))
    x0 = cp.Parameter((dynamics.num_states,))

    cost = 0

    constrs = []
    constrs.append(x[0, :] == x0)
    step = 0
    while step < n_mpc:
        constr = x[step + 1, :] == dynamics.dynamics_step(
            x[step, :], u[step, :]
        )
        constrs.append(constr)

        # Input constraints
        constrs.append(u[step] <= dynamics.u_limits[:, 1])
        constrs.append(u[step] >= dynamics.u_limits[:, 0])

        # Control cost
        cost += cp.quad_form(u[step, :], control_cost_matrix)

        # State stage cost
        cost += cp.quad_form(x[step, :], state_cost_matrix)

        step += 1

    # Terminal state cost
    cost += cp.quad_form(x[n_mpc, :], terminal_cost_matrix)

    prob = cp.Problem(cp.Minimize(cost), constrs)

    for i in tqdm(range(num_initial_pts)):
        x0.value = np.asarray(x0s[i])
        prob.solve()
        us[i] = u.value[0, :]

    return us


def generate_mpc_dataset(
    dynamics,
    seed=1701,
    num_initial_pts=10,
    xt0_bounds=jax_verify.IntervalBound(
        jnp.array([[2.5, -0.25]]), jnp.array([[3.0, 0.25]])
    ),
    num_timesteps=5,
):
    """Computes a bunch of (state, action) pairs with an MPC algorithm
    starting in some initial state set.

    Args:
      dynamics: nfl_veripy.dynamics.Dynamics instance, describing state
        transition dynamics of system being controlled by NN policy
      seed: jax random seed for initial state sampling
      num_initial_pts: num of states to randomly sample (and start from) in MPC
      xt0_bounds: jax_verify.IntervalBound with min/max values of initial state
        set
      num_timesteps: number of discrete timesteps to run MPC forward

    Returns:
      xs: jnp array of size (num_timesteps, num_initial_pts, num_states) with
      states visited during the MPC rollouts
      us: jnp array of size (num_timesteps, num_initial_pts, num_inputs)
      with the actions taken during the corresponding state in xs
    """
    key = jax.random.PRNGKey(seed)
    xt_mc = jax.random.uniform(
        key,
        (num_initial_pts, dynamics.num_states),
        minval=xt0_bounds.lower,
        maxval=xt0_bounds.upper,
    )
    state_cost_matrix = jnp.eye(dynamics.num_states)
    control_cost_matrix = jnp.diag(jnp.ones((dynamics.num_inputs,)))
    terminal_cost_matrix = solve_discrete_are(
        dynamics.At, dynamics.bt, state_cost_matrix, control_cost_matrix
    )

    xs = np.zeros((num_timesteps, num_initial_pts, dynamics.num_states))
    us = np.zeros((num_timesteps, num_initial_pts, dynamics.num_inputs))

    for t in range(num_timesteps):
        xs[t] = xt_mc
        u_t = get_next_action_with_mpc(
            xt_mc,
            dynamics,
            state_cost_matrix,
            control_cost_matrix,
            terminal_cost_matrix,
            n_mpc=5,
        )
        xt_mc = dynamics.dynamics_step_jnp(xt_mc, u_t)
        us[t] = u_t

    xs = xs.reshape((-1, dynamics.num_states))
    us = us.reshape((-1, dynamics.num_inputs))
    return xs, us


def run_sims(
    xt_mc, policy, dynamics: nfl_dynamics.Dynamics, num_timesteps: int = 5
):
    """Runs simulations starting from states xt_mc, using given policy and
    dynamics."""
    xs = np.zeros((num_timesteps,) + xt_mc.shape)
    for t in range(num_timesteps):
        xs[t] = xt_mc
        u_t = policy(xt_mc)
        xt_mc = dynamics.dynamics_step_jnp(xt_mc, u_t)
    return xs


def run_policies(
    policies_dict,
    dynamics,
    xt0_bounds,
    num_timesteps=3,
    num_initial_pts=100,
    seed=1701,
):
    """Simulates and plots trajectories with reachable sets using various NN
    controller realizations."""

    key = jax.random.PRNGKey(seed)
    xt_mc = jax.random.uniform(
        key,
        (num_initial_pts, dynamics.num_states),
        minval=xt0_bounds.lower,
        maxval=xt0_bounds.upper,
    )

    batched_predict = jax.vmap(predict_mlp, in_axes=(None, None, 0))

    settings = [
        {
            "name": "NN: MPC",
            "params_key": "params_mpc",
            "marker": "o",
            "ls": "--",
        },
        {
            "name": "NN: Reach (1-step)",
            "params_key": "params_reach_1",
            "marker": "^",
            "ls": "-.",
        },
        {
            "name": "NN: Reach (2-step)",
            "params_key": "params_reach_2",
            "marker": "^",
            "ls": "-.",
        },
        {
            "name": "NN: MPC+Reach (2-step)",
            "params_key": "params_mpc_reach_2",
            "marker": "*",
            "ls": ":",
        },
        {"name": "NN", "params_key": "params", "marker": ".", "ls": "-"},
    ]

    for d in settings:
        if d["params_key"] not in policies_dict:
            continue
        params = policies_dict[d["params_key"]]
        d["params"] = params
        d["nn_function"] = functools.partial(
            batched_predict, params, dynamics.u_limits
        )

    for i in range(len(settings)):
        if settings[i]["params_key"] not in policies_dict:
            continue
        policy = settings[i]["nn_function"]
        marker = settings[i]["marker"]
        xs = run_sims(
            xt_mc.copy(), policy, dynamics, num_timesteps=num_timesteps
        )
        for t in range(num_timesteps):
            label = settings[i]["name"] if t == 0 else None
            plt.scatter(
                xs[t, :, 0],
                xs[t, :, 1],
                marker=marker,
                color=colors(t),
                label=label,
            )

    for i in range(len(settings)):
        if settings[i]["params_key"] not in policies_dict:
            continue
        params = settings[i]["params"]
        xt_bounds = xt0_bounds
        plt.gca().add_patch(
            Rectangle(
                xt_bounds.lower[0],
                xt_bounds.upper[0, 0] - xt_bounds.lower[0, 0],
                xt_bounds.upper[0, 1] - xt_bounds.lower[0, 1],
                facecolor="None",
                edgecolor=colors(0),
            )
        )

        reach_fn = get_multi_step_reachable_sets_unrolled
        verif_fn = jax_verify.forward_crown_bound_propagation
        reachable_sets = reach_fn(
            params, xt_bounds, dynamics, num_timesteps, verif_fn=verif_fn
        )

        for t in range(1, num_timesteps):
            xt_bounds = reachable_sets[t]
            plt.gca().add_patch(
                Rectangle(
                    xt_bounds.lower[0],
                    xt_bounds.upper[0, 0] - xt_bounds.lower[0, 0],
                    xt_bounds.upper[0, 1] - xt_bounds.lower[0, 1],
                    facecolor="None",
                    edgecolor=colors(t),
                    ls=settings[i]["ls"],
                )
            )

    plt.legend()
    plt.show()
