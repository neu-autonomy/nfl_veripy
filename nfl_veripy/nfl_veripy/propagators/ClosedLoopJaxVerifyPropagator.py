"""Compute reachable set of neural feedback loop in Jax."""
import functools
import itertools

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pypoman
import torch
from nfl_veripy import constraints
from nfl_veripy.utils.closed_loop_verification_jax import (
    backward_crown_bound_propagation_linfun,
    get_multi_step_reachable_sets_unrolled,
)
from nfl_veripy.utils.nn_jax import (
    predict_future_states,
    predict_mlp,
    predict_next_state,
)
from nfl_veripy.utils.utils import range_to_polytope
from tqdm import tqdm

import jax_verify

from .ClosedLoopPropagator import ClosedLoopPropagator


class ClosedLoopJaxPropagator(ClosedLoopPropagator):
    """Abstract class for fwd/bwd reachability using jax_verify library."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.reach_fn = None
        self.verif_fn = None
        self.num_iterations = num_iterations
        self.pre_compile = pre_compile

    def torch2network(self, torch_model):
        params = []
        act = None

        # Extract params (weights, biases) from torch layers, to be used in
        # jax.
        # Note: This propagator assumes a feed-forward relu NN.
        for m in torch_model.modules():
            if isinstance(m, torch.nn.Sequential):
                continue
            elif isinstance(m, torch.nn.ReLU):
                if act is None or act == "relu":
                    act = "relu"
                else:
                    raise ValueError(
                        "Don't support >1 types of activations in model."
                    )
            elif isinstance(m, torch.nn.Linear):
                w = m.weight.data.numpy().T
                b = m.bias.data.numpy()
                params.append((w, b))
            else:
                raise ValueError("That layer isn't supported.")
        self.params = params

        # Internally, we'll just use the typical torch stuff
        return torch_model

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_one_step_backprojection_set(
        self, output_constraint, input_constraint, num_partitions=None
    ):
        raise NotImplementedError


class ClosedLoopJaxIterativePropagator(ClosedLoopJaxPropagator):
    """Fwd/Bwd reachability using jax_verify library."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )
        self.verif_fn = jax_verify.backward_crown_bound_propagation

    def get_reachable_set(self, input_constraint, t_max):
        xt_bounds = jax_verify.IntervalBound(
            jnp.array([input_constraint.range[..., 0]]),
            jnp.array([input_constraint.range[..., 1]]),
        )
        num_timesteps = self.dynamics.tmax_to_num_timesteps(t_max)

        fun_to_prop = functools.partial(
            predict_next_state, self.params, self.dynamics
        )
        # fun_to_prop = functools.partial(predict_mlp_unclipped, self.params)
        all_bounds = [xt_bounds]
        for _ in range(num_timesteps):
            xt_bounds = self.verif_fn(fun_to_prop, xt_bounds)
            all_bounds.append(xt_bounds)
        reachable_sets = jnp.array(
            [
                jnp.array([all_bounds[i].lower, all_bounds[i].upper]).T
                for i in range(1, len(all_bounds))
            ]
        )

        reachable_sets = np.array(reachable_sets[:, :, 0, :])

        output_constraints = constraints.MultiTimestepLpConstraint(
            range=reachable_sets
        )

        return output_constraints, {}

    def get_backprojection_set(
        self,
        output_constraints,
        input_constraint,
        t_max,
        num_partitions=None,
        overapprox=False,
        refined=False,
    ):
        bps = []
        bp = output_constraints[0]
        if isinstance(bp, constraints.LpConstraint):
            A, b = range_to_polytope(bp.range)
            bp = constraints.PolytopeConstraint(A=A, b=b)
        infos = {"per_timestep": []}
        num_timesteps = self.dynamics.tmax_to_num_timesteps(t_max)
        for _ in range(num_timesteps):
            bp, info = self.get_one_step_backprojection_set(
                bp, num_partitions=num_partitions
            )
            bps.append(bp)
            infos["per_timestep"].append(info)
        return [bps], [infos]

    def get_one_step_backprojection_set(
        self,
        backreachable_set,
        target_sets,
        overapprox=False,
    ):
        info = {}
        # backreachable_set = self.get_one_step_backreachable_set(target_set)
        # info['backreachable_set'] = backreachable_set
        # info['target_set'] = copy.deepcopy(target_set)

        return self.get_one_step_backprojection_set_without_partitioning(
            backreachable_set, target_sets, info
        )
        # else:
        #   return self.get_one_step_backprojection_set_with_partitioning(
        #       backreachable_set, target_sets, info, num_partitions)

    def get_one_step_backprojection_set_without_partitioning(
        self, backreachable_set, target_sets, info
    ):
        A, b = range_to_polytope(backreachable_set.range)
        backprojection_set = constraints.PolytopeConstraint(A=A, b=b)
        for _ in range(self.num_iterations):
            backprojection_set = self.improve_backprojection_set(
                backprojection_set, target_sets
            )

        return backprojection_set, info

    def get_one_step_backprojection_set_with_partitioning(
        self, backreachable_set, target_set, info, num_partitions
    ):
        slope = np.divide(
            (backreachable_set.range[:, 1] - backreachable_set.range[:, 0]),
            num_partitions,
        )
        vertices = []

        # Iterate through each partition
        for element in itertools.product(
            *[range(num) for num in num_partitions.flatten()]
        ):
            element_ = np.array(element).reshape((self.dynamics.num_states,))
            br_cell_range = np.empty_like(backreachable_set.range)
            br_cell_range[:, 0] = backreachable_set.range[:, 0] + np.multiply(
                element_, slope
            )
            br_cell_range[:, 1] = backreachable_set.range[:, 0] + np.multiply(
                element_ + 1, slope
            )
            A, b = range_to_polytope(br_cell_range)
            backprojection_set = constraints.PolytopeConstraint(A=A, b=b)

            backprojection_set = self.iteratively_improve_backprojection_set(
                backprojection_set, target_set
            )
            if backprojection_set == "infeasible":
                continue

            if isinstance(backprojection_set, constraints.LpConstraint):
                v = np.array(
                    list(itertools.product(*backprojection_set.range))
                )
            else:
                v = pypoman.duality.compute_polytope_vertices(
                    backprojection_set.A, backprojection_set.b
                )
            if len(v) == 0:
                continue
            vertices.append(v)

        # Merge vertices of all partitions' BPOAs into a single polytope
        A, b = pypoman.duality.compute_polytope_halfspaces(
            pypoman.duality.convex_hull(np.vstack(vertices))
        )
        backprojection_set = constraints.PolytopeConstraint(A=A, b=b)

        return backprojection_set, info

    def iteratively_improve_backprojection_set(
        self, backprojection_set, target_set
    ):
        for _ in range(self.num_iterations):
            backprojection_set = self.improve_backprojection_set(
                backprojection_set, target_set
            )
            if backprojection_set == "infeasible":
                return "infeasible"

        return backprojection_set

    def get_one_step_backreachable_set(self, target_set):
        """Find (hyperrectangle) set of states that lead to target_set while
        following dynamics, x_limits, u_limits."""

        if self.dynamics.u_limits is None:
            print("self.dynamics.u_limits is None")
            print(
                "==> The backreachable set is probably the whole state space."
            )
            print("Giving up.")
            raise NotImplementedError
        else:
            u_min = self.dynamics.u_limits[:, 0]
            u_max = self.dynamics.u_limits[:, 1]

        xt = cp.Variable((self.dynamics.num_states, 2))
        ut = cp.Variable(self.dynamics.num_inputs)

        # For each dimension of the output constraint (facet/lp-dimension):
        # compute a bound of the NN output using the pre-computed matrices
        xt = cp.Variable(self.dynamics.num_states)
        ut = cp.Variable(self.dynamics.num_inputs)
        constrs = []
        constrs += [u_min <= ut]
        constrs += [ut <= u_max]

        # Note: state limits are not included in CDC 2022 paper
        # results/discussion. Included state limits to reduce size of
        # backreachable sets by eliminating states that are not physically
        # possible (e.g., maximum velocities)
        if self.dynamics.x_limits is not None:
            if isinstance(self.dynamics.x_limits, dict):
                for state in self.dynamics.x_limits:
                    constrs += [xt[state] >= self.dynamics.x_limits[state][0]]
                    constrs += [xt[state] <= self.dynamics.x_limits[state][1]]
            else:
                constrs += [xt >= self.dynamics.x_limits[:, 0]]
                constrs += [xt <= self.dynamics.x_limits[:, 1]]

        if isinstance(target_set, constraints.PolytopeConstraint):
            constrs += [
                target_set.A @ self.dynamics.dynamics_step(xt, ut)
                <= target_set.b
            ]
        elif isinstance(target_set, constraints.LpConstraint):
            constrs += [
                self.dynamics.dynamics_step(xt, ut) <= target_set.range[..., 1]
            ]
            constrs += [
                self.dynamics.dynamics_step(xt, ut) >= target_set.range[..., 0]
            ]

        obj_facets = np.eye(self.dynamics.num_states)
        num_facets = obj_facets.shape[0]
        coords = np.empty(
            (2 * self.dynamics.num_states, self.dynamics.num_states)
        )

        obj_facets_i = cp.Parameter(self.dynamics.num_states)
        obj = obj_facets_i @ xt
        min_prob = cp.Problem(cp.Minimize(obj), constrs)
        max_prob = cp.Problem(cp.Maximize(obj), constrs)
        for i in range(num_facets):
            obj_facets_i.value = obj_facets[i, :]
            min_prob.solve()
            coords[2 * i, :] = xt.value
            max_prob.solve()
            coords[2 * i + 1, :] = xt.value

        # min/max of each element of xt in the backreachable set
        ranges = np.vstack([coords.min(axis=0), coords.max(axis=0)]).T

        backreachable_set = constraints.LpConstraint(range=ranges)
        return backreachable_set


class ClosedLoopJaxPolytopePropagator(ClosedLoopJaxIterativePropagator):
    """Backward reachability using jax_verify, where BP sets are improved
    using closed-form solution based on polytope relaxation domains (DRIP)."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )
        self.boundary_type = "polytope"

    def improve_backprojection_set(
        self, initial_backprojection_set, target_set
    ):
        fun_to_prop = functools.partial(
            predict_next_state, self.params, self.dynamics
        )

        vertices = np.stack(
            pypoman.compute_polytope_vertices(
                initial_backprojection_set.A, initial_backprojection_set.b
            )
        )
        # num_vertices = vertices.shape[0]

        input_bounds = None  # simplex bound goes here
        raise NotImplementedError
        input_interval_bounds = jax_verify.IntervalBound(
            jnp.array(np.min(vertices, axis=0)),
            jnp.array(np.max(vertices, axis=0)),
        )

        def predict_next_state_simplex(params, dynamics, xt_simplex):
            xt = jnp.dot(xt_simplex, vertices)
            ut = predict_mlp(params, dynamics.u_limits, xt)
            xt1 = dynamics.dynamics_step_jnp(xt, ut)
            return xt1

        fun_to_prop = functools.partial(
            predict_next_state_simplex, self.params, self.dynamics
        )

        obj = jnp.expand_dims(target_set.A, 1)
        linfuns = backward_crown_bound_propagation_linfun(
            fun_to_prop, input_bounds, obj=obj
        )

        B = jnp.vstack(
            [
                linfuns[0].lin_coeffs,
                jnp.eye(self.dynamics.num_states),
                -jnp.eye(self.dynamics.num_states),
            ]
        )
        c = jnp.hstack(
            [
                target_set.b - linfuns[0].offset,
                input_interval_bounds.upper,
                -input_interval_bounds.lower,
            ]
        )
        backprojection_set = constraints.PolytopeConstraint(
            A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        )

        return backprojection_set


class ClosedLoopJaxPolytopeJittedPropagator(ClosedLoopJaxIterativePropagator):
    """JIT-enabled backward reachability using jax_verify, where BP sets are
    improved using closed-form solution based on polytope relaxation domains
    (DRIP).
    """

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )

    def torch2network(self, torch_model):
        return_args = super().torch2network(torch_model)

        self.fun_to_prop = functools.partial(
            predict_next_state, self.params, self.dynamics
        )
        if self.pre_compile:
            vertices_shapes = [(4, 2), (8, 2), (16, 2), (32, 2), (64, 2)]
            obj_shapes = [
                (4, 1, 2),
                (8, 1, 2),
                (16, 1, 2),
                (32, 1, 2),
                (64, 1, 2),
            ]

            num_elements = len(vertices_shapes) * len(obj_shapes)
            pre_compile = False
            if pre_compile:
                with tqdm(total=num_elements) as pbar:
                    for obj_shape in obj_shapes:
                        for vertices_shape in vertices_shapes:
                            vertices = np.zeros(vertices_shape)
                            obj = np.zeros(obj_shape)
                            input_bounds = None  # simplex bounds go here
                            jittable_input_bounds = input_bounds.to_jittable()
                            _, _ = bound_prop_fun(
                                jittable_input_bounds,
                                vertices,
                                obj,
                                self.fun_to_prop,
                            )
                            pbar.update(1)

        return return_args

    def improve_backprojection_set(
        self, initial_backprojection_set, target_set
    ):
        vertices = pypoman.compute_polytope_vertices(
            initial_backprojection_set.A, initial_backprojection_set.b
        )
        vertices = np.stack(vertices)
        num_vertices_to_pad = (
            2 ** int(jnp.ceil(jnp.log2(vertices.shape[0]))) - vertices.shape[0]
        )
        vertices = np.vstack(
            [vertices, np.tile(vertices[-1, :], (num_vertices_to_pad, 1))]
        )
        input_bounds = None  # simplex_bounds go here.
        raise NotImplementedError
        input_interval_bounds = jax_verify.IntervalBound(
            jnp.array(np.min(vertices, axis=0)),
            jnp.array(np.max(vertices, axis=0)),
        )

        obj = target_set.A
        num_facets_to_pad = (
            2 ** int(jnp.ceil(jnp.log2(obj.shape[0]))) - obj.shape[0]
        )
        obj = np.vstack([obj, np.zeros((num_facets_to_pad, 2))])
        obj = jnp.expand_dims(obj, 1)

        jittable_input_bounds = input_bounds.to_jittable()

        lin_coeffs, offset = simplex_bound_prop_fun(
            jittable_input_bounds, vertices, obj, self.fun_to_prop
        )

        if num_facets_to_pad > 0:
            lin_coeffs = lin_coeffs[:-num_facets_to_pad]
            offset = offset[:-num_facets_to_pad]

        # This could be tighter if we used initial_backprojection_set instead
        # of input_interval_bounds, but the former adds a lot of redundant
        # constraints that would need to be dealt with.
        B = jnp.vstack(
            [
                lin_coeffs,
                jnp.eye(self.dynamics.num_states),
                -jnp.eye(self.dynamics.num_states),
            ]
        )
        c = jnp.hstack(
            [
                target_set.b - offset,
                input_interval_bounds.upper,
                -input_interval_bounds.lower,
            ]
        )
        backprojection_set = constraints.PolytopeConstraint(
            A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        )

        return backprojection_set


class ClosedLoopJaxRectanglePropagator(ClosedLoopJaxIterativePropagator):
    """Backward reachability using jax_verify, where BP sets are improved
    using a closed-form solution based on hyperrectangle relaxation domains
    (DRIP-HPoly)."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )
        self.boundary_type = "polytope"

    def improve_backprojection_set(
        self, initial_backprojection_set, target_sets
    ):
        if isinstance(
            initial_backprojection_set, constraints.PolytopeConstraint
        ):
            input_bounds_np = initial_backprojection_set.to_linf()
            input_bounds = jax_verify.IntervalBound(
                jnp.array([input_bounds_np[:, 0]]),
                jnp.array([input_bounds_np[:, 1]]),
            )
        elif isinstance(initial_backprojection_set, constraints.LpConstraint):
            input_bounds = jax_verify.IntervalBound(
                jnp.array([initial_backprojection_set.range[:, 0]]),
                jnp.array([initial_backprojection_set.range[:, 1]]),
            )

        fun_to_prop = functools.partial(
            predict_next_state, self.params, self.dynamics
        )

        target_set = target_sets.get_constraint_at_time_index(-1)
        if isinstance(target_sets, constraints.LpConstraint):
            A, b = target_set.get_polytope()
            target_set = constraints.PolytopeConstraint(A, b)

        obj = jnp.expand_dims(target_set.A, 1)
        linfuns = backward_crown_bound_propagation_linfun(
            fun_to_prop, input_bounds, obj=obj
        )

        B = jnp.vstack(
            [
                linfuns[0].lin_coeffs[:, 0, :],
                jnp.eye(self.dynamics.num_states),
                -jnp.eye(self.dynamics.num_states),
            ]
        )
        c = jnp.hstack(
            [
                target_set.b - linfuns[0].offset,
                input_bounds.upper[0, :],
                -input_bounds.lower[0, :],
            ]
        )

        backprojection_set = constraints.PolytopeConstraint(
            A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        )

        return backprojection_set


class ClosedLoopJaxRectangleJittedPropagator(ClosedLoopJaxIterativePropagator):
    """JIT-enabled backward reachability using jax_verify, where BP sets are
    improved using a closed-form solution based on hyperrectangle relaxation
    domains (DRIP-HPoly)."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )

    def torch2network(self, torch_model):
        return_args = super().torch2network(torch_model)

        self.fun_to_prop = functools.partial(
            predict_next_state, self.params, self.dynamics
        )

        if self.pre_compile:
            # exponential increase in number of vertices w/ 2-state system
            vertices_shapes = [(4, 2), (8, 2), (16, 2), (32, 2), (64, 2)]
            # exponential increase in number of polytope facets w/ 2-state sys
            obj_shapes = [
                (4, 1, 2),
                (8, 1, 2),
                (16, 1, 2),
                (32, 1, 2),
                (64, 1, 2),
            ]

            num_elements = len(vertices_shapes) * len(obj_shapes)
            with tqdm(total=num_elements) as pbar:
                for obj_shape in obj_shapes:
                    for vertices_shape in vertices_shapes:
                        vertices = np.zeros(vertices_shape)
                        obj = np.zeros(obj_shape)
                        input_bounds = None  # simplex bound goes here
                        jittable_input_bounds = input_bounds.to_jittable()
                        _, _ = bound_prop_fun(
                            jittable_input_bounds,
                            vertices,
                            obj,
                            self.fun_to_prop,
                        )
                        pbar.update(1)

        return return_args

    def improve_backprojection_set(
        self, initial_backprojection_set, target_set
    ):
        if isinstance(
            initial_backprojection_set, constraints.PolytopeConstraint
        ):
            input_bounds_np = initial_backprojection_set.to_linf()
            input_bounds = jax_verify.IntervalBound(
                jnp.array(input_bounds_np[:, 0]),
                jnp.array(input_bounds_np[:, 1]),
            )
        elif isinstance(initial_backprojection_set, constraints.LpConstraint):
            input_bounds = jax_verify.IntervalBound(
                jnp.array(initial_backprojection_set.range[:, 0]),
                jnp.array(initial_backprojection_set.range[:, 1]),
            )

        obj = target_set.A
        num_facets_to_pad = (
            2 ** int(jnp.ceil(jnp.log2(obj.shape[0]))) - obj.shape[0]
        )
        obj = np.vstack([obj, np.zeros((num_facets_to_pad, 2))])
        obj = jnp.expand_dims(obj, 1)

        jittable_input_bounds = input_bounds.to_jittable()

        lin_coeffs, offset = bound_prop_fun(
            jittable_input_bounds, obj, self.fun_to_prop
        )
        if num_facets_to_pad > 0:
            lin_coeffs = lin_coeffs[:-num_facets_to_pad]
            offset = offset[:-num_facets_to_pad]

        B = jnp.vstack(
            [
                lin_coeffs,
                jnp.eye(self.dynamics.num_states),
                -jnp.eye(self.dynamics.num_states),
            ]
        )
        c = jnp.hstack(
            [target_set.b - offset, input_bounds.upper, -input_bounds.lower]
        )

        backprojection_set = constraints.PolytopeConstraint(
            A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        )

        return backprojection_set


class ClosedLoopJaxLPJittedPropagator(ClosedLoopJaxIterativePropagator):
    """JIT-enabled backward reachability using jax_verify, where BP sets are
    improved using a LP formulation."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )

    def torch2network(self, torch_model):
        return_args = super().torch2network(torch_model)

        self.fun_to_prop = functools.partial(
            predict_mlp, self.params, self.dynamics.u_limits
        )

        if self.pre_compile:
            raise NotImplementedError

        return return_args

    def improve_backprojection_set(
        self, initial_backprojection_set, target_set
    ):
        if isinstance(
            initial_backprojection_set, constraints.PolytopeConstraint
        ):
            initial_backprojection_set = constraints.LpConstraint(
                range=initial_backprojection_set.to_linf()
            )

        input_bounds = jax_verify.IntervalBound(
            jnp.array(initial_backprojection_set.range[:, 0]),
            jnp.array(initial_backprojection_set.range[:, 1]),
        )
        jittable_input_bounds = input_bounds.to_jittable()

        obj = jnp.expand_dims(
            jnp.vstack(
                [
                    jnp.eye(self.dynamics.num_inputs),
                    -jnp.eye(self.dynamics.num_inputs),
                ]
            ),
            1,
        )

        lin_coeffs, offset = bound_prop_fun(
            jittable_input_bounds, obj, self.fun_to_prop
        )

        # An over-approximation of the backprojection set is the set of:
        # all xt s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        xt = cp.Variable((self.dynamics.num_states,))
        ut = cp.Variable((self.dynamics.num_inputs,))
        constrs = []

        # Constraints to ensure that xt stays within the backreachable set
        constrs += [input_bounds.lower <= xt]
        constrs += [xt <= input_bounds.upper]

        # Constraints to ensure xt reaches the target set given ut
        if isinstance(target_set, constraints.PolytopeConstraint):
            constrs += [
                target_set.A @ self.dynamics.dynamics_step(xt, ut)
                <= target_set.b
            ]
        elif isinstance(target_set, constraints.LpConstraint):
            constrs += [
                self.dynamics.dynamics_step(xt, ut) <= target_set.range[:, 1]
            ]
            constrs += [
                self.dynamics.dynamics_step(xt, ut) >= target_set.range[:, 0]
            ]

        half_index = self.dynamics.num_states // 2
        lower_slope = lin_coeffs[:half_index]
        lower_intercept = offset[:half_index]
        upper_slope = -lin_coeffs[half_index:]
        upper_intercept = -offset[half_index:]

        # Constraints to ensure that ut satisfies the affine bounds
        constrs += [lower_slope @ xt + lower_intercept <= ut]
        constrs += [ut <= upper_slope @ xt + upper_intercept]

        # Solve optimization problem (min and max) for each state
        obj_facets = np.vstack(
            [
                np.eye(self.dynamics.num_states),
                -np.eye(self.dynamics.num_states),
            ]
        )
        obj_facets_i = cp.Parameter(self.dynamics.num_states)
        obj = obj_facets_i @ xt
        prob = cp.Problem(cp.Maximize(obj), constrs)
        b_ = np.empty(2 * self.dynamics.num_states)
        for i in range(2 * self.dynamics.num_states):
            obj_facets_i.value = obj_facets[i, :]
            prob.solve()
        ranges = np.vstack(
            [-b_[self.dynamics.num_states :], b_[: self.dynamics.num_states]]
        ).T

        backprojection_set = constraints.LpConstraint(range=ranges)

        return backprojection_set


class ClosedLoopJaxLPPropagator(ClosedLoopJaxIterativePropagator):
    """Backward reachability using jax_verify, where BP sets are improved
    using a LP formulation."""

    def __init__(
        self,
        input_shape=None,
        dynamics=None,
        num_iterations=1,
        pre_compile=False,
    ):
        super().__init__(
            input_shape=input_shape,
            dynamics=dynamics,
            num_iterations=num_iterations,
            pre_compile=pre_compile,
        )
        self.boundary_type = "rectangle"

    def torch2network(self, torch_model):
        return_args = super().torch2network(torch_model)

        self.fun_to_prop = functools.partial(
            predict_mlp, self.params, self.dynamics.u_limits
        )

        if self.pre_compile:
            raise NotImplementedError

        return return_args

    def improve_backprojection_set(
        self, initial_backprojection_set, target_sets
    ):
        if isinstance(
            initial_backprojection_set, constraints.PolytopeConstraint
        ):
            initial_backprojection_set = constraints.LpConstraint(
                range=initial_backprojection_set.to_linf()
            )

        input_bounds = jax_verify.IntervalBound(
            jnp.array([initial_backprojection_set.range[:, 0]]),
            jnp.array([initial_backprojection_set.range[:, 1]]),
        )

        obj = jnp.expand_dims(
            jnp.vstack(
                [
                    jnp.eye(self.dynamics.num_inputs),
                    -jnp.eye(self.dynamics.num_inputs),
                ]
            ),
            1,
        )
        linfuns = backward_crown_bound_propagation_linfun(
            self.fun_to_prop, input_bounds, obj=obj
        )

        # An over-approximation of the backprojection set is the set of:
        # all xt s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        xt = cp.Variable((self.dynamics.num_states,))
        ut = cp.Variable((self.dynamics.num_inputs,))
        constrs = []

        # Constraints to ensure that xt stays within the backreachable set
        constrs += [input_bounds.lower[0, :] <= xt]
        constrs += [xt <= input_bounds.upper[0, :]]

        target_set = target_sets.get_constraint_at_time_index(-1)

        # Constraints to ensure xt reaches the target set given ut
        if isinstance(target_set, constraints.PolytopeConstraint):
            constrs += [
                target_set.A @ self.dynamics.dynamics_step(xt, ut)
                <= target_set.b
            ]
        elif isinstance(target_set, constraints.LpConstraint):
            constrs += [
                self.dynamics.dynamics_step(xt, ut) <= target_set.range[:, 1]
            ]
            constrs += [
                self.dynamics.dynamics_step(xt, ut) >= target_set.range[:, 0]
            ]

        half_index = linfuns[0].shape[0] // 2
        lower_slope = linfuns[0].lin_coeffs[:half_index][0]
        lower_intercept = linfuns[0].offset[:half_index][0]
        upper_slope = -linfuns[0].lin_coeffs[half_index:][0]
        upper_intercept = -linfuns[0].offset[half_index:][0]

        # Constraints to ensure that ut satisfies the affine bounds
        constrs += [lower_slope @ xt + lower_intercept <= ut]
        constrs += [ut <= upper_slope @ xt + upper_intercept]

        # Solve optimization problem (min and max) for each state
        obj_facets = np.vstack(
            [
                np.eye(self.dynamics.num_states),
                -np.eye(self.dynamics.num_states),
            ]
        )
        obj_facets_i = cp.Parameter(self.dynamics.num_states)
        obj = obj_facets_i @ xt
        prob = cp.Problem(cp.Maximize(obj), constrs)
        b_ = np.empty(2 * self.dynamics.num_states)
        for i in range(2 * self.dynamics.num_states):
            obj_facets_i.value = obj_facets[i, :]
            prob.solve()
            b_[i] = prob.value

        if prob.status == "infeasible":
            return "infeasible"
        ranges = np.vstack(
            [-b_[self.dynamics.num_states :], b_[: self.dynamics.num_states]]
        ).T
        backprojection_set = constraints.LpConstraint(range=ranges)

        return backprojection_set


class ClosedLoopJaxUnrolledPropagator(ClosedLoopJaxPropagator):
    """Run CROWN on unrolled closed-loop dynamics, dyn(con(...dyn(con(x))))."""

    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.reach_fn = get_multi_step_reachable_sets_unrolled
        self.verif_fn = jax_verify.backward_crown_bound_propagation

    def get_reachable_set(self, input_constraint, t_max):
        xt_bounds = jax_verify.IntervalBound(
            jnp.array(
                input_constraint.range[..., 0].reshape(
                    -1, self.dynamics.num_states
                )
            ),
            jnp.array(
                input_constraint.range[..., 1].reshape(
                    -1, self.dynamics.num_states
                )
            ),
        )

        num_timesteps = self.dynamics.tmax_to_num_timesteps(t_max)

        fun_to_prop = functools.partial(
            predict_future_states, self.params, self.dynamics, num_timesteps
        )
        bounds = self.verif_fn(fun_to_prop, xt_bounds)

        lbs = jnp.array([bounds[i].lower for i in range(1, num_timesteps + 1)])
        ubs = jnp.array([bounds[i].upper for i in range(1, num_timesteps + 1)])
        reachable_sets = jnp.stack([lbs, ubs]).transpose(2, 1, 3, 0).squeeze()

        reachable_sets = np.array(reachable_sets)

        output_constraints = constraints.MultiTimestepLpConstraint(
            range=reachable_sets
        )

        return output_constraints, {}


class ClosedLoopJaxUnrolledJittedPropagator(ClosedLoopJaxPropagator):
    """(JIT-enabled) Run CROWN on unrolled closed-loop dynamics,
    dyn(con(...dyn(con(x))))."""

    def __init__(self, input_shape=None, dynamics=None):
        super().__init__(input_shape=input_shape, dynamics=dynamics)
        self.reach_fn = get_multi_step_reachable_sets_unrolled
        self.verif_fn = jax_verify.backward_crown_bound_propagation

    def get_reachable_set(self, initial_state_set, t_max):
        initial_state_set_jit = initial_state_set.to_jittable()
        reachable_sets_jnp, info = self.get_reachable_set_jitted(
            initial_state_set_jit, t_max
        )
        reachable_sets = constraints.MultiTimestepLpConstraint(
            range=np.array(reachable_sets_jnp)
        )
        return reachable_sets, info

    @functools.partial(jax.jit, static_argnames=["self", "t_max"])
    def get_reachable_set_jitted(self, initial_state_set_jit, t_max):
        num_timesteps = self.dynamics.tmax_to_num_timesteps(t_max)

        def bound_prop_fun_(inp_bound, fun_to_prop):
            (inp_bound_unjit,) = constraints.unjit_lp_constraints(inp_bound)
            bounds = self.verif_fn(fun_to_prop, inp_bound_unjit)
            lbs = jnp.array(
                [bounds[i].lower for i in range(1, num_timesteps + 1)]
            )
            ubs = jnp.array(
                [bounds[i].upper for i in range(1, num_timesteps + 1)]
            )
            return jnp.stack([lbs, ubs]).transpose(1, 2, 0)

        fun_to_prop = functools.partial(
            predict_future_states, self.params, self.dynamics, num_timesteps
        )
        reachable_sets_jnp = bound_prop_fun_(
            initial_state_set_jit, fun_to_prop
        )

        return reachable_sets_jnp, {}


@functools.partial(jax.jit, static_argnames=["fun_to_prop"])
def simplex_bound_prop_fun(inp_bound, vertices, obj, fun_to_prop):
    (inp_bound,) = jax_verify.src.bound_propagation.unjit_inputs(inp_bound)
    linfuns = backward_crown_bound_propagation_linfun(
        lambda x: fun_to_prop(jnp.dot(x, vertices)), inp_bound, obj=obj
    )
    return linfuns[0].lin_coeffs, linfuns[0].offset


@functools.partial(jax.jit, static_argnames=["fun_to_prop"])
def bound_prop_fun(inp_bound, obj, fun_to_prop):
    (inp_bound,) = jax_verify.src.bound_propagation.unjit_inputs(inp_bound)
    linfuns = backward_crown_bound_propagation_linfun(
        fun_to_prop, inp_bound, obj=obj
    )
    return linfuns[0].lin_coeffs, linfuns[0].offset
