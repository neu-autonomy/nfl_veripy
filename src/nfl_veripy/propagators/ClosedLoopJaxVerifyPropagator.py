"""Compute reachable set of neural feedback loop in Jax."""
import functools
from typing import Callable, Optional

import cvxpy as cp
import jax
import jax.numpy as jnp
import jax_verify
import numpy as np
import torch
from tqdm import tqdm

import nfl_veripy.dynamics as dynamics
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

from .ClosedLoopPropagator import ClosedLoopPropagator


class ClosedLoopJaxPropagator(ClosedLoopPropagator):
    """Abstract class for fwd/bwd reachability using jax_verify library."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.reach_fn: Callable = get_multi_step_reachable_sets_unrolled
        self.verif_fn: Callable = jax_verify.backward_crown_bound_propagation
        self.pre_compile: bool = False

    def torch2network(
        self, torch_model: torch.nn.Sequential
    ) -> torch.nn.Sequential:
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

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_one_step_backprojection_set(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
        overapprox: bool = False,
        infos: dict = {},
        facet_inds_to_optimize: Optional[np.ndarray] = None,
    ) -> tuple[Optional[constraints.SingleTimestepConstraint], dict]:
        raise NotImplementedError


class ClosedLoopJaxIterativePropagator(ClosedLoopJaxPropagator):
    """Fwd/Bwd reachability using jax_verify library."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.verif_fn = jax_verify.backward_crown_bound_propagation

    def get_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint, t_max: int
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        initial_set_range = initial_set.to_range()
        xt_bounds = jax_verify.IntervalBound(
            jnp.array([initial_set_range[..., 0]]),
            jnp.array([initial_set_range[..., 1]]),
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
        reachable_sets_jnp = jnp.array(
            [
                jnp.array([all_bounds[i].lower, all_bounds[i].upper]).T
                for i in range(1, len(all_bounds))
            ]
        )

        reachable_sets_np = np.array(reachable_sets_jnp[:, :, 0, :])

        reachable_sets = constraints.MultiTimestepConstraint(
            constraints=[
                constraints.LpConstraint(range=r) for r in reachable_sets_np
            ]
        )

        return reachable_sets, {}

    def get_one_step_backprojection_set(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
        overapprox: bool = False,
        info: dict = {},
        facet_inds_to_optimize: Optional[np.ndarray] = None,
    ) -> tuple[Optional[constraints.SingleTimestepConstraint], dict]:
        A, b = range_to_polytope(backreachable_set.range)
        backprojection_set = constraints.PolytopeConstraint(A=A, b=b)
        improved_backprojection_set = (
            self.iteratively_improve_backprojection_set(
                backprojection_set, target_sets
            )
        )

        return improved_backprojection_set, info

    def improve_backprojection_set(
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        raise NotImplementedError

    def iteratively_improve_backprojection_set(
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        for _ in range(self.num_iterations):
            backprojection_set = self.improve_backprojection_set(
                backprojection_set, target_sets
            )  # type: ignore
            if backprojection_set.is_infeasible:
                return backprojection_set

        return backprojection_set


class ClosedLoopJaxPolytopePropagator(ClosedLoopJaxIterativePropagator):
    """Backward reachability using jax_verify, where BP sets are improved
    using closed-form solution based on polytope relaxation domains (DRIP)."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "polytope"

    def improve_backprojection_set(
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        raise NotImplementedError

        # fun_to_prop = functools.partial(
        #     predict_next_state, self.params, self.dynamics
        # )

        # vertices = backprojection_set.get_vertices()

        # input_bounds = None  # simplex bound goes here

        # input_interval_bounds = jax_verify.IntervalBound(
        #     jnp.array(np.min(vertices, axis=0)),
        #     jnp.array(np.max(vertices, axis=0)),
        # )

        # def predict_next_state_simplex(params, dynamics, xt_simplex):
        #     xt = jnp.dot(xt_simplex, vertices)
        #     ut = predict_mlp(params, dynamics.u_limits, xt)
        #     xt1 = dynamics.dynamics_step_jnp(xt, ut)
        #     return xt1

        # fun_to_prop = functools.partial(
        #     predict_next_state_simplex, self.params, self.dynamics
        # )

        # obj = jnp.expand_dims(target_set.A, 1)
        # linfuns = backward_crown_bound_propagation_linfun(
        #     fun_to_prop, input_bounds, obj=obj
        # )

        # B = jnp.vstack(
        #     [
        #         linfuns[0].lin_coeffs,
        #         jnp.eye(self.dynamics.num_states),
        #         -jnp.eye(self.dynamics.num_states),
        #     ]
        # )
        # c = jnp.hstack(
        #     [
        #         target_set.b - linfuns[0].offset,
        #         input_interval_bounds.upper,
        #         -input_interval_bounds.lower,
        #     ]
        # )
        # backprojection_set = constraints.PolytopeConstraint(
        #     A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        # )

        # return backprojection_set


class ClosedLoopJaxPolytopeJittedPropagator(ClosedLoopJaxIterativePropagator):
    """JIT-enabled backward reachability using jax_verify, where BP sets are
    improved using closed-form solution based on polytope relaxation domains
    (DRIP).
    """

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "polytope"

    def torch2network(
        self, torch_model: torch.nn.Sequential
    ) -> torch.nn.Sequential:
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
                            raise NotImplementedError
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
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        raise NotImplementedError

        # vertices = backprojection_set.get_vertices()
        # num_vertices_to_pad = (
        #     2 ** int(jnp.ceil(jnp.log2(vertices.shape[0])))
        #     - vertices.shape[0]
        # )
        # vertices = np.vstack(
        #     [vertices, np.tile(vertices[-1, :], (num_vertices_to_pad, 1))]
        # )
        # input_bounds = None  # simplex_bounds go here.
        # input_interval_bounds = jax_verify.IntervalBound(
        #     jnp.array(np.min(vertices, axis=0)),
        #     jnp.array(np.max(vertices, axis=0)),
        # )

        # obj = target_set.A
        # num_facets_to_pad = (
        #     2 ** int(jnp.ceil(jnp.log2(obj.shape[0]))) - obj.shape[0]
        # )
        # obj = np.vstack([obj, np.zeros((num_facets_to_pad, 2))])
        # obj = jnp.expand_dims(obj, 1)

        # jittable_input_bounds = input_bounds.to_jittable()

        # lin_coeffs, offset = simplex_bound_prop_fun(
        #     jittable_input_bounds, vertices, obj, self.fun_to_prop
        # )

        # if num_facets_to_pad > 0:
        #     lin_coeffs = lin_coeffs[:-num_facets_to_pad]
        #     offset = offset[:-num_facets_to_pad]

        # # This could be tighter if we used initial_backprojection_set instead
        # # of input_interval_bounds, but the former adds a lot of redundant
        # # constraints that would need to be dealt with.
        # B = jnp.vstack(
        #     [
        #         lin_coeffs,
        #         jnp.eye(self.dynamics.num_states),
        #         -jnp.eye(self.dynamics.num_states),
        #     ]
        # )
        # c = jnp.hstack(
        #     [
        #         target_set.b - offset,
        #         input_interval_bounds.upper,
        #         -input_interval_bounds.lower,
        #     ]
        # )
        # backprojection_set = constraints.PolytopeConstraint(
        #     A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        # )

        # return backprojection_set


class ClosedLoopJaxRectanglePropagator(ClosedLoopJaxIterativePropagator):
    """Backward reachability using jax_verify, where BP sets are improved
    using a closed-form solution based on hyperrectangle relaxation domains
    (DRIP-HPoly)."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "polytope"

    def improve_backprojection_set(
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        input_bounds_np = backprojection_set.to_range()
        input_bounds = jax_verify.IntervalBound(
            jnp.array([input_bounds_np[:, 0]]),
            jnp.array([input_bounds_np[:, 1]]),
        )

        fun_to_prop = functools.partial(
            predict_next_state, self.params, self.dynamics
        )

        target_set = target_sets.get_constraint_at_time_index(-1)
        target_set_A, target_set_b = target_set.get_polytope()

        obj = jnp.expand_dims(target_set_A, 1)
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
                target_set_b - linfuns[0].offset,
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

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "rectangle"

    def torch2network(
        self, torch_model: torch.nn.Sequential
    ) -> torch.nn.Sequential:
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
                        raise NotImplementedError
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
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        input_bounds_np = backprojection_set.to_range()
        input_bounds = jax_verify.IntervalBound(
            jnp.array([input_bounds_np[:, 0]]),
            jnp.array([input_bounds_np[:, 1]]),
        )

        target_set = target_sets.get_constraint_at_time_index(-1)
        target_set_A, target_set_b = target_set.get_polytope()
        num_facets_to_pad = (
            2 ** int(jnp.ceil(jnp.log2(target_set_A.shape[0])))
            - target_set_A.shape[0]
        )
        obj = jnp.expand_dims(
            np.vstack([target_set_A, np.zeros((num_facets_to_pad, 2))]), 1
        )

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
            [target_set_b - offset, input_bounds.upper, -input_bounds.lower]
        )

        backprojection_set = constraints.PolytopeConstraint(
            A=np.array(B, dtype=np.double), b=np.array(c, dtype=np.double)
        )

        return backprojection_set


class ClosedLoopJaxLPJittedPropagator(ClosedLoopJaxIterativePropagator):
    """JIT-enabled backward reachability using jax_verify, where BP sets are
    improved using a LP formulation."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "rectangle"

    def torch2network(
        self, torch_model: torch.nn.Sequential
    ) -> torch.nn.Sequential:
        return_args = super().torch2network(torch_model)

        self.fun_to_prop = functools.partial(
            predict_mlp, self.params, self.dynamics.u_limits
        )

        if self.pre_compile:
            raise NotImplementedError

        return return_args

    def improve_backprojection_set(
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        input_bounds_np = backprojection_set.to_range()
        input_bounds = jax_verify.IntervalBound(
            jnp.array([input_bounds_np[:, 0]]),
            jnp.array([input_bounds_np[:, 1]]),
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
        target_set = target_sets.get_constraint_at_time_index(-1)
        target_set_A, target_set_b = target_set.get_polytope()
        constrs += [
            target_set_A @ self.dynamics.dynamics_step(xt, ut) <= target_set_b
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

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "rectangle"

    def torch2network(
        self, torch_model: torch.nn.Sequential
    ) -> torch.nn.Sequential:
        return_args = super().torch2network(torch_model)

        self.fun_to_prop = functools.partial(
            predict_mlp, self.params, self.dynamics.u_limits
        )

        if self.pre_compile:
            raise NotImplementedError

        return return_args

    def improve_backprojection_set(
        self,
        backprojection_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:
        input_bounds_np = backprojection_set.to_range()
        input_bounds = jax_verify.IntervalBound(
            jnp.array([input_bounds_np[:, 0]]),
            jnp.array([input_bounds_np[:, 1]]),
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

        # Constraints to ensure xt reaches the target set given ut
        target_set = target_sets.get_constraint_at_time_index(-1)
        target_set_A, target_set_b = target_set.get_polytope()
        constrs += [
            target_set_A @ self.dynamics.dynamics_step(xt, ut) <= target_set_b
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
            backprojection_set = constraints.LpConstraint()
            backprojection_set.is_infeasible = True
            return backprojection_set
        ranges = np.vstack(
            [-b_[self.dynamics.num_states :], b_[: self.dynamics.num_states]]
        ).T
        backprojection_set = constraints.LpConstraint(range=ranges)

        return backprojection_set


class ClosedLoopJaxUnrolledPropagator(ClosedLoopJaxPropagator):
    """Run CROWN on unrolled closed-loop dynamics, dyn(con(...dyn(con(x))))."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "rectangle"
        self.reach_fn = get_multi_step_reachable_sets_unrolled
        self.verif_fn = jax_verify.backward_crown_bound_propagation

    def get_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint, t_max: int
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        initial_set_range = initial_set.to_range()
        xt_bounds = jax_verify.IntervalBound(
            jnp.array(
                initial_set_range[..., 0].reshape(-1, self.dynamics.num_states)
            ),
            jnp.array(
                initial_set_range[..., 1].reshape(-1, self.dynamics.num_states)
            ),
        )

        num_timesteps = self.dynamics.tmax_to_num_timesteps(t_max)

        fun_to_prop = functools.partial(
            predict_future_states, self.params, self.dynamics, num_timesteps
        )
        bounds = self.verif_fn(fun_to_prop, xt_bounds)

        lbs = jnp.array([bounds[i].lower for i in range(1, num_timesteps + 1)])
        ubs = jnp.array([bounds[i].upper for i in range(1, num_timesteps + 1)])
        reachable_sets_jnp = (
            jnp.stack([lbs, ubs]).transpose(2, 1, 3, 0).squeeze()
        )

        reachable_sets_np = np.array(reachable_sets_jnp)

        reachable_sets = constraints.MultiTimestepConstraint(
            constraints=[
                constraints.LpConstraint(range=r) for r in reachable_sets_np
            ]
        )

        return reachable_sets, {}


class ClosedLoopJaxUnrolledJittedPropagator(ClosedLoopJaxPropagator):
    """(JIT-enabled) Run CROWN on unrolled closed-loop dynamics,
    dyn(con(...dyn(con(x))))."""

    def __init__(self, dynamics: dynamics.Dynamics):
        super().__init__(dynamics=dynamics)
        self.boundary_type = "rectangle"
        self.reach_fn = get_multi_step_reachable_sets_unrolled
        self.verif_fn = jax_verify.backward_crown_bound_propagation

    def get_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint, t_max: int
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        initial_set_jit = initial_set.to_jittable()
        reachable_sets_jnp, info = self.get_reachable_set_jitted(
            initial_set_jit, t_max
        )
        reachable_sets = constraints.MultiTimestepConstraint(
            constraints=[
                constraints.LpConstraint(range=r) for r in reachable_sets_jnp
            ]
        )
        return reachable_sets, info

    @functools.partial(jax.jit, static_argnames=["self", "t_max"])
    def get_reachable_set_jitted(
        self,
        initial_set_jitted: constraints.JittableSingleTimestepConstraint,
        t_max: int,
    ) -> tuple[jnp.ndarray, dict]:
        num_timesteps = self.dynamics.tmax_to_num_timesteps(t_max)

        def bound_prop_fun_(
            inp_bound: constraints.JittableSingleTimestepConstraint,
            fun_to_prop: Callable,
        ) -> jnp.ndarray:
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
        reachable_sets_jnp = bound_prop_fun_(initial_set_jitted, fun_to_prop)

        return reachable_sets_jnp, {}


@functools.partial(jax.jit, static_argnames=["fun_to_prop"])
def simplex_bound_prop_fun(
    inp_bound: constraints.JittableConstraint,
    vertices: jnp.ndarray,
    obj: jnp.ndarray,
    fun_to_prop: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    (inp_bound,) = jax_verify.src.bound_propagation.unjit_inputs(inp_bound)
    linfuns = backward_crown_bound_propagation_linfun(
        lambda x: fun_to_prop(jnp.dot(x, vertices)), inp_bound, obj=obj
    )
    return linfuns[0].lin_coeffs, linfuns[0].offset


@functools.partial(jax.jit, static_argnames=["fun_to_prop"])
def bound_prop_fun(
    inp_bound: constraints.JittableConstraint,
    obj: jnp.ndarray,
    fun_to_prop: Callable,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    (inp_bound,) = jax_verify.src.bound_propagation.unjit_inputs(inp_bound)
    linfuns = backward_crown_bound_propagation_linfun(
        fun_to_prop, inp_bound, obj=obj
    )
    return linfuns[0].lin_coeffs, linfuns[0].offset
