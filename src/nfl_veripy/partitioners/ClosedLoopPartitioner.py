from copy import deepcopy
from typing import Any, Optional, Union

import cvxpy as cp
import numpy as np
import pypoman
from scipy.spatial import ConvexHull

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.optimization_utils import optimize_over_all_states
from nfl_veripy.utils.utils import range_to_polytope


class ClosedLoopPartitioner:
    def __init__(self, dynamics: dynamics.Dynamics):
        self.dynamics = dynamics

    def get_one_step_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        reachable_set, info = propagator.get_one_step_reachable_set(
            initial_set
        )
        return reachable_set, info

    def get_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        reachable_set, info = propagator.get_reachable_set(initial_set, t_max)
        return reachable_set, info

        # # TODO: this is repeated from UniformPartitioner...
        # # might be more efficient to directly return from propagator?
        # _ = reachable_set.add_cell(reachable_set_this_cell)

        # return reachable_set, info

    def get_error(  # type: ignore
        self,
        initial_set: constraints.SingleTimestepConstraint,
        reachable_sets: constraints.MultiTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        errors = []

        if isinstance(initial_set, constraints.LpConstraint):
            estimated_reachable_set_ranges = reachable_sets.range
            true_reachable_set_ranges = self.get_sampled_out_range(
                initial_set, propagator, t_max, num_samples=1000
            )
            num_steps = true_reachable_set_ranges.shape[0]
            for t in range(num_steps):
                true_area = np.product(
                    true_reachable_set_ranges[t][..., 1]
                    - true_reachable_set_ranges[t][..., 0]
                )
                estimated_area = np.product(
                    estimated_reachable_set_ranges[t][..., 1]
                    - estimated_reachable_set_ranges[t][..., 0]
                )
                errors.append((estimated_area - true_area) / true_area)
        elif isinstance(initial_set, constraints.PolytopeConstraint):
            # Note: This compares the estimated polytope
            # with the "best" polytope with those facets.
            # There could be a much better polytope with lots of facets.
            true_verts = self.get_sampled_out_range(
                initial_set,
                propagator,
                t_max,
                num_samples=1000,
                output_constraint=reachable_sets,
            )

            num_steps = reachable_sets.get_t_max()
            from scipy.spatial import ConvexHull

            for t in range(num_steps):
                true_hull = ConvexHull(true_verts[:, t + 1, :])
                true_area = true_hull.volume
                A, b = reachable_sets.get_constraint_at_time_index(
                    t
                ).get_polytope()
                estimated_verts = pypoman.polygon.compute_polygon_hull(A, b)
                estimated_hull = ConvexHull(estimated_verts)
                estimated_area = estimated_hull.volume
                errors.append((estimated_area - true_area) / true_area)
        else:
            raise ValueError(
                "initial_set and reachable_sets need to both be Lp or both be"
                " Polytope."
            )
        final_error = errors[-1]
        avg_error = np.mean(errors)
        return final_error, avg_error, np.array(errors)

    def get_sampled_out_range(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int = 5,
        num_samples: int = 1000,
        output_constraint: Optional[constraints.Constraint] = None,
    ) -> np.ndarray:
        # TODO: change output_constraint to better name
        return self.dynamics.get_sampled_output_range(
            initial_set,
            t_max,
            num_samples,
            controller=propagator.network,
            output_constraint=output_constraint,
        )

    def get_sampled_out_range_guidance(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int = 5,
        num_samples: int = 1000,
    ) -> np.ndarray:
        # Duplicate of get_sampled_out_range, but called during partitioning
        return self.get_sampled_out_range(
            initial_set, propagator, t_max=t_max, num_samples=num_samples
        )

    def get_one_step_backreachable_set(
        self,
        target_sets: Union[
            constraints.SingleTimestepConstraint,
            constraints.MultiTimestepConstraint,
        ],
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        # Given a target_set, compute the backreachable_set
        # that ensures that starting from within the backreachable_set
        # will lead to a state within the target_set
        info = {}  # type: dict[str, Any]
        # if collected_input_constraints is None:
        #     collected_input_constraints = [input_constraint]

        # Extract elementwise bounds on xt1 from the lp-ball or polytope
        # constraint
        A_target, b_target = target_sets.get_constraint_at_time_index(
            -1
        ).get_polytope()

        """
        Step 1:
        Find backreachable set: all the xt for which there is
        some u in U that leads to a state xt1 in output_constraint
        """

        if self.dynamics.u_limits is None:
            print(
                "self.dynamics.u_limits is None ==>                 The"
                " backreachable set is probably the whole state space.        "
                "         Giving up."
            )
            raise NotImplementedError
        else:
            u_min = self.dynamics.u_limits[:, 0]
            u_max = self.dynamics.u_limits[:, 1]

        num_states = self.dynamics.num_states
        num_control_inputs = self.dynamics.num_inputs

        xt = cp.Variable(num_states)
        ut = cp.Variable(num_control_inputs)

        # For each dimension of the output constraint (facet/lp-dimension):
        # compute a bound of the NN output using the pre-computed matrices
        constrs = []
        constrs += [u_min <= ut]
        constrs += [ut <= u_max]

        # Included state limits to reduce size of backreachable sets by
        # eliminating states that are not physically possible
        # (e.g., maximum velocities)
        if self.dynamics.x_limits is not None:
            for state in self.dynamics.x_limits:
                constrs += [self.dynamics.x_limits[state][0] <= xt[state]]
                constrs += [xt[state] <= self.dynamics.x_limits[state][1]]

        constrs += [A_target @ self.dynamics.dynamics_step(xt, ut) <= b_target]

        b, status = optimize_over_all_states(xt, constrs)
        ranges = np.vstack([-b[num_states:], b[:num_states]]).T

        backreachable_set = constraints.LpConstraint(range=ranges)
        info["backreachable_set"] = backreachable_set
        info["target_sets"] = target_sets

        return backreachable_set, info

    def get_one_step_backprojection_set(
        self,
        target_sets: constraints.MultiTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        overapprox: bool = False,
    ) -> tuple[Optional[constraints.SingleTimestepConstraint], dict]:
        backreachable_set, info = self.get_one_step_backreachable_set(
            target_sets
        )
        info["backreachable_set"] = backreachable_set

        backprojection_set, _ = propagator.get_one_step_backprojection_set(
            backreachable_set,
            target_sets,
            overapprox=overapprox,
        )

        # if overapprox:
        #     # These will be used to further backproject this set in time
        #     backprojection_set.crown_matrices = get_crown_matrices(
        #         propagator,
        #         backprojection_set,
        #         self.dynamics.num_inputs,
        #         self.dynamics.sensor_noise
        #     )

        return backprojection_set, info

    """
    Inputs:
    - target_set: describes goal/avoid set at t=t_max
    - propagator:
    - t_max: how many timesteps to backproject
    - num_partitions: number of splits per dimension in each backreachable set
    - overapprox: bool
        True = compute outer bounds of BP sets (for collision avoidance)
        False = computer inner bounds of BP sets (for goal arrival)

    Returns:
    - backprojection_sets: [BP_{-1}, ..., BP_{-t_max}]
          i.e., [ set of states that will get to goal/avoid set in 1 step,
                  ...,
                  set of states that will get to goal/avoid set in t_max steps
                ]
    - info: TODO
    """

    def get_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        # Initialize data structures to hold results
        backprojection_sets = (
            constraints.create_empty_multi_timestep_constraint(
                propagator.boundary_type
            )
        )
        info = {"per_timestep": []}  # type: dict[str, Any]

        # Run one step of backprojection analysis
        backprojection_set_this_timestep, info_this_timestep = (
            self.get_one_step_backprojection_set(
                target_set.to_multistep_constraint(),
                propagator,
                overapprox=overapprox,
            )
        )

        # Store that step's results
        assert backprojection_set_this_timestep is not None
        backprojection_sets = backprojection_sets.add_timestep_constraint(
            backprojection_set_this_timestep
        )
        info["per_timestep"].append(info_this_timestep)

        if overapprox:
            for i in np.arange(
                0 + propagator.dynamics.dt + 1e-10,
                t_max,
                propagator.dynamics.dt,
            ):
                # Run one step of backprojection analysis
                backprojection_set_this_timestep, info_this_timestep = (
                    self.get_one_step_backprojection_set(
                        target_set.add_timestep_constraint(
                            backprojection_sets
                        ),
                        propagator,
                        overapprox=overapprox,
                    )
                )
                assert backprojection_set_this_timestep is not None
                backprojection_sets = (
                    backprojection_sets.add_timestep_constraint(
                        backprojection_set_this_timestep
                    )
                )
                info["per_timestep"].append(info_this_timestep)
        else:
            for i in np.arange(
                0 + propagator.dynamics.dt + 1e-10,
                t_max,
                propagator.dynamics.dt,
            ):
                # TODO: Support N-step backprojection in the
                # under-approximation case
                raise NotImplementedError

        return backprojection_sets, info

    def get_backprojection_error(
        self,
        target_set: constraints.SingleTimestepConstraint,
        backprojection_sets: constraints.MultiTimestepConstraint,
        propagator: propagators.ClosedLoopPropagator,
        t_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Note: This almost certainly got messed up in the merge (3/24/23)

        errors = []

        true_verts_reversed = self.dynamics.get_true_backprojection_set(
            backprojection_sets.get_constraint_at_time_index(-1),
            target_set,
            t_max,
            controller=propagator.network,
        )
        true_verts = np.flip(true_verts_reversed, axis=1)
        num_steps = backprojection_sets.get_t_max()

        for t in range(num_steps):
            x_min = np.min(true_verts[:, t + 1, :], axis=0)
            x_max = np.max(true_verts[:, t + 1, :], axis=0)

        if isinstance(target_set, constraints.LpConstraint):
            Ats, bts = range_to_polytope(target_set.range)
            true_verts_reversed = self.dynamics.get_true_backprojection_set(
                backprojection_sets.get_constraint_at_time_index(-1),
                target_set,
                t_max,
                controller=propagator.network,
                num_samples=1e8,
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            num_steps = backprojection_sets.get_t_max()

            x_range = x_max - x_min
            true_area = np.prod(x_range)

            estimated_area = backprojection_sets.get_constraint_at_time_index(
                t
            ).get_area()

            # true_hull = ConvexHull(true_verts[:, t+1, :])
            # true_area = true_hull.volume

            Abp, bbp = backprojection_sets.get_constraint_at_time_index(
                t
            ).get_polytope()
            estimated_verts = pypoman.polygon.compute_polygon_hull(Abp, bbp)
            estimated_hull = ConvexHull(estimated_verts)
            estimated_area = estimated_hull.volume

            # print(f"estimated: estimated_area --- true: {true_area}")
            # print(
            #     "estimated range: {} --- true range: {}".format(
            #         backprojection_sets[t].range, x_range
            #     )
            # )

            errors.append((estimated_area - true_area) / true_area)
        elif isinstance(target_set, constraints.RotatedLpConstraint):
            true_verts_reversed = self.dynamics.get_true_backprojection_set(
                # backreachable_sets[-1],
                target_set,
                t_max,
                controller=propagator.network,
                num_samples=1e8,
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            num_steps = len(backprojection_sets)

            for t in range(num_steps):
                # x_min = np.min(true_verts[:,t+1,:], axis=0)
                # x_max = np.max(true_verts[:,t+1,:], axis=0)

                # x_range = x_max-x_min
                # true_area = np.prod(x_range)
                true_hull = ConvexHull(true_verts[:, t + 1, :])
                true_area = true_hull.volume

                # true_hull = ConvexHull(true_verts[:, t+1, :])
                # true_area = true_hull.volume

                # Abp, bbp = range_to_polytope(backprojection_sets[t].range)
                # estimated_verts = (
                #   pypoman.polygon.compute_polygon_hull(Abp, bbp)
                # )
                estimated_hull = ConvexHull(backprojection_sets[t].vertices)
                estimated_area = estimated_hull.volume

                # print(
                #     "estimated: {} --- true: {}".format(
                #         estimated_area, true_area
                #     )
                # )
                # print(
                #     "estimated range: {} --- true range: {}".format(
                #         backprojection_sets[t].range, x_range
                #     )
                # )

                errors.append((estimated_area - true_area) / true_area)
        else:
            # This implementation should actually be moved to Lp constraint

            # Note: This compares the estimated polytope
            # with the "best" polytope with those facets.
            # There could be a much better polytope with lots of facets.
            true_verts_reversed = self.dynamics.get_true_backprojection_set(
                backprojection_sets.get_constraint_at_time_index(-1),
                target_set,
                t_max,
                controller=propagator.network,
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            num_steps = backprojection_sets.get_t_max()

            for t in range(num_steps):
                x_min = np.min(true_verts[:, t + 1, :], axis=0)
                x_max = np.max(true_verts[:, t + 1, :], axis=0)

                x_range = x_max - x_min
                true_area = np.prod(x_range)

                Abp, bbp = backprojection_sets.get_constraint_at_time_index(
                    t
                ).get_polytope()
                estimated_verts = pypoman.polygon.compute_polygon_hull(
                    Abp, bbp
                )
                estimated_hull = ConvexHull(estimated_verts)
                estimated_area = estimated_hull.volume

                errors.append((estimated_area - true_area) / true_area)

        final_error = errors[-1]
        avg_error = np.mean(errors)
        return final_error, avg_error, np.array(errors)

    def get_N_step_backprojection_set(
        self,
        output_constraint,
        input_constraint,
        propagator,
        t_max,
        num_partitions=None,
        overapprox=False,
        heuristic="guided",
        all_lps=False,
        slow_cvxpy=False,
    ):
        input_constraint_, info = propagator.get_N_step_backprojection_set(
            output_constraint,
            deepcopy(input_constraint),
            t_max,
            num_partitions=num_partitions,
            overapprox=overapprox,
            heuristic=heuristic,
            all_lps=all_lps,
            slow_cvxpy=slow_cvxpy,
        )
        input_constraint = input_constraint_.copy()

        return input_constraint, info
