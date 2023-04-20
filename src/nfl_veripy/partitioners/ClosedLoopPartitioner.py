import os
from copy import deepcopy
from typing import Any, Optional, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pypoman
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.optimization_utils import optimize_over_all_states
from nfl_veripy.utils.utils import range_to_polytope

from .Partitioner import Partitioner


class ClosedLoopPartitioner(Partitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
        make_animation: bool = False,
        show_animation: bool = False,
    ):
        super().__init__()
        self.dynamics = dynamics

        # Animation-related flags
        self.make_animation = make_animation
        self.show_animation = show_animation
        self.tmp_animation_save_dir = "{}/../../results/tmp_animation/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.animation_save_dir = "{}/../../results/animations/".format(
            os.path.dirname(os.path.abspath(__file__))
        )

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

        if isinstance(initial_set, constraints.LpConstraint) and isinstance(
            reachable_sets, constraints.MultiTimestepLpConstraint
        ):
            estimated_reachable_set_ranges = reachable_sets.to_range()
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
        elif isinstance(
            initial_set, constraints.PolytopeConstraint
        ) and isinstance(
            reachable_sets, constraints.MultiTimestepPolytopeConstraint
        ):
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
                if reachable_sets.A is None or reachable_sets.b is None:
                    raise ValueError(
                        "Can't compute polygon hull because reachable_sets has"
                        " Nones in it."
                    )
                estimated_verts = pypoman.polygon.compute_polygon_hull(
                    reachable_sets.A[t], reachable_sets.b[t]
                )
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
        output_constraint: Optional[
            constraints.SingleTimestepConstraint
        ] = None,
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

    def setup_visualization(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        t_max: int,
        propagator: propagators.ClosedLoopPropagator,
        show_samples: bool = True,
        show_samples_from_cells: bool = True,
        show_trajectories: bool = False,
        axis_labels: Optional[list] = None,
        axis_dims: Optional[list] = None,
        aspect: str = "auto",
        initial_set_color: Optional[str] = None,
        initial_set_zorder: Optional[int] = None,
        extra_set_color: Optional[str] = None,
        extra_set_zorder: Optional[int] = None,
        sample_zorder: Optional[int] = None,
        sample_colors: Optional[str] = None,
        extra_constraint: Optional[
            constraints.SingleTimestepConstraint
        ] = None,
        plot_lims: Optional[list] = None,
        controller_name: Optional[str] = None,
    ) -> None:
        self.default_patches = []
        self.default_lines = []

        self.axis_dims = axis_dims

        if len(axis_dims) == 2:
            projection = None
            self.plot_2d = True
            self.linewidth = 2
        elif len(axis_dims) == 3:
            projection = "3d"
            self.plot_2d = False
            self.linewidth = 1
            aspect = "auto"

        self.animate_fig, self.animate_axes = plt.subplots(
            1, 1, subplot_kw=dict(projection=projection)
        )
        if controller_name is not None:
            from nfl_veripy.utils.controller_generation import (
                display_ground_robot_control_field,
            )

            display_ground_robot_control_field(
                name=controller_name, ax=self.animate_axes
            )

        # if controller_name is not None:
        #     from nfl_veripy.utils.controller_generation import (
        #         display_ground_robot_DI_control_field,
        #     )

        #     display_ground_robot_DI_control_field(
        #         name=controller_name, ax=self.animate_axes
        #     )

        self.animate_axes.set_aspect(aspect)

        if show_samples:
            self.dynamics.show_samples(
                t_max * self.dynamics.dt,
                initial_set,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=axis_dims,
                zorder=sample_zorder,
                colors=sample_colors,
            )

        if show_samples_from_cells:
            for initial_set_cell in initial_set.cells:
                self.dynamics.show_samples(
                    t_max * self.dynamics.dt,
                    initial_set_cell,
                    ax=self.animate_axes,
                    controller=propagator.network,
                    input_dims=axis_dims,
                    zorder=sample_zorder,
                    colors=sample_colors,
                )

        if show_trajectories:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                initial_set,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=axis_dims,
                zorder=sample_zorder,
                colors=sample_colors,
            )

        self.animate_axes.set_xlabel(axis_labels[0])
        self.animate_axes.set_ylabel(axis_labels[1])
        if not self.plot_2d:
            self.animate_axes.set_zlabel(axis_labels[2])

        # Plot the initial state set's boundaries
        if initial_set_color is None:
            initial_set_color = "tab:grey"
        rect = initial_set.plot(
            self.animate_axes,
            axis_dims,
            initial_set_color,
            zorder=initial_set_zorder,
            linewidth=self.linewidth,
            plot_2d=self.plot_2d,
        )
        self.default_patches += rect

        if show_samples_from_cells:
            for cell in initial_set.cells:
                rect = initial_set_cell.plot(
                    self.animate_axes,
                    axis_dims,
                    initial_set_color,
                    zorder=initial_set_zorder,
                    linewidth=self.linewidth,
                    plot_2d=self.plot_2d,
                )
                self.default_patches += rect

        if extra_set_color is None:
            extra_set_color = "tab:red"
        # if extra_constraint[0] is not None:
        #     for i in range(len(extra_constraint)):
        #         rect = extra_constraint[i].plot(
        #             self.animate_axes,
        #             input_dims,
        #             extra_set_color,
        #             zorder=extra_set_zorder,
        #             linewidth=self.linewidth,
        #             plot_2d=self.plot_2d,
        #         )
        #         self.default_patches += rect

    def visualize(  # type: ignore
        self,
        M: list,
        interior_M: list,
        reachable_sets: constraints.MultiTimestepConstraint,
        iteration: int = 0,
        title: Optional[str] = None,
        reachable_set_color: Optional[str] = None,
        reachable_set_zorder: Optional[int] = None,
        reachable_set_ls: Optional[str] = None,
        dont_tighten_layout: bool = False,
        plot_lims: Optional[str] = None,
    ) -> None:
        # Bring forward whatever default items should be in the plot
        # (e.g., MC samples, initial state set boundaries)
        for item in self.default_patches + self.default_lines:
            if isinstance(item, Patch):
                self.animate_axes.add_patch(item)
            elif isinstance(item, Line2D):
                self.animate_axes.add_line(item)

        self.plot_reachable_sets(
            reachable_sets,
            self.axis_dims,
            reachable_set_color=reachable_set_color,
            reachable_set_zorder=reachable_set_zorder,
            reachable_set_ls=reachable_set_ls,
        )

        if plot_lims is not None:
            import ast

            plot_lims_arr = np.array(ast.literal_eval(plot_lims))
            plt.xlim(plot_lims_arr[0])
            plt.ylim(plot_lims_arr[1])

        # Do auxiliary stuff to make sure animations look nice
        if title is not None:
            plt.suptitle(title)

        if (iteration == 0 or iteration == -1) and not dont_tighten_layout:
            plt.tight_layout()

        if self.show_animation:
            plt.pause(0.01)

        if self.make_animation and iteration is not None:
            os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
            filename = self.get_tmp_animation_filename(iteration)
            plt.savefig(filename)

        if self.make_animation and not self.plot_2d:
            # Make an animated 3d view
            os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
            for i, angle in enumerate(range(-100, 0, 2)):
                self.animate_axes.view_init(30, angle)
                filename = self.get_tmp_animation_filename(i)
                plt.savefig(filename)
            self.compile_animation(i, delete_files=True, duration=0.2)

    def plot_reachable_sets(
        self,
        constraint: constraints.MultiTimestepConstraint,
        dims: list,
        reachable_set_color: Optional[str] = None,
        reachable_set_zorder: Optional[int] = None,
        reachable_set_ls: Optional[str] = None,
        reachable_set_lw: Optional[int] = None,
    ):
        if reachable_set_color is None:
            reachable_set_color = "tab:blue"
        if reachable_set_ls is None:
            reachable_set_ls = "-"
        if reachable_set_lw is None:
            reachable_set_lw = self.linewidth
        fc_color = "None"
        constraint.plot(
            self.animate_axes,
            dims,
            reachable_set_color,
            fc_color=fc_color,
            zorder=reachable_set_zorder,
            plot_2d=self.plot_2d,
            linewidth=reachable_set_lw,
            ls=reachable_set_ls,
        )

    def plot_partition(self, constraint, dims, color):
        # This if shouldn't really be necessary -- someone is calling
        # self.plot_partitions with something other than a
        # (constraint, ___) element in M?
        if isinstance(constraint, np.ndarray):
            constraint = constraints.LpConstraint(range=constraint)

        constraint.plot(
            self.animate_axes, dims, color, linewidth=1, plot_2d=self.plot_2d
        )

    def plot_partitions(
        self,
        M: list[tuple[constraints.SingleTimestepConstraint, np.ndarray]],
        dims: list,
    ) -> None:
        # first = True
        for input_constraint, output_range in M:
            # Next state constraint of that cell
            output_constraint_ = constraints.LpConstraint(range=output_range)
            self.plot_partition(output_constraint_, dims, "grey")

            # Initial state constraint of that cell
            self.plot_partition(input_constraint, dims, "tab:red")

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
        # if self.dynamics.x_limits is not None:
        #     x_llim = self.dynamics.x_limits[:, 0]
        #     x_ulim = self.dynamics.x_limits[:, 1]
        #     constrs += [x_llim <= xt]
        #     constrs += [xt <= x_ulim]
        #     # Also constrain the future state to be within the state limits
        #     constrs += [self.dynamics.dynamics_step(xt,ut) <= x_ulim]
        #     constrs += [self.dynamics.dynamics_step(xt,ut) >= x_llim]

        # if self.dynamics.x_limits is not None:
        #     for state in self.dynamics.x_limits:
        #         constrs += [self.dynamics.x_limits[state][0] <= xt[state]]
        #         constrs += [xt[state] <= self.dynamics.x_limits[state][1]]

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
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
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
                backprojection_sets[-1],
                target_set,
                t_max,
                controller=propagator.network,
                num_samples=1e8,
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            num_steps = len(backprojection_sets)

            x_range = x_max - x_min
            true_area = np.prod(x_range)

            estimated_area = backprojection_sets.get_constraint_at_time_index(
                t
            ).get_area()

            # true_hull = ConvexHull(true_verts[:, t+1, :])
            # true_area = true_hull.volume

            Abp, bbp = range_to_polytope(backprojection_sets[t].range)
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
                backprojection_sets[-1],
                target_set,
                t_max,
                controller=propagator.network,
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            num_steps = len(backprojection_sets)

            for t in range(num_steps):
                x_min = np.min(true_verts[:, t + 1, :], axis=0)
                x_max = np.max(true_verts[:, t + 1, :], axis=0)

                x_range = x_max - x_min
                true_area = np.prod(x_range)

                estimated_verts = pypoman.polygon.compute_polygon_hull(
                    backprojection_sets[t].A[0], backprojection_sets[t].b[0]
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
