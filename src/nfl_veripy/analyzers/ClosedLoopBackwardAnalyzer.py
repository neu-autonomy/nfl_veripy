import ast
from copy import deepcopy
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull

import nfl_veripy.analyzers as analyzers
import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.partitioners as partitioners
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.utils import get_sampled_outputs, samples_to_range

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'


class ClosedLoopBackwardAnalyzer(analyzers.Analyzer):
    def __init__(
        self, torch_model: torch.nn.Sequential, dynamics: dynamics.Dynamics
    ):
        self.torch_model = torch_model
        self.dynamics = dynamics
        analyzers.Analyzer.__init__(self, torch_model=torch_model)

        self.true_backprojection_set_color = "k"
        self.estimated_backprojection_set_color = "tab:blue"
        self.estimated_one_step_backprojection_set_color = "tab:orange"
        self.estimated_backprojection_partitioned_set_color = "tab:gray"
        self.backreachable_set_color = "tab:cyan"
        self.target_set_color = "tab:red"
        self.initial_set_color = "k"

        self.true_backprojection_set_zorder = 3
        self.estimated_backprojection_set_zorder = 2
        self.estimated_one_step_backprojection_set_zorder = -1
        self.estimated_backprojection_partitioned_set_zorder = 5
        self.backreachable_set_zorder = -1
        self.target_set_zorder = 1
        self.initial_set_zorder = 1

        self.true_backprojection_set_linestyle = "-"
        self.estimated_backprojection_set_linestyle = "-"
        self.estimated_one_step_backprojection_set_linestyle = "-"
        self.estimated_backprojection_partitioned_set_linestyle = "-"
        self.backreachable_set_linestyle = "--"
        self.target_set_linestyle = "-"

        # self.reachable_set_color = 'tab:green'
        # self.reachable_set_zorder = 4
        # self.initial_set_color = 'tab:gray'
        # self.initial_set_zorder = 1

    @property
    def partitioner_dict(self):
        return partitioners.partitioner_dict

    @property
    def propagator_dict(self):
        return propagators.propagator_dict

    def instantiate_partitioner(
        self, partitioner: str, hyperparams: dict[str, Any]
    ) -> partitioners.ClosedLoopPartitioner:
        return partitioners.partitioner_dict[partitioner](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def instantiate_propagator(
        self, propagator: str, hyperparams: dict[str, Any]
    ) -> propagators.ClosedLoopPropagator:
        return propagators.propagator_dict[propagator](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def get_one_step_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        backprojection_set, info = (
            self.partitioner.get_one_step_backprojection_set(
                target_set, self.propagator, overapprox=overapprox
            )
        )
        return backprojection_set, info

    def get_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        backprojection_set, info = self.partitioner.get_backprojection_set(
            target_set, self.propagator, t_max, overapprox=overapprox
        )
        return backprojection_set, info

    def get_N_step_backprojection_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
        num_partitions=None,
        overapprox: bool = False,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        backprojection_set, info = (
            self.partitioner.get_N_step_backprojection_set(
                target_set,
                self.propagator,
                t_max,
                num_partitions=num_partitions,
                overapprox=overapprox,
            )
        )
        return backprojection_set, info

    def get_backprojection_error(
        self,
        target_set: constraints.SingleTimestepConstraint,
        backprojection_sets: constraints.MultiTimestepConstraint,
        t_max: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.partitioner.get_backprojection_error(
            target_set,
            backprojection_sets,
            self.propagator,
            t_max,
            backreachable_sets=None,
        )

    def visualize(  # type: ignore
        self,
        backprojection_sets: constraints.MultiTimestepConstraint,
        target_set: constraints.SingleTimestepConstraint,
        info: dict,
        initial_constraint: Optional[
            constraints.SingleTimestepConstraint
        ] = None,
        show: bool = True,
        show_samples: bool = False,
        show_samples_from_cells: bool = False,
        show_trajectories: bool = False,
        show_convex_hulls: bool = False,
        aspect: str = "auto",
        axis_labels: list = [],
        axis_dims: list = [],
        plot_lims: Optional[str] = None,
        controller_name: Optional[str] = None,
        show_BReach: bool = False,
    ) -> None:
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(
            backprojection_sets.get_constraint_at_time_index(-1),
            backprojection_sets.get_t_max(),
            self.propagator,
            show_samples=show_samples,
            show_samples_from_cells=show_samples_from_cells,
            axis_dims=axis_dims,
            axis_labels=axis_labels,
            aspect=aspect,
            initial_set_color=self.estimated_backprojection_set_color,
            initial_set_zorder=self.estimated_backprojection_set_zorder,
            extra_constraint=initial_constraint,
            extra_set_color=self.initial_set_color,
            extra_set_zorder=self.initial_set_zorder,
            controller_name=controller_name,
        )

        self.visualize_single_set(
            backprojection_sets,
            target_set,
            initial_constraint=initial_constraint,
            show_samples=show_samples,
            show_trajectories=show_trajectories,
            show_convex_hulls=show_convex_hulls,
            show=show,
            aspect=aspect,
            plot_lims=plot_lims,
            axis_dims=axis_dims,
            show_BReach=show_BReach,
            **info
        )
        self.partitioner.animate_fig.tight_layout()

        if plot_lims is not None:
            plot_lims_arr = np.array(ast.literal_eval(plot_lims))
            plt.xlim(plot_lims_arr[0])
            plt.ylim(plot_lims_arr[1])

        if info.get("save_name", None) is not None:
            plt.savefig(info["save_name"])

        plt.gca().autoscale()

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_single_set(
        self,
        backprojection_sets: constraints.MultiTimestepConstraint,
        target_set: constraints.SingleTimestepConstraint,
        initial_constraint: Optional[
            constraints.SingleTimestepConstraint
        ] = None,
        show: bool = True,
        show_samples: bool = False,
        show_trajectories: bool = False,
        show_convex_hulls: bool = False,
        aspect: str = "auto",
        axis_labels: list = [],
        plot_lims: Optional[str] = None,
        inputs_to_highlight: Optional[list[dict]] = None,
        show_BReach: bool = False,
        **kwargs
    ) -> None:
        rects = backprojection_sets.plot(
            self.partitioner.animate_axes,
            self.partitioner.axis_dims,
            self.estimated_backprojection_set_color,
            zorder=self.estimated_backprojection_set_zorder,
            linewidth=self.partitioner.linewidth,
            plot_2d=self.partitioner.plot_2d,
        )
        self.partitioner.default_patches += rects
        for cell in backprojection_sets.cells:
            rects = cell.plot(
                self.partitioner.animate_axes,
                self.partitioner.axis_dims,
                self.estimated_backprojection_set_color,
                zorder=self.estimated_backprojection_set_zorder,
                linewidth=self.partitioner.linewidth,
                plot_2d=self.partitioner.plot_2d,
            )
            self.partitioner.default_patches += rects

        # Show the target set
        self.plot_target_set(
            target_set,
            color=self.target_set_color,
            zorder=self.target_set_zorder,
            linestyle=self.target_set_linestyle,
        )

        # Show the "true" N-Step backprojection set as a convex hull
        backreachable_set = kwargs["per_timestep"][-1]["backreachable_set"]
        t_max = backprojection_sets.get_t_max()
        if show_convex_hulls:
            self.plot_true_backprojection_sets(
                backreachable_set,
                target_set,
                t_max=t_max,
                color=self.true_backprojection_set_color,
                zorder=self.true_backprojection_set_zorder,
                linestyle=self.true_backprojection_set_linestyle,
                show_samples=False,
            )

        # Sketchy workaround to trajectories not showing up
        if show_trajectories and initial_constraint is not None:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                initial_constraint,
                ax=self.partitioner.animate_axes,
                controller=self.propagator.network,
            )

    def get_sampled_outputs(self, input_range, N=1000):
        return get_sampled_outputs(input_range, self.propagator, N=N)

    def get_sampled_output_range(
        self, input_constraint, t_max=5, num_samples=1000
    ):
        return self.partitioner.get_sampled_out_range(
            input_constraint, self.propagator, t_max, num_samples
        )

    def get_output_range(self, input_constraint, output_constraint):
        return self.partitioner.get_output_range(
            input_constraint, output_constraint
        )

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

    def get_error(self, input_constraint, output_constraint, t_max):
        return self.partitioner.get_error(
            input_constraint, output_constraint, self.propagator, t_max
        )

    def plot_backreachable_set(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        color: str = "cyan",
        zorder: Optional[int] = None,
        linestyle: str = "-",
    ) -> None:
        self.partitioner.plot_reachable_sets(
            backreachable_set,
            self.partitioner.axis_dims,
            reachable_set_color=color,
            reachable_set_zorder=zorder,
            reachable_set_ls=linestyle,
        )

    def plot_target_set(
        self,
        target_set: constraints.SingleTimestepConstraint,
        color: str = "cyan",
        zorder: Optional[int] = None,
        linestyle: str = "-",
        linewidth: float = 2.5,
    ) -> None:
        self.partitioner.plot_reachable_sets(
            target_set,
            self.partitioner.axis_dims,
            reachable_set_color=color,
            reachable_set_zorder=zorder,
            reachable_set_ls=linestyle,
            reachable_set_lw=linewidth,
        )

    def plot_tightened_backprojection_set(
        self,
        tightened_set: constraints.SingleTimestepConstraint,
        color: str = "darkred",
        zorder: Optional[int] = None,
        linestyle: str = "-",
    ) -> None:
        self.partitioner.plot_reachable_sets(
            tightened_set,
            self.partitioner.axis_dims,
            reachable_set_color=color,
            reachable_set_zorder=zorder,
            reachable_set_ls=linestyle,
        )

    def plot_backprojection_set(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        target_set: constraints.SingleTimestepConstraint,
        show_samples: bool = False,
        color: str = "g",
        zorder: Optional[int] = None,
        linestyle: str = "-",
    ) -> None:
        # Sample a bunch of pts from our "true" backreachable set
        # (it's actually the tightest axis-aligned rectangle around the
        # set) and run them forward 1 step in time under the NN policy
        xt_samples_from_backreachable_set, xt1_from_those_samples = (
            self.partitioner.dynamics.get_state_and_next_state_samples(
                backreachable_set,
                num_samples=1e5,
                controller=self.propagator.network,
            )
        )

        # Find which of the xt+1 points actually end up in the target set
        target_set_A, target_set_b = target_set.get_polytope()
        within_constraint_inds = np.where(
            np.all(
                (
                    np.dot(target_set_A, xt1_from_those_samples.T)
                    - np.expand_dims(target_set_b, axis=-1)
                )
                <= 0,
                axis=0,
            )
        )
        xt_samples_inside_backprojection_set = (
            xt_samples_from_backreachable_set[(within_constraint_inds)]
        )

        if show_samples:
            xt1_from_those_samples_ = xt1_from_those_samples[
                (within_constraint_inds)
            ]

            # Show samples from inside the backprojection set and their
            # futures under the NN (should end in target set)
            self.partitioner.dynamics.show_samples(
                None,
                None,
                ax=self.partitioner.animate_axes,
                controller=None,
                input_dims=self.partitioner.axis_dims,
                zorder=1,
                xs=np.dstack(
                    [
                        xt_samples_inside_backprojection_set,
                        xt1_from_those_samples_,
                    ]
                ).transpose(0, 2, 1),
                colors=None,
            )

        # Compute and draw a convex hull around all the backprojection set
        # samples. This is our "true" backprojection set -- but...
        # it is sampling-based so that is an under-approximation,
        # and it is a convex hull, so that is an over-approximation,
        # and it is computed only for one step, so that's an over-approximation
        _ = plot_convex_hull(
            xt_samples_inside_backprojection_set,
            dims=self.partitioner.axis_dims,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            zorder=zorder,
            label="Backprojection Set (True)",
            axes=self.partitioner.animate_axes,
        )
        # self.default_lines[self.output_axis].append(conv_hull_line[0])

    def plot_true_backprojection_sets(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
        show_samples: bool = False,
        color: str = "g",
        zorder: Optional[int] = None,
        linestyle: str = "-",
    ) -> None:
        # Sample a bunch of pts from our "true" backreachable set
        # (it's actually the tightest axis-aligned rectangle around the
        # backreachable set) and run them forward t_max steps in time under
        # the NN policy
        x_samples_inside_backprojection_set = (
            self.dynamics.get_true_backprojection_set(
                backreachable_set,
                target_set,
                t_max=t_max,
                controller=self.propagator.network,
            )
        )

        if show_samples:
            # xt1_from_those_samples_ = xt1_from_those_samples[
            #     (within_constraint_inds)
            # ]

            # Show samples from inside the backprojection set and their
            # futures under the NN (should end in target set)
            self.partitioner.dynamics.show_samples(
                None,
                None,
                ax=self.partitioner.animate_axes,
                controller=None,
                input_dims=self.partitioner.axis_dims,
                zorder=1,
                xs=x_samples_inside_backprojection_set,
                colors=None,
            )
            print(self.partitioner.animate_axes.get_ylabel())

        # Compute and draw a convex hull around all the backprojection set
        # samples. This is our "true" backprojection set -- but...
        # it is sampling-based so that is an under-approximation,
        # and it is a convex hull, so that is an over-approximation.
        for t in range(t_max):
            _ = plot_convex_hull(
                x_samples_inside_backprojection_set[:, t, :],
                dims=self.partitioner.axis_dims,
                color=color,
                linewidth=2,
                linestyle=linestyle,
                zorder=zorder,
                label="Backprojection Set (True)",
                axes=self.partitioner.animate_axes,
            )
        # self.default_lines[self.output_axis].append(conv_hull_line[0])

    def plot_relaxed_sequence(
        self, start_region, bp_sets, crown_bounds, target_set, marker="o"
    ):
        bp_sets = deepcopy(bp_sets)
        bp_sets.reverse()
        bp_sets.append(target_set)
        sequences = (
            self.partitioner.dynamics.get_relaxed_backprojection_samples(
                start_region, bp_sets, crown_bounds, target_set
            )
        )

        ax = (self.partitioner.animate_axes,)
        x = []
        y = []
        for seq in [sequences[0]]:
            for point in seq:
                x.append(point[0])
                y.append(point[1])

        # import pdb; pdb.set_trace()
        ax[0].scatter(x, y, s=40, zorder=15, c="k", marker=marker)


def plot_convex_hull(
    samples: np.ndarray,
    dims: list,
    color: str,
    linewidth: float,
    linestyle: str,
    zorder: Optional[int],
    label: str,
    axes: matplotlib.axes,
) -> matplotlib.lines.Line2D:
    hull = ConvexHull(samples[..., dims].squeeze())
    line = axes.plot(
        np.append(
            samples[hull.vertices][..., dims[0]],
            samples[hull.vertices[0]][..., dims[0]],
        ),
        np.append(
            samples[hull.vertices][..., dims[1]],
            samples[hull.vertices[0]][..., dims[1]],
        ),
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        label=label,
    )
    return line
