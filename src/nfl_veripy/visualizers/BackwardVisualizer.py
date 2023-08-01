import ast
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.controller_generation import (
    display_ground_robot_control_field,
)


class BackwardVisualizer:
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        self.dynamics = dynamics

        # Animation-related flags
        self.make_animation: bool = False
        self.show_animation: bool = False
        self.tmp_animation_save_dir = "{}/../../results/tmp_animation/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.animation_save_dir = "{}/../../results/animations/".format(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.initial_state_range: Optional[np.ndarray] = None

        self.sample_zorder: int = 1
        self.sample_colors: Optional[str] = None

        self.show_samples_from_cells: bool = False
        self.show_trajectories: bool = False
        self.show_policy: bool = False

        self.show_true_backprojection_sets: bool = True
        self.show_true_backprojection_set_samples: bool = True
        self.show_target_set: bool = True
        self.show_backreachable_sets: bool = True
        self.show_backprojection_sets: bool = True
        self.show_backprojection_set_cells: bool = True
        self.show_initial_state_set: bool = False

        self.show: bool = False
        self.save_plot: bool = True
        self.plot_axis_labels: list = ["$x_0$", "$x_1$"]
        self.plot_dims: list = [0, 1]
        self.aspect: str = "auto"
        self.plot_lims: Optional[list] = None
        self.tight_layout: bool = True

        self.plot_filename: Optional[str] = None
        self.network: Optional[torch.nn.Sequential] = None

        self.true_backprojection_set_color: str = "k"
        self.true_backprojection_set_zorder: int = 3
        self.true_backprojection_set_ls: str = "-"
        self.true_backprojection_set_lw: int = 2

        self.estimated_backprojection_set_color: str = "tab:blue"
        self.estimated_backprojection_set_zorder: int = 2
        self.estimated_backprojection_set_ls: str = "-"
        self.estimated_backprojection_set_lw: int = 2

        self.estimated_backprojection_set_cell_color: str = "tab:gray"
        self.estimated_backprojection_set_cell_zorder: int = 2
        self.estimated_backprojection_set_cell_ls: str = "-"
        self.estimated_backprojection_set_cell_lw: int = 1

        self.estimated_one_step_backprojection_set_color: str = "tab:orange"
        self.estimated_one_step_backprojection_set_zorder: int = -1
        self.estimated_one_step_backprojection_set_ls: str = "-"
        self.estimated_one_step_backprojection_set_lw: int = 1

        self.estimated_backprojection_partitioned_set_color: str = "tab:gray"
        self.estimated_backprojection_partitioned_set_zorder: int = 5
        self.estimated_backprojection_partitioned_set_ls: str = "-"
        self.estimated_backprojection_partitioned_set_lw: int = 1

        self.backreachable_set_color: str = "tab:cyan"
        self.backreachable_set_zorder: int = -1
        self.backreachable_set_ls: str = "--"
        self.backreachable_set_lw: int = 1

        self.target_set_color: str = "tab:red"
        self.target_set_zorder: int = 1
        self.target_set_ls: str = "-"
        self.target_set_lw: int = 2

        self.initial_set_color: str = "k"
        self.initial_set_zorder: int = 1
        self.initial_set_ls: str = "-"
        self.initial_set_lw: int = 1

    @property
    def plot_dims(self):
        return self._plot_dims

    @plot_dims.setter
    def plot_dims(self, plot_dims):
        self._plot_dims = [[a] for a in plot_dims]

        if len(plot_dims) == 2:
            self.projection = None
            self.plot_2d = True
            self.linewidth = 2
        elif len(plot_dims) == 3:
            self.projection = "3d"
            self.plot_2d = False
            self.linewidth = 1
            self.aspect = "auto"

    @property
    def initial_state_range(self) -> np.ndarray:
        return self._initial_state_range

    @initial_state_range.setter
    def initial_state_range(
        self, initial_state_range_str: Optional[str]
    ) -> None:
        if initial_state_range_str is None:
            self._initial_state_range = None
            self.initial_state_set = None
            return
        self._initial_state_range = np.array(
            ast.literal_eval(initial_state_range_str)
        )
        self.initial_state_set = constraints.LpConstraint(
            self._initial_state_range
        )

    def visualize(
        self,
        target_set: constraints.SingleTimestepConstraint,
        backprojection_sets: constraints.MultiTimestepConstraint,
        network: torch.nn.Sequential,
        **kwargs: dict,
    ) -> None:
        self.network = network
        backreachable_set = kwargs["per_timestep"][-1]["backreachable_set"]
        self.setup_visualization(
            target_set,
            backprojection_sets.get_t_max(),
            backreachable_set,
        )

        self.visualize_estimates(backprojection_sets)

        if self.tight_layout:
            self.animate_fig.tight_layout()

        if self.plot_lims is not None:
            plot_lims_arr = np.array(ast.literal_eval(self.plot_lims))
            self.animate_axes.set_xlim(plot_lims_arr[0])
            self.animate_axes.set_ylim(plot_lims_arr[1])
            if not self.plot_2d:
                self.animate_axes.set_zlim(plot_lims_arr[2])

        if self.save_plot:
            plt.savefig(self.plot_filename)

        if self.show:
            plt.show()

    def get_tmp_animation_filename(self, iteration):
        filename = self.tmp_animation_save_dir + "tmp_{}.png".format(
            str(iteration).zfill(6)
        )
        return filename

    def setup_visualization(
        self,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
        backreachable_set: constraints.MultiTimestepConstraint,
    ) -> None:
        self.default_patches: list = []
        self.default_lines: list = []

        self.animate_fig, self.animate_axes = plt.subplots(
            1, 1, subplot_kw=dict(projection=self.projection)
        )

        if self.show_target_set:
            target_set.plot(
                self.animate_axes,
                self.plot_dims,
                self.target_set_color,
                fc_color="None",
                zorder=self.target_set_zorder,
                plot_2d=self.plot_2d,
                linewidth=self.target_set_lw,
                ls=self.target_set_ls,
            )

        if self.show_backreachable_sets:
            backreachable_set.plot(
                self.animate_axes,
                self.plot_dims,
                self.backreachable_set_color,
                fc_color="None",
                zorder=self.backreachable_set_zorder,
                plot_2d=self.plot_2d,
                linewidth=self.backreachable_set_lw,
                ls=self.backreachable_set_ls,
            )

        if self.show_true_backprojection_sets:
            self.plot_true_backprojection_sets(
                backreachable_set,
                target_set,
                t_max=t_max,
            )

        if self.show_policy:
            display_ground_robot_control_field(
                controller=self.network, ax=self.animate_axes
            )

        # Sketchy workaround to trajectories not showing up
        if self.show_trajectories and self.initial_state_set is not None:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                self.initial_state_set,
                ax=self.animate_axes,
                controller=self.network,
                input_dims=self.plot_dims,
            )

        if self.show_initial_state_set and self.initial_state_set is not None:
            self.initial_state_set.plot(
                self.animate_axes,
                self.plot_dims,
                self.initial_set_color,
                fc_color="None",
                zorder=self.initial_set_zorder,
                plot_2d=self.plot_2d,
                linewidth=self.initial_set_lw,
                ls=self.initial_set_ls,
            )

        self.animate_axes.set_aspect(self.aspect)

        self.animate_axes.set_xlabel(self.plot_axis_labels[0])
        self.animate_axes.set_ylabel(self.plot_axis_labels[1])
        if not self.plot_2d:
            self.animate_axes.set_zlabel(self.plot_axis_labels[2])

    def visualize_estimates(
        self,
        backprojection_sets: constraints.MultiTimestepConstraint,
    ) -> None:
        # Bring forward whatever default items should be in the plot
        # (e.g., MC samples, target set boundaries)
        for item in self.default_patches + self.default_lines:
            if isinstance(item, Patch):
                self.animate_axes.add_patch(item)
            elif isinstance(item, Line2D):
                self.animate_axes.add_line(item)

        if self.show_backprojection_sets:
            rects = backprojection_sets.plot(
                self.animate_axes,
                self.plot_dims,
                self.estimated_backprojection_set_color,
                zorder=self.estimated_backprojection_set_zorder,
                linewidth=self.estimated_backprojection_set_lw,
                plot_2d=self.plot_2d,
            )
            self.default_patches += rects

        if self.show_backprojection_set_cells:
            for backprojection_set in backprojection_sets.constraints:
                for cell in backprojection_set.cells:
                    rects = cell.plot(
                        self.animate_axes,
                        self.plot_dims,
                        self.estimated_backprojection_set_cell_color,
                        zorder=self.estimated_backprojection_set_cell_zorder,
                        linewidth=self.estimated_backprojection_set_cell_lw,
                        plot_2d=self.plot_2d,
                    )
                    self.default_patches += rects

        if self.show_samples_from_cells:
            for backprojection_set in backprojection_sets.constraints:
                for cell in backprojection_set.cells:
                    self.dynamics.show_samples(
                        backprojection_sets.get_t_max() * self.dynamics.dt,
                        cell,
                        ax=self.animate_axes,
                        controller=self.network,
                        input_dims=self.plot_dims,
                    )

        # # Do auxiliary stuff to make sure animations look nice
        # if title is not None:
        #     plt.suptitle(title)

        # if (iteration == 0 or iteration == -1) and not dont_tighten_layout:
        #     plt.tight_layout()

        # if self.show_animation:
        #     plt.pause(0.01)

        # if self.make_animation and iteration is not None:
        #     os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
        #     filename = self.get_tmp_animation_filename(iteration)
        #     plt.savefig(filename)

        # if self.make_animation and not self.plot_2d:
        #     # Make an animated 3d view
        #     os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
        #     for i, angle in enumerate(range(-100, 0, 2)):
        #         self.animate_axes.view_init(30, angle)
        #         filename = self.get_tmp_animation_filename(i)
        #         plt.savefig(filename)
        #     self.compile_animation(i, delete_files=True, duration=0.2)

    def plot_true_backprojection_sets(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        target_set: constraints.SingleTimestepConstraint,
        t_max: int,
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
                controller=self.network,
            )
        )

        if self.show_true_backprojection_set_samples:
            # xt1_from_those_samples_ = xt1_from_those_samples[
            #     (within_constraint_inds)
            # ]

            # Show samples from inside the backprojection set and their
            # futures under the NN (should end in target set)
            self.dynamics.show_samples(
                None,
                None,
                ax=self.animate_axes,
                controller=None,
                input_dims=self.plot_dims,
                zorder=1,
                xs=x_samples_inside_backprojection_set,
                colors=None,
            )

        # Compute and draw a convex hull around all the backprojection set
        # samples. This is our "true" backprojection set -- but...
        # it is sampling-based so that is an under-approximation,
        # and it is a convex hull, so that is an over-approximation.
        for t in range(t_max):
            _ = plot_convex_hull(
                x_samples_inside_backprojection_set[:, t, :],
                dims=self.plot_dims,
                color=self.true_backprojection_set_color,
                linewidth=self.true_backprojection_set_lw,
                linestyle=self.true_backprojection_set_ls,
                zorder=self.true_backprojection_set_zorder,
                label="Backprojection Set (True)",
                axes=self.animate_axes,
            )
        # self.default_lines[self.output_axis].append(conv_hull_line[0])


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
