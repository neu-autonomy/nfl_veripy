import ast
import os
from typing import Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pygifsicle import optimize

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.controller_generation import (
    display_ground_robot_control_field,
)


class ForwardVisualizer:
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

        self.reachable_set_color: str = "tab:blue"
        self.reachable_set_zorder: int = 2
        self.reachable_set_ls: str = "-"
        self.reachable_set_lw: int = 1

        self.reachable_set_cell_color: str = "tab:gray"
        self.reachable_set_cell_zorder: int = 2
        self.reachable_set_cell_ls: str = "-"
        self.reachable_set_cell_lw: int = 1

        self.initial_set_color: str = "k"
        self.initial_set_zorder: int = 2
        self.initial_set_ls: str = "-"
        self.initial_set_lw: int = 1

        self.target_set_color: str = "tab:red"
        self.target_set_zorder: int = 2
        self.sample_zorder: int = 1
        self.sample_colors: Optional[str] = None

        self.show_samples: bool = True
        self.show_reachable_set_cells: bool = False
        self.show_samples_from_cells: bool = False
        self.show_trajectories: bool = False
        self.show: bool = False
        self.save_plot: bool = True
        self.plot_axis_labels: list = ["$x_0$", "$x_1$"]
        self.plot_dims: list = [0, 1]
        self.aspect: str = "auto"
        self.plot_lims: Optional[list] = None
        self.controller_name: Optional[str] = None
        self.tight_layout: bool = True

        self.plot_filename: Optional[str] = None
        self.network: Optional[torch.nn.Sequential] = None

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

    def visualize(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        reachable_sets: constraints.MultiTimestepConstraint,
        network: torch.nn.Sequential,
        **kwargs: dict,
    ) -> None:
        self.network = network
        self.setup_visualization(
            initial_set,
            reachable_sets.get_t_max(),
        )

        # kwargs.get(
        #         "exterior_partitions", kwargs.get("all_partitions", [])
        #     ),
        #     kwargs.get("interior_partitions", []),

        self.visualize_estimates(reachable_sets)

        if self.tight_layout:
            self.animate_fig.tight_layout()

        if self.plot_lims is not None:
            plot_lims_arr = np.array(ast.literal_eval(self.plot_lims))
            self.animate_axes.set_xlim(plot_lims_arr[0])
            self.animate_axes.set_ylim(plot_lims_arr[1])

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
        initial_set: constraints.SingleTimestepConstraint,
        t_max: int,
    ) -> None:
        self.default_patches: list = []
        self.default_lines: list = []

        self.animate_fig, self.animate_axes = plt.subplots(
            1, 1, subplot_kw=dict(projection=self.projection)
        )
        if self.controller_name is not None:
            display_ground_robot_control_field(
                name=self.controller_name, ax=self.animate_axes
            )

        self.animate_axes.set_aspect(self.aspect)

        rect = initial_set.plot(
            self.animate_axes,
            self.plot_dims,
            self.initial_set_color,
            zorder=self.initial_set_zorder,
            linewidth=self.initial_set_lw,
            ls=self.initial_set_ls,
            plot_2d=self.plot_2d,
        )
        self.default_patches += rect

        if self.show_samples:
            self.dynamics.show_samples(
                t_max * self.dynamics.dt,
                initial_set,
                ax=self.animate_axes,
                controller=self.network,
                input_dims=self.plot_dims,
                zorder=self.sample_zorder,
                colors=self.sample_colors,
            )

        if self.show_samples_from_cells:
            for initial_set_cell in initial_set.cells:
                self.dynamics.show_samples(
                    t_max * self.dynamics.dt,
                    initial_set_cell,
                    ax=self.animate_axes,
                    controller=self.network,
                    input_dims=self.plot_dims,
                    zorder=self.sample_zorder,
                    colors=self.sample_colors,
                )

        if self.show_trajectories:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                initial_set,
                ax=self.animate_axes,
                controller=self.network,
                input_dims=self.plot_dims,
                zorder=self.sample_zorder,
                colors=self.sample_colors,
            )

        self.animate_axes.set_xlabel(self.plot_axis_labels[0])
        self.animate_axes.set_ylabel(self.plot_axis_labels[1])
        if not self.plot_2d:
            self.animate_axes.set_zlabel(self.plot_axis_labels[2])

    def visualize_estimates(
        self,
        reachable_sets: constraints.MultiTimestepConstraint,
    ) -> None:
        # Bring forward whatever default items should be in the plot
        # (e.g., MC samples, initial state set boundaries)
        for item in self.default_patches + self.default_lines:
            if isinstance(item, Patch):
                self.animate_axes.add_patch(item)
            elif isinstance(item, Line2D):
                self.animate_axes.add_line(item)

        reachable_sets.plot(
            self.animate_axes,
            self.plot_dims,
            self.reachable_set_color,
            fc_color="None",
            zorder=self.reachable_set_zorder,
            plot_2d=self.plot_2d,
            linewidth=self.reachable_set_lw,
            ls=self.reachable_set_ls,
        )

        if self.show_reachable_set_cells:
            for cell in reachable_sets.cells:
                cell.plot(
                    self.animate_axes,
                    self.plot_dims,
                    self.reachable_set_cell_color,
                    fc_color="None",
                    zorder=self.reachable_set_cell_zorder,
                    plot_2d=self.plot_2d,
                    linewidth=self.reachable_set_cell_lw,
                    ls=self.reachable_set_cell_ls,
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

    def compile_animation(
        self, iteration, delete_files=False, start_iteration=0, duration=0.1
    ):
        filenames = [
            self.get_tmp_animation_filename(i)
            for i in range(start_iteration, iteration)
        ]
        images = []
        for filename in filenames:
            try:
                image = imageio.imread(filename)
            except FileNotFoundError:
                # not every iteration has a plot
                continue
            images.append(image)
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            if delete_files:
                os.remove(filename)

        # Save the gif in a new animations sub-folder
        os.makedirs(self.animation_save_dir, exist_ok=True)
        animation_filename = (
            self.animation_save_dir + self.get_animation_filename()
        )
        imageio.mimsave(animation_filename, images, duration=duration)
        optimize(animation_filename)  # compress gif file size

    def get_animation_filename(self):
        animation_filename = self.__class__.__name__ + ".gif"
        return animation_filename

    # from Analyzer.py
    def visualize_from_analyzer(
        self,
        input_range,
        output_range_estimate,
        show=True,
        show_samples=True,
        show_legend=True,
        show_input=True,
        show_output=True,
        title=None,
        labels={},
        aspects={},
        **kwargs,
    ):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(
            input_range,
            output_range_estimate,
            self.propagator,
            show_samples=show_samples,
            inputs_to_highlight=kwargs.get("inputs_to_highlight", None),
            outputs_to_highlight=kwargs.get("outputs_to_highlight", None),
            show_input=show_input,
            show_output=show_output,
            labels=labels,
            aspects=aspects,
        )
        self.partitioner.visualize(
            kwargs.get(
                "exterior_partitions", kwargs.get("all_partitions", [])
            ),
            kwargs.get("interior_partitions", []),
            output_range_estimate,
            show_input=show_input,
            show_output=show_output,
        )

        if show_legend:
            if show_input:
                self.partitioner.input_axis.legend(
                    bbox_to_anchor=(0, 1.02, 1, 0.2),
                    loc="lower left",
                    mode="expand",
                    borderaxespad=0,
                    ncol=1,
                )
            if show_output:
                self.partitioner.output_axis.legend(
                    bbox_to_anchor=(0, 1.02, 1, 0.2),
                    loc="lower left",
                    mode="expand",
                    borderaxespad=0,
                    ncol=2,
                )

        if title is not None:
            plt.title(title)

        plt.tight_layout()

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])

        if show:
            plt.show()
        else:
            plt.close()

    # from closedloopsimguidedpartitioner
    def call_visualizer(
        self,
        output_range_sim,
        M,
        num_propagator_calls,
        interior_M,
        iteration,
        dont_tighten_layout=False,
    ):
        u_e = self.squash_down_to_one_range(output_range_sim, M)
        # title = "# Partitions: {}, Error: {}".format(
        #     str(len(M) + len(interior_M)), str(round(error, 3))
        # )
        title = "# Propagator Calls: {}".format(str(int(num_propagator_calls)))
        # title = None

        output_constraint = constraints.LpConstraint(range=u_e)
        self.visualize(
            M,
            interior_M,
            output_constraint,
            iteration=iteration,
            title=title,
            reachable_set_color=self.reachable_set_color,
            reachable_set_zorder=self.reachable_set_zorder,
            dont_tighten_layout=dont_tighten_layout,
        )
