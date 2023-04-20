import os
import time
from itertools import product

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pygifsicle import optimize

from nfl_veripy.utils.utils import get_sampled_outputs, samples_to_range, sect

label_dict = {
    "linf": "$\ell_\infty$-ball",
    "convex_hull": "Convex Hull",
    "lower_bnds": "Lower Bounds",
}


class Partitioner:
    def __init__(self):
        self.tmp_animation_save_dir = "{}/../../results/tmp_animation/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.animation_save_dir = "{}/../../results/animations/".format(
            os.path.dirname(os.path.abspath(__file__))
        )

    def get_output_range(self):
        raise NotImplementedError

    def get_sampled_outputs(self, input_range, propagator, N=1000):
        return get_sampled_outputs(input_range, propagator, N=N)

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def squash_down_to_one_range_old(self, u_e, M):
        u_e_ = u_e.copy()
        if len(M) > 0:
            # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            M_range = np.empty_like(u_e_)
            M_range[:, 1] = np.max(M_numpy[:, 1, :], axis=1)
            M_range[:, 0] = np.min(M_numpy[:, 0, :], axis=1)

            # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([u_e_, M_range])
            u_e_[:, 1] = np.max(tmp[:, 1, :], axis=1)
            u_e_[:, 0] = np.min(tmp[:, 0, :], axis=1)
        return u_e_

    def squash_down_to_one_range(self, output_range_sim, M):
        u_e = np.empty_like(output_range_sim)
        if len(M) > 0:
            # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            u_e[:, 1] = np.max(M_numpy[:, 1, :], axis=1)
            u_e[:, 0] = np.min(M_numpy[:, 0, :], axis=1)

            # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([output_range_sim, u_e])
            u_e[:, 1] = np.max(tmp[:, 1, :], axis=1)
            u_e[:, 0] = np.min(tmp[:, 0, :], axis=1)
        return u_e

    def squash_down_to_convex_hull(self, M, sim_hull_pts=None):
        from scipy.spatial import ConvexHull

        ndim = M[0][1].shape[0]
        pts = np.empty((len(M) * (2 ** (ndim)), ndim))
        i = 0
        for (input_range, output_range) in M:
            for pt in product(*output_range):
                pts[i, :] = pt
                i += 1
        hull = ConvexHull(pts, incremental=True)
        if sim_hull_pts is not None:
            hull.add_points(sim_hull_pts)
        return hull

    def setup_visualization(
        self,
        input_range,
        output_range,
        propagator,
        show_samples=True,
        outputs_to_highlight=None,
        inputs_to_highlight=None,
        show_input=True,
        show_output=True,
        labels={},
        aspects={},
    ):
        num_subplots = int(show_input) + int(show_output)
        self.animate_fig, self.animate_axes = plt.subplots(1, num_subplots)

        if num_subplots == 1:
            if show_input:
                self.ax_input = self.animate_axes
            else:
                self.ax_output = self.animate_axes
        else:
            self.ax_input, self.ax_output = self.animate_axes

        self.input_axis = 0
        self.output_axis = int(show_input)

        if inputs_to_highlight is None:
            # Automatically detect which input dims to show based on input_range
            num_input_dimensions_to_plot = 2
            input_shape = input_range.shape[:-1]
            lengths = (
                input_range[..., 1].flatten() - input_range[..., 0].flatten()
            )
            flat_dims = np.argpartition(
                lengths, -num_input_dimensions_to_plot
            )[-num_input_dimensions_to_plot:]
            flat_dims.sort()
            input_dims = [
                np.unravel_index(flat_dim, input_range.shape[:-1])
                for flat_dim in flat_dims
            ]
            input_names = [
                "NN Input Dim. {}".format(input_dims[0][0]),
                "NN Input Dim. {}".format(input_dims[1][0]),
            ]
        else:
            input_dims = [x["dim"] for x in inputs_to_highlight]
            input_names = [x["name"] for x in inputs_to_highlight]
        self.input_dims_ = tuple(
            [
                tuple([input_dims[j][i] for j in range(len(input_dims))])
                for i in range(len(input_dims[0]))
            ]
        )

        if outputs_to_highlight is None:
            output_dims = [(0,), (1,)]
            output_names = ["NN Output Dim. 0", "NN Output Dim. 1"]
        else:
            output_dims = [x["dim"] for x in outputs_to_highlight]
            output_names = [x["name"] for x in outputs_to_highlight]
        self.output_dims_ = tuple(
            [
                tuple([output_dims[j][i] for j in range(len(output_dims))])
                for i in range(len(output_dims[0]))
            ]
        )

        if show_input:
            scale = 0.05
            x_off = max(
                (
                    input_range[input_dims[0] + (1,)]
                    - input_range[input_dims[0] + (0,)]
                )
                * (scale),
                1e-5,
            )
            y_off = max(
                (
                    input_range[input_dims[1] + (1,)]
                    - input_range[input_dims[1] + (0,)]
                )
                * (scale),
                1e-5,
            )
            self.ax_input.set_xlim(
                input_range[input_dims[0] + (0,)] - x_off,
                input_range[input_dims[0] + (1,)] + x_off,
            )
            self.ax_input.set_ylim(
                input_range[input_dims[1] + (0,)] - y_off,
                input_range[input_dims[1] + (1,)] + y_off,
            )

        if show_output:
            scale = 0.05
            x_off = max(
                (
                    output_range[output_dims[0] + (1,)]
                    - output_range[output_dims[0] + (0,)]
                )
                * (scale),
                1e-5,
            )
            y_off = max(
                (
                    output_range[output_dims[1] + (1,)]
                    - output_range[output_dims[1] + (0,)]
                )
                * (scale),
                1e-5,
            )
            self.ax_output.set_xlim(
                output_range[output_dims[0] + (0,)] - x_off,
                output_range[output_dims[0] + (1,)] + x_off,
            )
            self.ax_output.set_ylim(
                output_range[output_dims[1] + (0,)] - y_off,
                output_range[output_dims[1] + (1,)] + y_off,
            )

        if show_input:
            if (
                "input" in labels
                and len([x for x in labels["input"] if x is not None]) > 0
            ):
                self.ax_input.set_xlabel(labels["input"][0])
                self.ax_input.set_ylabel(labels["input"][1])
            else:
                self.ax_input.set_xlabel(input_names[0])
                self.ax_input.set_ylabel(input_names[1])
            if "input" in aspects:
                self.ax_input.set_aspect(aspects["input"])

        if show_output:
            if (
                "output" in labels
                and len([x for x in labels["output"] if x is not None]) > 0
            ):
                self.ax_output.set_xlabel(labels["output"][0])
                self.ax_output.set_ylabel(labels["output"][1])
            else:
                self.ax_output.set_xlabel(output_names[0])
                self.ax_output.set_ylabel(output_names[1])
            if "output" in aspects:
                self.ax_output.set_aspect(aspects["output"])

        # Make a rectangle for the Exact boundaries
        sampled_outputs = self.get_sampled_outputs(input_range, propagator)
        if show_output:
            if show_samples:
                self.ax_output.scatter(
                    sampled_outputs[..., output_dims[0]],
                    sampled_outputs[..., output_dims[1]],
                    c="k",
                    marker=".",
                    zorder=2,
                    label="Sampled Outputs",
                )

        self.default_patches = [[] for _ in range(num_subplots)]
        self.default_lines = [[] for _ in range(num_subplots)]

        # Full input range
        if show_input:
            input_range__ = input_range[self.input_dims_]
            input_rect = Rectangle(
                input_range__[:2, 0],
                input_range__[0, 1] - input_range__[0, 0],
                input_range__[1, 1] - input_range__[1, 0],
                fc="none",
                linewidth=2,
                edgecolor="k",
                zorder=3,
                label="Full Input Set",
            )
            self.ax_input.add_patch(input_rect)
            self.default_patches[0] = [input_rect]

        if show_output:
            # Exact output range
            color = "red"
            linestyle = "--"
            linewidth = 2
            zorder = 3
            if self.interior_condition == "linf":
                output_range_exact = self.samples_to_range(sampled_outputs)
                output_range_exact_ = output_range_exact[self.output_dims_]
                rect = Rectangle(
                    output_range_exact_[:2, 0],
                    output_range_exact_[0, 1] - output_range_exact_[0, 0],
                    output_range_exact_[1, 1] - output_range_exact_[1, 0],
                    fc="none",
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                    edgecolor=color,
                    label="True Bounds ({})".format(
                        label_dict[self.interior_condition]
                    ),
                )
                self.ax_output.add_patch(rect)
                self.default_patches[self.output_axis].append(rect)
            elif self.interior_condition == "lower_bnds":
                output_range_exact = self.samples_to_range(sampled_outputs)
                output_range_exact_ = output_range_exact[self.output_dims_]
                line1 = self.ax_output.axhline(
                    output_range_exact_[1, 0],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                    color=color,
                    label="True Bounds ({})".format(
                        label_dict[self.interior_condition]
                    ),
                )
                line2 = self.ax_output.axvline(
                    output_range_exact_[0, 0],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                    color=color,
                )
                self.default_lines[self.output_axis].append(line1)
                self.default_lines[self.output_axis].append(line2)
            elif self.interior_condition == "convex_hull":
                from scipy.spatial import ConvexHull

                self.true_hull = ConvexHull(sampled_outputs)
                self.true_hull_ = ConvexHull(
                    sampled_outputs[..., output_dims].squeeze()
                )
                line = self.ax_output.plot(
                    np.append(
                        sampled_outputs[self.true_hull_.vertices][
                            ..., output_dims[0]
                        ],
                        sampled_outputs[self.true_hull_.vertices[0]][
                            ..., output_dims[0]
                        ],
                    ),
                    np.append(
                        sampled_outputs[self.true_hull_.vertices][
                            ..., output_dims[1]
                        ],
                        sampled_outputs[self.true_hull_.vertices[0]][
                            ..., output_dims[1]
                        ],
                    ),
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    zorder=zorder,
                    label="True Bounds ({})".format(
                        label_dict[self.interior_condition]
                    ),
                )
                self.default_lines[self.output_axis].append(line[0])
            else:
                raise NotImplementedError

        plt.tight_layout()

    def visualize(
        self,
        M,
        interior_M,
        u_e,
        iteration=None,
        show_input=True,
        show_output=True,
        title=None,
    ):
        if show_input:
            for patch in self.default_patches[self.input_axis]:
                self.ax_input.add_patch(patch)
            for line in self.default_lines[self.input_axis]:
                self.ax_input.add_line(line)
        if show_output:
            for patch in self.default_patches[self.output_axis]:
                self.ax_output.add_patch(patch)
            for line in self.default_lines[self.output_axis]:
                self.ax_output.add_line(line)
        input_dims_ = self.input_dims_

        # Rectangles that might still be outside the sim pts
        first = True
        for (input_range_, output_range_) in M:
            if first:
                input_label = "Cell of Partition"
                output_label = "One Cell's Estimated Bounds"
                first = False
            else:
                input_label = None
                output_label = None

            if show_output:
                output_range__ = output_range_[self.output_dims_]
                rect = Rectangle(
                    output_range__[:, 0],
                    output_range__[0, 1] - output_range__[0, 0],
                    output_range__[1, 1] - output_range__[1, 0],
                    fc="none",
                    linewidth=1,
                    edgecolor="tab:blue",
                    label=output_label,
                )
                self.ax_output.add_patch(rect)

            if show_input:
                input_range__ = input_range_[input_dims_]
                rect = Rectangle(
                    input_range__[:, 0],
                    input_range__[0, 1] - input_range__[0, 0],
                    input_range__[1, 1] - input_range__[1, 0],
                    fc="none",
                    linewidth=1,
                    edgecolor="tab:blue",
                    label=input_label,
                )
                self.ax_input.add_patch(rect)

        # Rectangles that are within the sim pts
        for (input_range_, output_range_) in interior_M:

            if show_output:
                output_range__ = output_range_[self.output_dims_]
                rect = Rectangle(
                    output_range__[:2, 0],
                    output_range__[0, 1] - output_range__[0, 0],
                    output_range__[1, 1] - output_range__[1, 0],
                    fc="none",
                    linewidth=1,
                    edgecolor="tab:blue",
                )
                self.ax_output.add_patch(rect)

            if show_input:
                input_range__ = input_range_[input_dims_]
                rect = Rectangle(
                    input_range__[:, 0],
                    input_range__[0, 1] - input_range__[0, 0],
                    input_range__[1, 1] - input_range__[1, 0],
                    fc="none",
                    linewidth=1,
                    edgecolor="tab:blue",
                )
                self.ax_input.add_patch(rect)

        if show_output:
            linewidth = 3
            color = "black"
            if self.interior_condition == "linf":
                # Make a rectangle for the estimated boundaries
                output_range_estimate = self.squash_down_to_one_range(u_e, M)
                output_range_estimate_ = output_range_estimate[
                    self.output_dims_
                ]
                rect = Rectangle(
                    output_range_estimate_[:2, 0],
                    output_range_estimate_[0, 1]
                    - output_range_estimate_[0, 0],
                    output_range_estimate_[1, 1]
                    - output_range_estimate_[1, 0],
                    fc="none",
                    linewidth=linewidth,
                    edgecolor=color,
                    label="Estimated Bounds ({})".format(
                        label_dict[self.interior_condition]
                    ),
                )
                self.ax_output.add_patch(rect)
            elif self.interior_condition == "lower_bnds":
                output_range_estimate = self.squash_down_to_one_range(u_e, M)
                output_range_estimate_ = output_range_estimate[
                    self.output_dims_
                ]
                self.ax_output.axhline(
                    output_range_estimate_[1, 0],
                    linewidth=linewidth,
                    color=color,
                    label="Estimated Bounds ({})".format(
                        label_dict[self.interior_condition]
                    ),
                )
                self.ax_output.axvline(
                    output_range_estimate_[0, 0],
                    linewidth=linewidth,
                    color=color,
                )
            elif self.interior_condition == "convex_hull":
                from scipy.spatial import ConvexHull

                M_ = [
                    (input_range_, output_range_[self.output_dims_])
                    for (input_range_, output_range_) in M
                ]
                hull = self.squash_down_to_convex_hull(
                    M_ + interior_M, self.true_hull_.points
                )
                self.ax_output.plot(
                    np.append(
                        hull.points[hull.vertices, 0],
                        hull.points[hull.vertices[0], 0],
                    ),
                    np.append(
                        hull.points[hull.vertices, 1],
                        hull.points[hull.vertices[0], 1],
                    ),
                    color=color,
                    linewidth=linewidth,
                    label="Estimated Bounds ({})".format(
                        label_dict[self.interior_condition]
                    ),
                )
            else:
                raise NotImplementedError

        if title is not None:
            plt.suptitle(title)

        if iteration == 0 or iteration == -1:
            plt.tight_layout()

        if self.show_animation:
            plt.pause(0.01)

        if self.make_animation and iteration is not None:
            os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
            filename = self.get_tmp_animation_filename(iteration)
            plt.savefig(filename)

    def get_tmp_animation_filename(self, iteration):
        filename = self.tmp_animation_save_dir + "tmp_{}.png".format(
            str(iteration).zfill(6)
        )
        return filename

    def get_error(self, output_range_exact, output_range_estimate):
        if self.interior_condition == "linf":
            true_area = np.product(
                output_range_exact[..., 1] - output_range_exact[..., 0]
            )
            estimated_area = np.product(
                output_range_estimate[..., 1] - output_range_estimate[..., 0]
            )
            error = (estimated_area - true_area) / true_area
        elif self.interior_condition == "lower_bnds":
            # Need to put lower bnd error into proper scale --> one idea is to use
            # length in each dimension of output (i.e., if you get 0.1 away
            # from lower bnd in a dimension that has range 100, that's more impressive
            # than in a dimension that has range 0.01)
            # lower_bnd_error_area = np.product(output_range_exact[...,0] - output_range_estimate[...,0])
            # true_area = np.product(output_range_exact[...,1] - output_range_exact[...,0])
            # error = lower_bnd_error_area / true_area

            # Just add up the distance btwn true and estimated lower bnds for each dimension
            error = np.sum(
                output_range_exact[..., 0] - output_range_estimate[..., 0]
            )

        elif self.interior_condition == "convex_hull":
            true_area = output_range_exact.volume
            estimated_area = output_range_estimate.volume
            error = (estimated_area - true_area) / true_area
        else:
            raise NotImplementedError
        return error

    def check_termination(
        self,
        input_range_,
        num_propagator_calls,
        u_e,
        output_range_sim,
        M,
        elapsed_time,
    ):
        if self.termination_condition_type == "input_cell_size":

            #  print(input_range_[...,1] - input_range_[...,0])
            M_numpy = np.dstack([input_range for (input_range, _) in M])

            terminate = (
                np.min(M_numpy[:, 1] - M_numpy[:, 0])
                <= self.termination_condition_value
            )
        elif self.termination_condition_type == "num_propagator_calls":
            terminate = (
                num_propagator_calls >= self.termination_condition_value
            )
        elif self.termination_condition_type == "pct_improvement":
            # This doesnt work very well, because a lot of times
            # the one-step improvement is zero
            last_u_e = u_e.copy()
            if self.interior_condition in ["lower_bnds", "linf"]:
                u_e = self.squash_down_to_one_range(output_range_sim, M)
                improvement = self.get_error(last_u_e, u_e)
                if iteration == 0:
                    improvement = np.inf
            elif self.interior_condition == "convex_hull":
                # raise NotImplementedError
                last_hull = estimated_hull.copy()

                estimated_hull = self.squash_down_to_convex_hull(
                    M, self.sim_convex_hull.points
                )
                improvement = self.get_error(last_hull, estimated_hull)
            terminate = improvement <= self.termination_condition_value
        elif self.termination_condition_type == "pct_error":
            if self.interior_condition in ["lower_bnds", "linf"]:
                u_e = self.squash_down_to_one_range(output_range_sim, M)
                error = self.get_error(output_range_sim, u_e)
            elif self.interior_condition == "convex_hull":
                estimated_hull = self.squash_down_to_convex_hull(
                    M, self.sim_convex_hull.points
                )
                error = self.get_error(self.sim_convex_hull, estimated_hull)
            terminate = error <= self.termination_condition_value
        #   print(error)
        elif self.termination_condition_type == "verify":
            M_ = M + [(input_range_, output_range_)]
            ndim = M_[0][1].shape[0]
            pts = np.empty((len(M_) * (2 ** (ndim)), ndim))
            i = 0
            for (input_range, output_range) in M:
                for pt in product(*output_range):
                    pts[i, :] = pt
                    i += 1
        elif self.termination_condition_type == "time_budget":
            terminate = elapsed_time >= self.termination_condition_value

            # print(pts)
            # output_ranges_ = [x[1] for x in M+[(input_range_, output_range_)]]
            # ndim = output_ranges_[0].shape[0]
            # pts = np.empty((ndim**2, ndim+1))
            # pts[:,-1] = 1.
            # for i, pt in enumerate(product(*output_ranges_)):
            #     pts[i,:-1] = pt
            # inside = np.all(np.matmul(self.termination_condition_value[0], pts.T) <= self.termination_condition_value[1])
        else:
            raise NotImplementedError
        return terminate

    def compile_info(
        self,
        output_range_sim,
        M,
        interior_M,
        num_propagator_calls,
        t_end_overall,
        t_start_overall,
        propagator_computation_time,
        iteration,
    ):
        info = {}
        estimated_range = None
        estimated_hull = None
        if self.interior_condition in ["lower_bnds", "linf"]:
            estimated_range = self.squash_down_to_one_range(
                output_range_sim, M
            )
            estimated_error = self.get_error(output_range_sim, estimated_range)
        elif self.interior_condition == "convex_hull":
            info["exact_hull"] = self.sim_convex_hull
            estimated_hull = self.squash_down_to_convex_hull(
                M + interior_M, self.sim_convex_hull.points
            )
            info["estimated_hull"] = estimated_hull
            estimated_error = self.get_error(
                self.sim_convex_hull, estimated_hull
            )

        info["all_partitions"] = M + interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        info["estimation_error"] = estimated_error
        info["computation_time"] = t_end_overall - t_start_overall
        info["propagator_computation_time"] = propagator_computation_time
        info["num_iteration"] = iteration
        info["estimated_range"] = estimated_range

        return info

    def partition_loop(
        self,
        M,
        interior_M,
        output_range_sim,
        sect_method,
        num_propagator_calls,
        input_range,
        u_e,
        propagator,
        propagator_computation_time,
        t_start_overall,
    ):
        if self.make_animation:
            self.call_visualizer(output_range_sim, M, num_propagator_calls, interior_M, iteration=-1)

        # Used by UnGuided, SimGuided, GreedySimGuided, etc.
        iteration = 0
        terminate = False
        start_time_partition_loop = t_start_overall
        while len(M) != 0 and not terminate:
            input_range_, output_range_ = self.grab_from_M(
                M, output_range_sim
            )  # (Line 9)

            if self.check_if_partition_within_sim_bnds(
                output_range_, output_range_sim
            ):
                # Line 11
                interior_M.append((input_range_, output_range_))
            else:
                # Line 14
                elapsed_time = time.time() - start_time_partition_loop
                terminate = self.check_termination(
                    input_range,
                    num_propagator_calls,
                    u_e,
                    output_range_sim,
                    M + [(input_range_, output_range_)] + interior_M,
                    elapsed_time,
                )

                if not terminate:
                    # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                    # Lines 16-17
                    for input_range_ in input_ranges_:
                        t_start = time.time()
                        output_range_, _ = propagator.get_output_range(
                            input_range_
                        )
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += 1
                        M.append((input_range_, output_range_))  # Line 18

                    # if self.interior_condition in ["lower_bnds", "linf"]:
                    #    estimated_range = self.squash_down_to_one_range(output_range_sim, M)
                    #    estimated_error = self.get_error(output_range_sim, estimated_range)
                    #  elif self.interior_condition == "convex_hull":
                    #      estimated_hull = self.squash_down_to_convex_hull(M+interior_M, self.sim_convex_hull.points)
                    #      estimated_error = self.get_error(self.sim_convex_hull, estimated_hull)

                else:  # Lines 19-20
                    M.append((input_range_, output_range_))

                if self.make_animation:
                    self.call_visualizer(output_range_sim, M, num_propagator_calls, interior_M, iteration=iteration)
            iteration += 1

        # Line 24
        u_e = self.squash_down_to_one_range(output_range_sim, M)
        t_end_overall = time.time()

        # Stats & Visualization
        info = self.compile_info(
            output_range_sim,
            M,
            interior_M,
            num_propagator_calls,
            t_end_overall,
            t_start_overall,
            propagator_computation_time,
            iteration,
        )
        if self.make_animation:
            self.compile_animation(iteration)

        return u_e, info

    def call_visualizer(self, output_range_sim, M, num_propagator_calls, interior_M, iteration):
        u_e = self.squash_down_to_one_range(output_range_sim, M)
        title = "# Propagator Calls: {}".format(
            str(num_propagator_calls)
        )
        # title = None

        self.visualize(
            M,
            interior_M,
            u_e,
            iteration=iteration,
            show_input=self.show_input,
            show_output=self.show_output,
            title=title,
        )

    def sample(self, input_range, propagator, N=None):
        # This is only used for error estimation (evaluation!)
        if N is None:
            N = self.num_simulations
        # Run N simulations (i.e., randomly sample N pts
        # from input range --> query NN --> get N output pts)
        # (Line 5)
        input_shape = input_range.shape[:-1]
        sampled_inputs = np.random.uniform(
            input_range[..., 0], input_range[..., 1], (int(N),) + input_shape
        )
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        output_range_sim = np.empty(sampled_outputs.shape[1:] + (2,))
        output_range_sim[:, 1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:, 0] = np.min(sampled_outputs, axis=0)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull

            self.sim_convex_hull = ConvexHull(sampled_outputs)

        return output_range_sim, sampled_outputs, sampled_inputs

    def compile_animation(self, iteration, delete_files=False, start_iteration=0, duration=0.1):
        filenames = [
            self.get_tmp_animation_filename(i) for i in range(start_iteration, iteration)
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
