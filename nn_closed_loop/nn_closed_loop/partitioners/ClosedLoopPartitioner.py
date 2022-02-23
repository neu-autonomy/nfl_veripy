import numpy as np
import nn_partition.partitioners as partitioners
from pandas.core.indexing import convert_to_index_sliceable
import pypoman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import nn_closed_loop.constraints as constraints
from copy import deepcopy
import os


class ClosedLoopPartitioner(partitioners.Partitioner):
    def __init__(self, dynamics, make_animation=False, show_animation=False):
        partitioners.Partitioner.__init__(self)
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
        self, input_constraint, output_constraint, propagator
    ):
        output_constraint, info = propagator.get_one_step_reachable_set(
            input_constraint, deepcopy(output_constraint)
        )
        return output_constraint, info

    def get_reachable_set(
        self, input_constraint, output_constraint, propagator, t_max
    ):
        output_constraint_, info = propagator.get_reachable_set(
            input_constraint, deepcopy(output_constraint), t_max
        )

        # TODO: this is repeated from UniformPartitioner... make more universal
        if isinstance(output_constraint, constraints.PolytopeConstraint):
            reachable_set_ = [o.b for o in output_constraint_]
            if output_constraint.b is None:
                output_constraint.b = np.stack(reachable_set_)

            tmp = np.dstack([output_constraint.b, np.stack(reachable_set_)])
            output_constraint.b = np.max(tmp, axis=-1)

            # ranges.append((input_range_, reachable_set_))
        elif isinstance(output_constraint, constraints.LpConstraint):
            reachable_set_ = [o.range for o in output_constraint_]
            if output_constraint.range is None:
                output_constraint.range = np.stack(reachable_set_)

            tmp = np.stack(
                [output_constraint.range, np.stack(reachable_set_)], axis=-1
            )
            output_constraint.range[..., 0] = np.min(tmp[..., 0, :], axis=-1)
            output_constraint.range[..., 1] = np.max(tmp[..., 1, :], axis=-1)

            # ranges.append((input_range_, np.stack(reachable_set_)))
        else:
            raise NotImplementedError

        return output_constraint, info

    def get_error(
        self, input_constraint, output_constraint, propagator, t_max
    ):
        errors = []

        if isinstance(input_constraint, constraints.LpConstraint):
            output_estimated_range = output_constraint.range
            output_range_exact = self.get_sampled_out_range(
                input_constraint, propagator, t_max, num_samples=1000
            )
            num_steps = len(output_constraint.range)
            for t in range(num_steps):
                true_area = np.product(
                    output_range_exact[t][..., 1]
                    - output_range_exact[t][..., 0]
                )
                estimated_area = np.product(
                    output_estimated_range[t][..., 1]
                    - output_estimated_range[t][..., 0]
                )
                errors.append((estimated_area - true_area) / true_area)
        else:
            # Note: This compares the estimated polytope
            # with the "best" polytope with those facets.
            # There could be a much better polytope with lots of facets.
            true_verts = self.get_sampled_out_range(
                input_constraint, propagator, t_max, num_samples=1000,
                output_constraint=output_constraint
            )
            # output_bs_exact = self.get_sampled_out_range(
            #     input_constraint, propagator, t_max, num_samples=1000,
            #     output_constraint=output_constraint
            # )
            num_steps = len(output_constraint.b)
            from scipy.spatial import ConvexHull
            for t in range(num_steps):
                # true_verts = pypoman.polygon.compute_polygon_hull(output_constraint.A, output_bs_exact[t])
                true_hull = ConvexHull(true_verts[:, t+1, :])
                true_area = true_hull.area
                estimated_verts = pypoman.polygon.compute_polygon_hull(output_constraint.A, output_constraint.b[t])
                estimated_hull = ConvexHull(estimated_verts)
                estimated_area = estimated_hull.area
                errors.append((estimated_area - true_area) / true_area)
        final_error = errors[-1]
        avg_error = np.mean(errors)
        return final_error, avg_error, np.array(errors)

    def get_sampled_out_range(
        self, input_constraint, propagator, t_max=5, num_samples=1000,
        output_constraint=None
    ):
        return self.dynamics.get_sampled_output_range(
            input_constraint, t_max, num_samples, controller=propagator.network,
            output_constraint=output_constraint
        )

    def get_sampled_out_range_guidance(
        self, input_constraint, propagator, t_max=5, num_samples=1000
    ):
        # Duplicate of get_sampled_out_range, but called during partitioning
        return self.get_sampled_out_range(input_constraint, propagator, t_max=t_max, num_samples=num_samples)

    def setup_visualization(
        self,
        input_constraint,
        t_max,
        propagator,
        show_samples=True,
        show_trajectories=False,
        inputs_to_highlight=None,
        aspect="auto",
        initial_set_color=None,
        initial_set_zorder=None,
        sample_zorder=None,
        sample_colors=None,
    ):

        self.default_patches = []
        self.default_lines = []

        if inputs_to_highlight is None:
            input_dims = [[0], [1]]
            input_names = [
                "State: {}".format(input_dims[0][0]),
                "State: {}".format(input_dims[1][0]),
            ]
        else:
            input_dims = [x["dim"] for x in inputs_to_highlight]
            input_names = [x["name"] for x in inputs_to_highlight]
        self.input_dims = input_dims

        if len(input_dims) == 2:
            projection = None
            self.plot_2d = True
            self.linewidth = 3
        elif len(input_dims) == 3:
            projection = '3d'
            self.plot_2d = False
            self.linewidth = 1
            aspect = "auto"

        self.animate_fig, self.animate_axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
        # from nn_training.ground_robot_testing.controller_generation import display_ground_robot_control_field
        # display_ground_robot_control_field(name='complex_potential_field',ax=self.animate_axes)

        self.animate_axes.set_aspect(aspect)
        # Double Integrator
        self.animate_axes.set_xlim([-3.8, 5.64])
        self.animate_axes.set_ylim([-0.64, 2.5])

        # Ground Robot
        # self.animate_axes.set_xlim([-6, 3])
        # self.animate_axes.set_ylim([-7, 7])


        if show_samples:
            self.dynamics.show_samples(
                t_max * self.dynamics.dt,
                input_constraint,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=input_dims,
                zorder=sample_zorder,
                colors=sample_colors,
            )
        
        if show_trajectories:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                input_constraint,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=input_dims,
                zorder=sample_zorder,
                colors=sample_colors,
            )

        self.animate_axes.set_xlabel(input_names[0])
        self.animate_axes.set_ylabel(input_names[1])
        if not self.plot_2d:
            self.animate_axes.set_zlabel(input_names[2])

        # Plot the initial state set's boundaries
        if initial_set_color is None:
            initial_set_color = "tab:grey"
        rect = input_constraint.plot(self.animate_axes, input_dims, initial_set_color, zorder=initial_set_zorder, linewidth=self.linewidth, plot_2d=self.plot_2d)
        self.default_patches += rect

        # # Reachable sets
        # self.plot_reachable_sets(output_constraint, input_dims)

    def visualize(self,
        M,
        interior_M,
        output_constraint,
        iteration=0,
        title=None,
        reachable_set_color=None,
        reachable_set_zorder=None,
        reachable_set_ls=None,
        dont_tighten_layout=False,
        plot_lims=None,
        ):

        # Bring forward whatever default items should be in the plot
        # (e.g., MC samples, initial state set boundaries)
        self.animate_axes.patches = self.default_patches.copy()
        self.animate_axes.lines = self.default_lines.copy()

        # Actually draw the reachable sets and partitions
        self.plot_reachable_sets(output_constraint, self.input_dims, reachable_set_color=reachable_set_color, reachable_set_zorder=reachable_set_zorder, reachable_set_ls=reachable_set_ls)
        self.plot_partitions(M, output_constraint, self.input_dims)

        # from nn_closed_loop.utils.utils import range_to_polytope
        # target_range = np.array(
        #     [
        #         [-1, 1],
        #         [-1, 1]
        #     ]
        # )
        # A, b = range_to_polytope(target_range)

        # target_constraint = constraints.PolytopeConstraint(A,b)
        # self.plot_reachable_sets(
        #     target_constraint,
        #     self.input_dims,
        #     reachable_set_color='tab:green',
        #     reachable_set_zorder=4,
        #     reachable_set_ls='-'
        # )
        # initial_range = np.array(
        #     [
        #         [-5.5, -5],
        #         [-0.5, 0.5]
        #     ]
        # )
        # A, b = range_to_polytope(target_range)

        # initial_constraint = constraints.LpConstraint(initial_range)
        # self.plot_reachable_sets(
        #     initial_constraint,
        #     self.input_dims,
        #     reachable_set_color='tab:grey',
        #     reachable_set_zorder=5,
        #     reachable_set_ls='-'
        # )

        

        if plot_lims is not None:
            import ast
            plot_lims_arr = np.array(
                ast.literal_eval(plot_lims)
            )
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

    def plot_reachable_sets(self, constraint, dims, reachable_set_color=None, reachable_set_zorder=None, reachable_set_ls=None, reachable_set_lw=None):
        if reachable_set_color is None:
            reachable_set_color = "tab:blue"
        if reachable_set_ls is None:
            reachable_set_ls = "-"
        if reachable_set_lw is None:
            reachable_set_lw = self.linewidth
        fc_color = "None"
        constraint.plot(self.animate_axes, dims, reachable_set_color, fc_color=fc_color, zorder=reachable_set_zorder, plot_2d=self.plot_2d, linewidth=reachable_set_lw, ls=reachable_set_ls)

    # def plot_partition(self, constraint, bounds, dims, color):
    def plot_partition(self, constraint, dims, color):

        # This if shouldn't really be necessary -- someone is calling self.plot_partitions with something other than a (constraint, ___) element in M?
        if isinstance(constraint, np.ndarray):
            constraint = constraints.LpConstraint(range=constraint)

        constraint.plot(self.animate_axes, dims, color, linewidth=1, plot_2d=self.plot_2d)

    def plot_partitions(self, M, output_constraint, dims):

        # first = True
        for (input_constraint, output_range) in M:
            # if first:
            #     input_label = "Cell of Partition"
            #     output_label = "One Cell's Estimated Bounds"
            #     first = False
            # else:
            #     input_label = None
            #     output_label = None

            # Next state constraint of that cell
            output_constraint_ = constraints.LpConstraint(range=output_range)
            self.plot_partition(output_constraint_, dims, "grey")

            # Initial state constraint of that cell
            self.plot_partition(input_constraint, dims, "tab:red")

    def get_one_step_backprojection_set(
        self, output_constraint, input_constraint, propagator, num_partitions=None, overapprox=False
    ):
        input_constraint, info = propagator.get_one_step_backprojection_set(
            output_constraint, deepcopy(input_constraint), num_partitions=num_partitions, overapprox=overapprox
        )
        return input_constraint, info

    def get_backprojection_set(
        self, output_constraint, input_constraint, propagator, t_max, num_partitions=None, overapprox=False
    ):
        input_constraint_, info = propagator.get_backprojection_set(
            output_constraint, deepcopy(input_constraint), t_max, num_partitions=num_partitions, overapprox=overapprox
        )
        input_constraint = input_constraint_.copy()

        return input_constraint, info

    def get_backprojection_error(
        self, target_set, backprojection_sets, propagator, t_max
    ):
        errors = []

        if isinstance(target_set, constraints.LpConstraint):
            raise NotImplementedError
            output_estimated_range = output_constraint.range
            output_range_exact = self.get_sampled_out_range(
                input_constraint, propagator, t_max, num_samples=1000
            )
            num_steps = len(output_constraint.range)
            for t in range(num_steps):
                true_area = np.product(
                    output_range_exact[t][..., 1]
                    - output_range_exact[t][..., 0]
                )
                estimated_area = np.product(
                    output_estimated_range[t][..., 1]
                    - output_estimated_range[t][..., 0]
                )
                errors.append((estimated_area - true_area) / true_area)
        else:
            # This implementation should actually be moved to Lp constraint



            # Note: This compares the estimated polytope
            # with the "best" polytope with those facets.
            # There could be a much better polytope with lots of facets.
            true_verts_reversed = self.dynamics.get_true_backprojection_set(
                backprojection_sets[-1], target_set, 
                t_max, controller=propagator.network
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            # output_bs_exact = self.get_sampled_out_range(
            #     input_constraint, propagator, t_max, num_samples=1000,
            #     output_constraint=output_constraint
            # )
            num_steps = len(backprojection_sets)
            from scipy.spatial import ConvexHull
            for t in range(num_steps):
                # true_verts = pypoman.polygon.compute_polygon_hull(output_constraint.A, output_bs_exact[t])
                x_min = np.min(true_verts[:,t+1,:], axis=0)
                x_max = np.max(true_verts[:,t+1,:], axis=0)

                x_range = x_max-x_min
                true_area = np.prod(x_range)



                # true_hull = ConvexHull(true_verts[:, t+1, :])
                # true_area = true_hull.area
                estimated_verts = pypoman.polygon.compute_polygon_hull(backprojection_sets[t].A, backprojection_sets[t].b)
                estimated_hull = ConvexHull(estimated_verts)
                estimated_area = estimated_hull.volume
                errors.append((estimated_area - true_area) / true_area)
        final_error = errors[-1]
        avg_error = np.mean(errors)
        return final_error, avg_error, np.array(errors)

    def get_N_step_backprojection_set(
        self, output_constraint, input_constraint, propagator, t_max, num_partitions=None, overapprox=False
    ):
        # import pdb; pdb.set_trace()
        input_constraint_, info = propagator.get_N_step_backprojection_set(
            output_constraint, deepcopy(input_constraint), t_max, num_partitions=num_partitions, overapprox=overapprox
        )
        input_constraint = input_constraint_.copy()

        return input_constraint, info

    # def setup_visualization_multiple(
    #     self,
    #     input_constraint,
    #     output_constraint,
    #     propagator,
    #     input_dims_,
    #     prob_list=None,
    #     show_samples=True,
    #     outputs_to_highlight=None,
    #     color="g",
    #     line_style="-",
    # ):
    #     input_dims = input_dims_
    #     if isinstance(output_constraint, constraints.PolytopeConstraint):
    #         A_out = output_constraint.A
    #         b_out = output_constraint.b
    #         t_max = len(b_out)
    #     elif isinstance(output_constraint, constraints.LpConstraint):
    #         output_range = output_constraint.range
    #         output_p = output_constraint.p
    #         output_prob = prob_list
    #         t_max = len(output_range)
    #     else:
    #         raise NotImplementedError
    #     if isinstance(input_constraint, constraints.PolytopeConstraint):
    #         A_inputs = input_constraint.A
    #         b_inputs = input_constraint.b
    #         num_states = A_inputs.shape[-1]
    #         output_prob = prob_list

    #     elif isinstance(input_constraint, constraints.LpConstraint):
    #         input_range = input_constraint.range
    #         input_p = input_constraint.p
    #         num_states = input_range.shape[0]
    #         output_prob = prob_list
    #     else:
    #         raise NotImplementedError

    #     # scale = 0.05
    #     # x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
    #     # y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
    #     # self.animate_axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
    #     # self.animate_axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)

    #     # if show_samples:
    #     #    self.dynamics.show_samples(t_max*self.dynamics.dt, input_constraint, ax=self.animate_axes, controller=propagator.network, input_dims= input_dims_)

    #     # # Make a rectangle for the Exact boundaries
    #     # sampled_outputs = self.get_sampled_outputs(input_range, propagator)
    #     # if show_samples:
    #     #    self.animate_axes.scatter(sampled_outputs[...,output_dims[0]], sampled_outputs[...,output_dims[1]], c='k', marker='.', zorder=2,
    #     #        label="Sampled States")

    #     linewidth = 2
    #     if show_samples:
    #         self.dynamics.show_samples(
    #             t_max * self.dynamics.dt,
    #             input_constraint,
    #             ax=self.animate_axes,
    #             controller=propagator.network,
    #             input_dims=input_dims,
    #         )

    #     # Initial state set
    #     init_state_color = "k"

    #     if isinstance(input_constraint, constraints.PolytopeConstraint):
    #         # TODO: this doesn't use the computed input_dims...
    #         try:
    #             vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
    #         except:
    #             print(
    #                 "[warning] Can't visualize polytopic input constraints for >2 states. Need to implement this to it extracts input_dims."
    #             )
    #             raise NotImplementedError
    #         self.animate_axes.plot(
    #             [v[0] for v in vertices] + [vertices[0][0]],
    #             [v[1] for v in vertices] + [vertices[0][1]],
    #             color=color,
    #             linewidth=linewidth,
    #             linestyle=line_style,
    #             label="Initial States",
    #         )
    #     elif isinstance(input_constraint, constraints.LpConstraint):
    #         rect = Rectangle(
    #             input_range[input_dims, 0],
    #             input_range[input_dims[0], 1] - input_range[input_dims[0], 0],
    #             input_range[input_dims[1], 1] - input_range[input_dims[1], 0],
    #             fc="none",
    #             linewidth=linewidth,
    #             linestyle=line_style,
    #             edgecolor=init_state_color,
    #         )
    #         self.animate_axes.add_patch(rect)
    #         # self.default_patches[1].append(rect)
    #     else:
    #         raise NotImplementedError

    #     linewidth = 1.5
    #     # Reachable sets
    #     if prob_list is None:
    #         fc_color = "none"
    #     else:
    #         fc_color = "none"
    #         alpha = 0.17
    #     if isinstance(output_constraint, constraints.PolytopeConstraint):
    #         # TODO: this doesn't use the computed input_dims...
    #         for i in range(len(b_out)):
    #             vertices = pypoman.compute_polygon_hull(A_out, b_out[i])
    #             self.animate_axes.plot(
    #                 [v[0] for v in vertices] + [vertices[0][0]],
    #                 [v[1] for v in vertices] + [vertices[0][1]],
    #                 color=color,
    #                 label="$\mathcal{R}_" + str(i + 1) + "$",
    #             )
    #     elif isinstance(output_constraint, constraints.LpConstraint):
    #         if prob_list is None:
    #             for output_range_ in output_range:
    #                 rect = Rectangle(
    #                     output_range_[input_dims, 0],
    #                     output_range_[input_dims[0], 1]
    #                     - output_range_[input_dims[0], 0],
    #                     output_range_[input_dims[1], 1]
    #                     - output_range_[input_dims[1], 0],
    #                     fc=fc_color,
    #                     linewidth=linewidth,
    #                     linestyle=line_style,
    #                     edgecolor=color,
    #                 )
    #                 self.animate_axes.add_patch(rect)

    #         else:
    #             for output_range_, prob in zip(output_range, prob_list):
    #                 fc_color = cm.get_cmap("Greens")(prob)
    #                 rect = Rectangle(
    #                     output_range_[input_dims, 0],
    #                     output_range_[input_dims[0], 1]
    #                     - output_range_[input_dims[0], 0],
    #                     output_range_[input_dims[1], 1]
    #                     - output_range_[input_dims[1], 0],
    #                     fc=fc_color,
    #                     alpha=alpha,
    #                     linewidth=linewidth,
    #                     linestyle=line_style,
    #                     edgecolor=None,
    #                 )
    #                 self.animate_axes.add_patch(rect)

    #     else:
    #         raise NotImplementedError

    #     # self.default_patches = [[], []]
    #     # self.default_lines = [[], []]
    #     # self.default_patches[0] = [input_rect]

    #     # # Exact output range
    #     # color = 'black'
    #     # linewidth = 3
    #     # if self.interior_condition == "linf":
    #     #     output_range_exact = self.samples_to_range(sampled_outputs)
    #     #     output_range_exact_ = output_range_exact[self.output_dims_]
    #     #     rect = Rectangle(output_range_exact_[:2,0], output_range_exact_[0,1]-output_range_exact_[0,0], output_range_exact_[1,1]-output_range_exact_[1,0],
    #     #                     fc='none', linewidth=linewidth,edgecolor=color,
    #     #                     label="True Bounds ({})".format(label_dict[self.interior_condition]))
    #     #     self.animate_axes[1].add_patch(rect)
    #     #     self.default_patches[1].append(rect)
    #     # elif self.interior_condition == "lower_bnds":
    #     #     output_range_exact = self.samples_to_range(sampled_outputs)
    #     #     output_range_exact_ = output_range_exact[self.output_dims_]
    #     #     line1 = self.animate_axes[1].axhline(output_range_exact_[1,0], linewidth=linewidth,color=color,
    #     #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
    #     #     line2 = self.animate_axes[1].axvline(output_range_exact_[0,0], linewidth=linewidth,color=color)
    #     #     self.default_lines[1].append(line1)
    #     #     self.default_lines[1].append(line2)
    #     # elif self.interior_condition == "convex_hull":
    #     #     from scipy.spatial import ConvexHull
    #     #     self.true_hull = ConvexHull(sampled_outputs)
    #     #     self.true_hull_ = ConvexHull(sampled_outputs[...,output_dims].squeeze())
    #     #     line = self.animate_axes[1].plot(
    #     #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[0]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[0]]),
    #     #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[1]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[1]]),
    #     #         color=color, linewidth=linewidftypeth,
    #     #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
    #     #     self.default_lines[1].append(line[0])
    #     # else:
    #     #     raise NotImplementedError
