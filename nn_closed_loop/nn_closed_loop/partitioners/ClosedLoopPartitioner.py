import numpy as np
import nn_partition.partitioners as partitioners
from pandas.core.indexing import convert_to_index_sliceable
import pypoman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import nn_closed_loop.constraints as constraints
from nn_closed_loop.utils.utils import range_to_polytope
from copy import deepcopy
import os

from nn_closed_loop.constraints.ClosedLoopConstraints import PolytopeConstraint


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
        extra_set_color=None,
        extra_set_zorder=None,
        sample_zorder=None,
        sample_colors=None,
        extra_constraint=None,
        plot_lims=None
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

        # import pdb; pdb.set_trace()
        if len(input_dims) == 2:
            projection = None
            self.plot_2d = True
            self.linewidth = 2
        elif len(input_dims) == 3:
            projection = '3d'
            self.plot_2d = False
            self.linewidth = 1
            aspect = "auto"

        self.animate_fig, self.animate_axes = plt.subplots(1, 1, subplot_kw=dict(projection=projection))
        # from nn_closed_loop.utils.controller_generation import display_ground_robot_control_field
        # display_ground_robot_control_field(name='complex_potential_field',ax=self.animate_axes)

        self.animate_axes.set_aspect(aspect)


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

        if extra_set_color is None:
            extra_set_color = "tab:red"
        if extra_constraint is not None:
            for i in range(len(extra_constraint)):
                rect = extra_constraint[i].plot(self.animate_axes, input_dims, extra_set_color, zorder=extra_set_zorder, linewidth=self.linewidth, plot_2d=self.plot_2d)
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
        # self.plot_partitions(M, output_constraint, self.input_dims)

        from nn_closed_loop.utils.utils import range_to_polytope
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
        self, output_constraint, input_constraint, propagator, num_partitions=None, overapprox=False, refined=False
    ):
        input_constraint, info = propagator.get_one_step_backprojection_set(
            output_constraint, deepcopy(input_constraint), num_partitions=num_partitions, overapprox=overapprox, refined=refined
        )
        return input_constraint, info

    def get_backprojection_set(
        self, output_constraint, input_constraint, propagator, t_max, num_partitions=None, overapprox=False, refined=False
    ):
        input_constraint_, info = propagator.get_backprojection_set(
            output_constraint, deepcopy(input_constraint), t_max, num_partitions=num_partitions, overapprox=overapprox, refined=refined
        )
        input_constraint = input_constraint_.copy()

        return input_constraint, info

    def get_backprojection_error(
        self, target_set, backprojection_sets, propagator, t_max
    ):
        errors = []
        from scipy.spatial import ConvexHull

        if isinstance(target_set, constraints.LpConstraint):
            # raise NotImplementedError
            # output_estimated_range = output_constraint.range
            # output_range_exact = self.get_sampled_out_range(
            #     input_constraint, propagator, t_max, num_samples=1000
            # )
            # num_steps = len(output_constraint.range)
            # for t in range(num_steps):
            #     true_area = np.product(
            #         output_range_exact[t][..., 1]
            #         - output_range_exact[t][..., 0]
            #     )
            #     estimated_area = np.product(
            #         output_estimated_range[t][..., 1]
            #         - output_estimated_range[t][..., 0]
            #     )
            #     errors.append((estimated_area - true_area) / true_area)
            Ats, bts = range_to_polytope(target_set.range)
            target_set_poly = PolytopeConstraint(A=Ats, b=bts)
            true_verts_reversed = self.dynamics.get_true_backprojection_set(
                backprojection_sets[-1], target_set, 
                t_max, controller=propagator.network
            )
            true_verts = np.flip(true_verts_reversed, axis=1)
            num_steps = len(backprojection_sets)

            for t in range(num_steps):
                # true_verts = pypoman.polygon.compute_polygon_hull(output_constraint.A, output_bs_exact[t])
                x_min = np.min(true_verts[:,t+1,:], axis=0)
                x_max = np.max(true_verts[:,t+1,:], axis=0)

                x_range = x_max-x_min
                true_area = np.prod(x_range)

                # true_hull = ConvexHull(true_verts[:, t+1, :])
                # true_area = true_hull.volume

                Abp, bbp = range_to_polytope(backprojection_sets[t].range)
                estimated_verts = pypoman.polygon.compute_polygon_hull(Abp, bbp)
                estimated_hull = ConvexHull(estimated_verts)
                estimated_area = estimated_hull.volume
                

                errors.append((estimated_area - true_area) / true_area)
            # import pdb; pdb.set_trace()
        else:
            # This implementation should actually be moved to Lp constraint


            # import pdb; pdb.set_trace()
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
            
            for t in range(num_steps):
                # true_verts = pypoman.polygon.compute_polygon_hull(output_constraint.A, output_bs_exact[t])
                x_min = np.min(true_verts[:,t+1,:], axis=0)
                x_max = np.max(true_verts[:,t+1,:], axis=0)

                x_range = x_max-x_min
                true_area = np.prod(x_range)

                # true_hull = ConvexHull(true_verts[:, t+1, :])
                # true_area = true_hull.volume
                estimated_verts = pypoman.polygon.compute_polygon_hull(backprojection_sets[t].A[0], backprojection_sets[t].b[0])
                estimated_hull = ConvexHull(estimated_verts)
                estimated_area = estimated_hull.volume
                


                errors.append((estimated_area - true_area) / true_area)
            import pdb; pdb.set_trace()
        
        final_error = errors[-1]
        avg_error = np.mean(errors)
        return final_error, avg_error, np.array(errors)

    def get_N_step_backprojection_set(
        self, output_constraint, input_constraint, propagator, t_max, num_partitions=None, overapprox=False
    ):
        input_constraint_, info = propagator.get_N_step_backprojection_set(
            output_constraint, deepcopy(input_constraint), t_max, num_partitions=num_partitions, overapprox=overapprox
        )
        input_constraint = input_constraint_.copy()

        return input_constraint, info
