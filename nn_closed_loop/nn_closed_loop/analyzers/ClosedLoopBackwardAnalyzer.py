from codecs import backslashreplace_errors
from copy import deepcopy
from re import S
from turtle import back
import nn_partition.analyzers as analyzers
import nn_closed_loop.partitioners as partitioners
import nn_closed_loop.propagators as propagators
from nn_partition.utils.utils import samples_to_range, get_sampled_outputs
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'


class ClosedLoopBackwardAnalyzer(analyzers.Analyzer):
    def __init__(self, torch_model, dynamics):
        self.torch_model = torch_model
        self.dynamics = dynamics
        analyzers.Analyzer.__init__(self, torch_model=torch_model)

        self.true_backprojection_set_color = 'k'#'darkgreen'
        self.estimated_backprojection_set_color = 'tab:blue'
        self.estimated_one_step_backprojection_set_color = 'tab:orange'
        self.estimated_backprojection_partitioned_set_color = 'tab:gray'
        self.backreachable_set_color = 'tab:cyan'
        self.target_set_color = 'tab:red'
        self.initial_set_color = 'k'
        
        self.true_backprojection_set_zorder = 3
        self.estimated_backprojection_set_zorder = 2
        self.estimated_one_step_backprojection_set_zorder = -1
        self.estimated_backprojection_partitioned_set_zorder = 5
        self.backreachable_set_zorder = -1
        self.target_set_zorder = 1
        self.initial_set_zorder = 1

        
        self.true_backprojection_set_linestyle = '-'
        self.estimated_backprojection_set_linestyle = '-'
        self.estimated_one_step_backprojection_set_linestyle = '-'
        self.estimated_backprojection_partitioned_set_linestyle = '-'
        self.backreachable_set_linestyle = '--'
        self.target_set_linestyle = '-'
        
        
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

    def instantiate_partitioner(self, partitioner, hyperparams):
        return partitioners.partitioner_dict[partitioner](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def instantiate_propagator(self, propagator, hyperparams):
        return propagators.propagator_dict[propagator](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def get_one_step_backprojection_set(self, output_constraint, input_constraint, num_partitions=None, overapprox=False, refined=False, heuristic='guided', all_lps=False, slow_cvxpy=False):
        backprojection_set, info = self.partitioner.get_one_step_backprojection_set(
            output_constraint, input_constraint, self.propagator, num_partitions=num_partitions, overapprox=overapprox, refined=refined, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy
        )
        return backprojection_set, info

    def get_backprojection_set(self, output_constraint, input_constraint, t_max, num_partitions=None, overapprox=False, refined=False, heuristic='guided', all_lps=False, slow_cvxpy=False):
        backprojection_set, info = self.partitioner.get_backprojection_set(
            output_constraint, input_constraint, self.propagator, t_max, num_partitions=num_partitions, overapprox=overapprox, refined=refined, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy
        )
        return backprojection_set, info
    
    def get_N_step_backprojection_set(self, output_constraint, input_constraint, t_max, num_partitions=None, overapprox=False, refined=False, heuristic='guided', all_lps=False, slow_cvxpy=False):
        backprojection_set, info = self.partitioner.get_N_step_backprojection_set(
            output_constraint, input_constraint, self.propagator, t_max, num_partitions=num_partitions, overapprox=overapprox, refined=refined, heuristic=heuristic, all_lps=all_lps, slow_cvxpy=slow_cvxpy
        )
        return backprojection_set, info

    def get_backprojection_error(self, target_set, backprojection_sets, t_max, backreachable_sets=None):
        return self.partitioner.get_backprojection_error(
            target_set, backprojection_sets, self.propagator, t_max, backreachable_sets=backreachable_sets
        )

    def visualize(
        self,
        input_constraint_list,
        output_constraint_list,
        info_list,
        initial_constraint=None,
        show=True,
        show_samples=False,
        show_trajectories=False,
        show_convex_hulls=False,
        aspect="auto",
        labels={},
        plot_lims=None,
        inputs_to_highlight=None,
        controller_name=None,
        show_BReach=False,
    ):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)
        if inputs_to_highlight is None:
            inputs_to_highlight=[
                {"dim": [0], "name": "$x$"},
                {"dim": [1], "name": "$\dot{x}$"},
            ]
        self.partitioner.setup_visualization(
            input_constraint_list[0][0],
            output_constraint_list[0].get_t_max(),
            self.propagator,
            # show_samples=False,
            show_samples=show_samples,
            inputs_to_highlight=inputs_to_highlight,
            aspect=aspect,
            initial_set_color=self.estimated_backprojection_set_color,
            initial_set_zorder=self.estimated_backprojection_set_zorder,
            extra_constraint = initial_constraint,
            extra_set_color=self.initial_set_color,
            extra_set_zorder=self.initial_set_zorder,
            controller_name=controller_name
        )

        for i in range(len(output_constraint_list)):
            self.visualize_single_set(
                input_constraint_list[i],
                output_constraint_list[i],
                initial_constraint=initial_constraint,
                show_samples=show_samples,
                show_trajectories=show_trajectories,
                show_convex_hulls=show_convex_hulls,
                show=show,
                labels=labels,
                aspect=aspect,
                plot_lims=plot_lims,
                inputs_to_highlight=inputs_to_highlight,
                show_BReach=show_BReach,
                **info_list[i]
            )
        self.partitioner.animate_fig.tight_layout()

        if plot_lims is not None:
            import ast
            plot_lims_arr = np.array(
                ast.literal_eval(plot_lims)
            )
            plt.xlim(plot_lims_arr[0])
            plt.ylim(plot_lims_arr[1])

        if "save_name" in info_list[0] and info_list[0]["save_name"] is not None:
            plt.savefig(info_list[0]["save_name"])

        if show:
            plt.show()
        else:
            plt.close()
    
    
    
    
    def visualize_single_set(
        self,
        input_constraints,
        output_constraint,
        initial_constraint=None,
        show=True,
        show_samples=False,
        show_trajectories=False,
        show_convex_hulls=False,
        aspect="auto",
        labels={},
        plot_lims=None,
        inputs_to_highlight=None,
        show_BReach=False,
        **kwargs
    ):
        # import pdb; pdb.set_trace()
        # if inputs_to_highlight is None:
        #     inputs_to_highlight=[
        #         {"dim": [0], "name": "$x$"},
        #         {"dim": [1], "name": "$\dot{x}$"},
        #     ]  

        # import nn_closed_loop.constraints as constraints
        # from nn_closed_loop.utils.utils import range_to_polytope
        # target_range = np.array(
        #     [
        #         [-1, 1],
        #         [-1, 1]
        #     ]
        # )
        # A, b = range_to_polytope(target_range)

        # target_constraint = constraints.PolytopeConstraint(A,b)
        # self.partitioner.plot_reachable_sets(
        #     target_constraint,
        #     self.partitioner.input_dims,
        #     reachable_set_color='tab:green',
        #     reachable_set_zorder=4,
        #     reachable_set_ls='-'
        # )
        # initial_range = np.array(
        #     [
        #         [-5.5, -4.5],
        #         [-0.5, 0.5]
        #     ]
        # )
        # A, b = range_to_polytope(target_range)
        
        # initial_constraint = constraints.LpConstraint(initial_range)
        # self.partitioner.plot_reachable_sets(
        #     initial_constraint,
        #     self.partitioner.input_dims,
        #     reachable_set_color='k',
        #     reachable_set_zorder=5,
        #     reachable_set_ls='-'
        # )

        # # import pdb; pdb.set_trace()
        # self.dynamics.show_trajectories(
        #         len(input_constraints) * self.dynamics.dt,
        #         initial_constraint,
        #         ax=self.partitioner.animate_axes,
        #         controller=self.propagator.network,
        #         zorder=1,
        #     )
        # from colour import Color
        # orange = Color("orange")
        # colors = list(orange.range_to(Color("purple"),len(input_constraints)))
        # import matplotlib as mpl
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(plt.gca())
        # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", [color.hex for color in colors])
        # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # axins = inset_axes(self.partitioner.animate_axes,
        #             width="5%",  
        #             height="100%",
        #             loc='right',
        #             borderpad=0.15
        #            )

        # cb1 = mpl.colorbar.ColorbarBase(axins, cmap=cmap, orientation='vertical', label='t (s)', values=range(len(input_constraints)))
        
        # plt.gcf().add_axes(ax_cb)

        # from colour import Color
        # orange = Color("orange")
        # colors = list(orange.range_to(Color("purple"),len(input_constraints)))
        # import matplotlib as mpl
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(plt.gca())
        # # ax_cb = divider.append_axes("right", size="50%", pad=0.05)
        # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        # # cax = self.partitioner.animate_fig.add_axes([0.9, 0, 0.05, 1])
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", [color.hex for color in colors])
        # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical', label='t (s)', values=range(len(input_constraints)))
        
        # plt.gcf().add_axes(ax_cb)

        # import pdb; pdb.set_trace()
        # import numpy as np
        # start_region = np.array(
        #     [
        #         [-2.2200, -2.1542],
        #         [0.6297, 0.6935]
        #     ]
        # )
        # # start_region = np.array(
        # #     [
        # #         [-2.3754, -2.2702],
        # #         [0.5622, 0.6644]
        # #     ]
        # # )
        # # start_region = np.array(
        # #     [
        # #         [-1.8, -1.75],
        # #         [0.5622, 0.6644]
        # #     ]
        # # )
        # crown_bounds = []
        # for step in kwargs.get('per_timestep', []):
        #     for info in step:
        #         crown_bounds.append((info.get('upper_A', None), info.get('lower_A', None), info.get('upper_sum_b', None), info.get('lower_sum_b', None)))
        # crown_bounds.reverse()

        # self.plot_relaxed_sequence([start_region], input_constraints[0:], crown_bounds, output_constraint, marker="^")

        # start_region = np.array(
        #     [
        #         [-1.3212, -1.2161],
        #         [0.5625, 0.6643]
        #     ]
        # )
        # self.plot_relaxed_sequence([start_region], input_constraints[0:], crown_bounds, output_constraint, marker='s')

        # Plot all our input constraints (i.e., our backprojection set estimates)
        import nn_closed_loop.constraints as constraints
        for j,ic in enumerate(input_constraints[0:]):
            if isinstance(output_constraint, constraints.LpConstraint):
                # rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, colors[j].hex_l, zorder=self.estimated_backprojection_set_zorder, linewidth=self.partitioner.linewidth, plot_2d=self.partitioner.plot_2d)
                rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, self.estimated_backprojection_set_color, zorder=self.estimated_backprojection_set_zorder, linewidth=self.partitioner.linewidth, plot_2d=self.partitioner.plot_2d)
                self.partitioner.default_patches += rect
            elif isinstance(output_constraint, constraints.RotatedLpConstraint):
                ic.plot(self.partitioner.animate_axes)#, self.partitioner.input_dims, self.estimated_backprojection_set_color, zorder=self.estimated_backprojection_set_zorder, linewidth=self.partitioner.linewidth, plot_2d=self.partitioner.plot_2d)
        # Show the target set
        self.plot_target_set(
            output_constraint,
            color=self.target_set_color,
            zorder=self.target_set_zorder,
            linestyle=self.target_set_linestyle,
        )

        # Show the "true" N-Step backprojection set as a convex hull
        # backreachable_set = kwargs['per_timestep'][-1]['backreachable_set']
        target_set = output_constraint
        t_max = len(kwargs['per_timestep'])*self.dynamics.dt
        if show_convex_hulls:
            try:
                # import pdb; pdb.set_trace()
                self.plot_true_backprojection_sets(
                    input_constraints[-1],
                    # backreachable_set, 
                    target_set,
                    t_max=t_max,
                    color=self.true_backprojection_set_color,
                    zorder=10,#self.true_backprojection_set_zorder,
                    linestyle=self.true_backprojection_set_linestyle,
                    show_samples=True,
                )
            except:
                print('faileeddd')
                pass
        # # If they exist, plot all our loose input constraints (i.e., our one-step backprojection set estimates)
        # # TODO: Make plotting these optional via a flag
        # if show_BReach:
        #     for info in kwargs.get('per_timestep', []):
        #         ic = info.get('one_step_backprojection_overapprox', None)
        #         if ic is None: continue
        #         rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, self.estimated_one_step_backprojection_set_color, zorder=self.estimated_one_step_backprojection_set_zorder, linewidth=self.partitioner.linewidth, plot_2d=self.partitioner.plot_2d)
        #         self.partitioner.default_patches += rect


        # # TODO: pass these as flags
        show_backreachable_set = False
        if show_backreachable_set:
            # import pdb; pdb.set_trace()
            for step in kwargs.get('per_timestep', [])[::4]:
                for info in step:
                    ic = info.get('backreachable_set', None)
                    if ic is None: continue
                    rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, self.backreachable_set_color, zorder=self.backreachable_set_zorder, linewidth=self.partitioner.linewidth, plot_2d=self.partitioner.plot_2d)
                    self.partitioner.default_patches += rect

        show_backreachable_set_partitions = False
        if show_backreachable_set_partitions:
            for step in [kwargs.get('per_timestep', [])[1]]:
                for info in step:
                    for partition in info.get('br_set_partitions', None):
                        ic = partition
                        if ic is None: continue
                        rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, self.estimated_backprojection_partitioned_set_color, zorder=self.estimated_backprojection_partitioned_set_zorder, linewidth=self.partitioner.linewidth*0.5, plot_2d=self.partitioner.plot_2d)
                        self.partitioner.default_patches += rect
        
        show_backprojection_set_partitions = False
        if show_backprojection_set_partitions:
            for step in kwargs.get('per_timestep', [])[::4]:
                for info in step:
                    for partition in info.get('bp_set_partitions', None):
                        ic = partition
                        if ic is None: continue
                        rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, 'grey', zorder=11, linewidth=self.partitioner.linewidth*0.3, plot_2d=self.partitioner.plot_2d)
                        self.partitioner.default_patches += rect

        # import pdb; pdb.set_trace()
        show_target_partition_bps = False
        if show_target_partition_bps:
            for step in kwargs.get('per_timestep', [])[::4]:
                for info in step:
                    ic = info.get('bp_set', None)
                    if ic is None: continue
                    rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, 'm', zorder=10, linewidth=self.partitioner.linewidth*0.82, plot_2d=self.partitioner.plot_2d)
                    self.partitioner.default_patches += rect

        # show_nstep_backprojection_set_partitions = False
        # if show_nstep_backprojection_set_partitions:
        #     for info in kwargs.get('per_timestep', []):
        #         for partition in info.get('nstep_bp_set_partitions', None):
        #             ic = partition
        #             if ic is None: continue
        #             rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, 'm', zorder=10, linewidth=self.partitioner.linewidth*0.75, plot_2d=self.partitioner.plot_2d)
        #             self.partitioner.default_patches += rect
        
        # show_mar = False
        # if show_mar:
        #     for info in kwargs.get('per_timestep', []):
        #         mar_hull = info.get('mar_hull', None)
        #         from scipy.spatial import ConvexHull, convex_hull_plot_2d
        #         convex_hull_plot_2d(mar_hull, ax=self.partitioner.animate_axes)
                    

        # Sketchy workaround to trajectories not showing up
        import numpy as np
        import nn_closed_loop.constraints as constraints
        # x0 = np.array(
        #     [  # (num_inputs, 2)
        #         [-5.5, -5.0],  # x0min, x0max
        #         [-0.5, 0.5],  # x1min, x1max
        #     ]
        # )
        x0 = np.array(
            [  # (num_inputs, 2)
                [-5.5, -5.0],  # x0min, x0max
                [-0.5, 0.5],  # x1min, x1max
            ]
        )
        # import pdb; pdb.set_trace()
        x0_constraint = constraints.LpConstraint(
            range=x0, p=np.inf
        )
        # import pdb; pdb.set_trace()
        if show_trajectories and initial_constraint is not None:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                initial_constraint[0],
                input_dims=inputs_to_highlight,
                ax=self.partitioner.animate_axes,
                controller=self.propagator.network,
            ) 

        import numpy as np
        import nn_closed_loop.constraints as constraints
        x0 = np.array(
            [  # (num_inputs, 2)
                [-5.5, -4.5],  # x0min, x0max
                [-0.5, 0.5],  # x1min, x1max
            ]
        )
        # x0 = np.array( # tree_trunks_vs_quadrotor_12__
        #         [  # (num_inputs, 2)
        #             [-6.5,-0.25, 2, 1.95, -0.01, -0.01],
        #             [-6, 0.25, 2.5, 2.0, 0.01, 0.01],
        #         ]
        #     ).T
        # x0 = np.array(
        #     [  # (num_inputs, 2)
        #         [-0.5, 0.5],
        #         [-0.5, 0.5],
        #         [-0.01, 0.01],
        #         [-0.01, 0.01],
        #     ]
        # )
        # x0 = np.array(
        #         [  # (num_inputs, 2)
        #             [-2-0.25, -4+0.25],  # x0min, x0max
        #             [-3., 3.],  # x1min, x1max
        #             [0.49, 0.50],
        #             [-0.01, 0.01]
        #         ]
        #     )
        # x0_constraint = constraints.LpConstraint(
        #     range=x0, p=np.inf
        # )
        # input_dims = [x["dim"] for x in inputs_to_highlight]
        # self.dynamics.show_trajectories(
        #     len(input_constraints) * self.dynamics.dt,
        #     x0_constraint,
        #     input_dims=input_dims,
        #     ax=self.partitioner.animate_axes,
        #     controller=self.propagator.network,
        #     zorder=10
        # ) 

        # # initial_range = np.array( # tree_trunks_vs_quadrotor_12__
        # #     [  # (num_inputs, 2)
        # #         [-6.5, 0.25-0.25, 2, .95, -0.01, -0.01],
        # #         [-6, 0.25+0.25, 2.5, 1.0, 0.01, 0.01],
        # #     ]
        # # ).T

        # initial_constraint = constraints.LpConstraint(x0)
        # self.partitioner.plot_reachable_sets(
        #     initial_constraint,
        #     input_dims,
        #     reachable_set_color='k',
        #     reachable_set_zorder=11,
        #     reachable_set_ls='-'
        # )

            


        # self.partitioner.animate_axes.legend(
        #     bbox_to_anchor=(0, 1.02, 1, 0.2),
        #     loc="lower left",
        #     mode="expand",
        #     borderaxespad=0,
        #     ncol=1,
        # )

        # self.partitioner.animate_fig.tight_layout()

        # if "save_name" in kwargs and kwargs["save_name"] is not None:
        #     plt.savefig(kwargs["save_name"])

        # if show:
        #     plt.show()
        # else:
        #     plt.close()
        # self.partitioner.animate_axes.set_xlim([-17.2, 3])
        # self.partitioner.animate_axes.set_ylim([-7.2, 7.2])


        # from colour import Color
        # orange = Color("orange")
        # colors = list(orange.range_to(Color("purple"),len(input_constraints)))
        # import matplotlib as mpl
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(plt.gca())
        # ax_cb = divider.append_axes("right", size="5%", pad=0.05)
        # # cax = self.partitioner.animate_fig.add_axes([0.9, 0, 0.05, 1])
        # cmap = mpl.colors.LinearSegmentedColormap.from_list("", [color.hex for color in colors])
        # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical', label='t (s)', values=range(len(input_constraints)))
        # plt.gcf().add_axes(ax_cb)

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

    def plot_backreachable_set(self, backreachable_set, color='cyan', zorder=None, linestyle='-'):
        self.partitioner.plot_reachable_sets(
            backreachable_set,
            self.partitioner.input_dims,
            reachable_set_color=color,
            reachable_set_zorder=zorder,
            reachable_set_ls=linestyle
        )

    def plot_target_set(self, target_set, color='cyan', zorder=None, linestyle='-',linewidth=2.5):
        self.partitioner.plot_reachable_sets(
            target_set,
            self.partitioner.input_dims,
            reachable_set_color=color,
            reachable_set_zorder=zorder,
            reachable_set_ls=linestyle,
            reachable_set_lw=linewidth
        )

    def plot_tightened_backprojection_set(self, tightened_set, color='darkred', zorder=None, linestyle='-'):
        self.partitioner.plot_reachable_sets(
            tightened_set,
            self.partitioner.input_dims,
            reachable_set_color=color,
            reachable_set_zorder=zorder,
            reachable_set_ls=linestyle
        )

    def plot_backprojection_set(self, backreachable_set, target_set, show_samples=False, color='g', zorder=None, linestyle='-'):

        # Sample a bunch of pts from our "true" backreachable set
        # (it's actually the tightest axis-aligned rectangle around the backreachable set)
        # and run them forward 1 step in time under the NN policy
        xt_samples_from_backreachable_set, xt1_from_those_samples = self.partitioner.dynamics.get_state_and_next_state_samples(
            backreachable_set,
            num_samples=1e5,
            controller=self.propagator.network,
        )

        # Find which of the xt+1 points actually end up in the target set
        within_constraint_inds = np.where(
            np.all(
                (
                    np.dot(target_set.A, xt1_from_those_samples.T)
                    - np.expand_dims(target_set.b[0], axis=-1)
                )
                <= 0,
                axis=0,
            )
        )
        xt_samples_inside_backprojection_set = xt_samples_from_backreachable_set[(within_constraint_inds)]

        if show_samples:
            xt1_from_those_samples_ = xt1_from_those_samples[(within_constraint_inds)]

            # Show samples from inside the backprojection set and their futures under the NN (should end in target set)
            self.partitioner.dynamics.show_samples(
                None,
                None,
                ax=self.partitioner.animate_axes,
                controller=None,
                input_dims=self.partitioner.input_dims,
                zorder=1,
                xs=np.dstack([xt_samples_inside_backprojection_set, xt1_from_those_samples_]).transpose(0, 2, 1),
                colors=None
            )

            # Show samples from inside the backreachable set and their futures under the NN (don't necessarily end in target set)
            # self.partitioner.dynamics.show_samples(
            #     None,
            #     None,
            #     ax=self.partitioner.animate_axes,
            #     controller=None,
            #     input_dims=self.partitioner.input_dims,
            #     zorder=0,
            #     xs=np.dstack([xt_samples_from_backreachable_set, xt1_from_those_samples]).transpose(0, 2, 1),
            #     colors=['g', 'r']
            # )

        # Compute and draw a convex hull around all the backprojection set samples
        # This is our "true" backprojection set -- but...
        # it is sampling-based so that is an under-approximation,
        # and it is a convex hull, so that is an over-approximation,
        # and it is computed only for one step, so that's an over-approximation
        conv_hull_line = plot_convex_hull(
            xt_samples_inside_backprojection_set,
            dims=self.partitioner.input_dims,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            zorder=zorder,
            label='Backprojection Set (True)',
            axes=self.partitioner.animate_axes,
        )
        # self.default_lines[self.output_axis].append(conv_hull_line[0])

    def plot_true_backprojection_sets(self, backreachable_set, target_set, t_max, show_samples=False, color='g', zorder=None, linestyle='-'):

        # Sample a bunch of pts from our "true" backreachable set
        # (it's actually the tightest axis-aligned rectangle around the backreachable set)
        # and run them forward t_max steps in time under the NN policy
        x_samples_inside_backprojection_set = self.dynamics.get_true_backprojection_set(backreachable_set, target_set, t_max=t_max, controller=self.propagator.network)

        if show_samples:
            # raise NotImplementedError
            # xt1_from_those_samples_ = xt1_from_those_samples[(within_constraint_inds)]

            # Show samples from inside the backprojection set and their futures under the NN (should end in target set)
            self.partitioner.dynamics.show_samples(
                None,
                None,
                ax=self.partitioner.animate_axes,
                controller=None,
                input_dims=self.partitioner.input_dims,
                zorder=1,
                xs=x_samples_inside_backprojection_set, # np.dstack([x_samples_inside_backprojection_set, xt1_from_those_samples_]).transpose(0, 2, 1),
                colors=None
            )
            print(self.partitioner.animate_axes.get_ylabel())

            # Show samples from inside the backreachable set and their futures under the NN (don't necessarily end in target set)
            # self.partitioner.dynamics.show_samples(
            #     None,
            #     None,
            #     ax=self.partitioner.animate_axes,
            #     controller=None,
            #     input_dims=self.partitioner.input_dims,
            #     zorder=0,
            #     xs=np.dstack([xt_samples_from_backreachable_set, xt1_from_those_samples]).transpose(0, 2, 1),
            #     colors=['g', 'r']
            # )

        # Compute and draw a convex hull around all the backprojection set samples
        # This is our "true" backprojection set -- but...
        # it is sampling-based so that is an under-approximation,
        # and it is a convex hull, so that is an over-approximation.
        for t in range(t_max):
            conv_hull_line = plot_convex_hull(
                x_samples_inside_backprojection_set[:, t, :],
                dims=self.partitioner.input_dims,
                color=color,
                linewidth=2,
                linestyle=linestyle,
                zorder=zorder,
                label='Backprojection Set (True)',
                axes=self.partitioner.animate_axes,
            )
        # self.default_lines[self.output_axis].append(conv_hull_line[0])

    def plot_relaxed_sequence(self, start_region, bp_sets, crown_bounds, target_set, marker='o'):
        bp_sets = deepcopy(bp_sets)
        bp_sets.reverse()
        bp_sets.append(target_set)
        sequences = self.partitioner.dynamics.get_relaxed_backprojection_samples(start_region, bp_sets, crown_bounds, target_set)

        ax=self.partitioner.animate_axes,
        x=[]
        y=[]
        for seq in [sequences[0]]:
            for point in seq:
                x.append(point[0])
                y.append(point[1])
        
        # import pdb; pdb.set_trace()
        ax[0].scatter(x, y, s=40, zorder=15, c='k', marker=marker)



def plot_convex_hull(samples, dims, color, linewidth, linestyle, zorder, label, axes):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(
        samples[..., dims].squeeze()
    )
    line = axes.plot(
        np.append(
            samples[hull.vertices][
                ..., dims[0]
            ],
            samples[hull.vertices[0]][
                ..., dims[0]
            ],
        ),
        np.append(
            samples[hull.vertices][
                ..., dims[1]
            ],
            samples[hull.vertices[0]][
                ..., dims[1]
            ],
        ),
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        label=label
    )
    return line
