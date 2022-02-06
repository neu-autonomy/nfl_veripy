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

        self.reachable_set_color = 'tab:green'
        self.reachable_set_zorder = 4
        self.initial_set_color = 'tab:red'
        self.initial_set_zorder = 4

    def instantiate_partitioner(self, partitioner, hyperparams):
        return partitioners.partitioner_dict[partitioner](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def instantiate_propagator(self, propagator, hyperparams):
        return propagators.propagator_dict[propagator](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def get_one_step_backprojection_set(self, output_constraint, input_constraint, num_partitions=None, overapprox=False):
        backprojection_set, info = self.partitioner.get_one_step_backprojection_set(
            output_constraint, input_constraint, self.propagator, num_partitions=num_partitions, overapprox=overapprox
        )
        return backprojection_set, info

    def get_backprojection_set(self, output_constraint, input_constraint, t_max, num_partitions=None, overapprox=False):
        backprojection_set, info = self.partitioner.get_backprojection_set(
            output_constraint, input_constraint, self.propagator, t_max, num_partitions=num_partitions, overapprox=overapprox
        )
        return backprojection_set, info

    def visualize(
        self,
        input_constraint,
        output_constraint,
        show=True,
        show_samples=False,
        aspect="auto",
        labels={},
        plot_lims=None,
        **kwargs
    ):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(
            input_constraint[0],
            output_constraint.get_t_max(),
            self.propagator,
            show_samples=False,
            # show_samples=show_samples,
            inputs_to_highlight=[
                {"dim": [0], "name": "$x$"},
                {"dim": [1], "name": "$\dot{x}$"},
            ],
            aspect=aspect,
            initial_set_color=self.initial_set_color,
            initial_set_zorder=self.initial_set_zorder,
        )

        # plot all our input constraints
        for ic in input_constraint[1:]:
            rect = ic.plot(self.partitioner.animate_axes, self.partitioner.input_dims, self.initial_set_color, zorder=self.initial_set_zorder, linewidth=self.partitioner.linewidth, plot_2d=self.partitioner.plot_2d)
            self.partitioner.default_patches += rect

        self.partitioner.visualize(
            kwargs.get(
                "exterior_partitions", kwargs.get("all_partitions", [])
            ),
            kwargs.get("interior_partitions", []),
            output_constraint,
            reachable_set_color=self.reachable_set_color,
            reachable_set_zorder=self.reachable_set_zorder,
            plot_lims=plot_lims,
        )

        # Show the backreachable set (tightest rectangular outer-bounds via our LP)
        for t in range(len(input_constraint)):
            backreachable_set = kwargs['per_timestep'][t]['backreachable_set']
            target_set = kwargs['per_timestep'][t]['target_set']
            self.plot_backreachable_set(
                backreachable_set,
                color='tab:purple',
                zorder=1,
                linestyle='--',
            )
            self.plot_backprojection_set(
                backreachable_set,
                target_set,
                color='tab:pink',
                zorder=1,
                linestyle='--',
            )
            self.plot_target_set(
                target_set,
                color='black',
                zorder=1,
                linestyle='--',
            )

        # Show the "true" N-Step backprojection set as a convex hull
        backreachable_set = kwargs['per_timestep'][-1]['backreachable_set']
        target_set = kwargs['per_timestep'][0]['target_set']
        t_max = len(kwargs['per_timestep'])
        self.plot_true_backprojection_sets(
            backreachable_set,
            target_set,
            t_max=t_max,
            color='yellow',
            zorder=2,
            linestyle='-',
        )

        # self.partitioner.animate_axes.legend(
        #     bbox_to_anchor=(0, 1.02, 1, 0.2),
        #     loc="lower left",
        #     mode="expand",
        #     borderaxespad=0,
        #     ncol=1,
        # )

        self.partitioner.animate_fig.tight_layout()

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])

        if show:
            plt.show()
        else:
            plt.close()

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

    def plot_target_set(self, target_set, color='cyan', zorder=None, linestyle='-'):
        self.partitioner.plot_reachable_sets(
            target_set,
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
            self.partitioner.dynamics.show_samples(
                None,
                None,
                ax=self.partitioner.animate_axes,
                controller=None,
                input_dims=self.partitioner.input_dims,
                zorder=0,
                xs=np.dstack([xt_samples_from_backreachable_set, xt1_from_those_samples]).transpose(0, 2, 1),
                colors=['g', 'r']
            )

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
        # and run them forward 1 step in time under the NN policy
        xs, _ = self.partitioner.dynamics.collect_data(
            t_max,
            backreachable_set,
            num_samples=1e5,
            controller=self.propagator.network,
            merge_cols=False,
        )

        # Find which of the xt+t_max points actually end up in the target set
        within_constraint_inds = np.where(
            np.all(
                (
                    np.dot(target_set.A, xs[:, -1, :].T)
                    - np.expand_dims(target_set.b[0], axis=-1)
                )
                <= 0,
                axis=0,
            )
        )
        x_samples_inside_backprojection_set = xs[within_constraint_inds]

        if show_samples:
            raise NotImplementedError
            # xt1_from_those_samples_ = xt1_from_those_samples[(within_constraint_inds)]

            # # Show samples from inside the backprojection set and their futures under the NN (should end in target set)
            # self.partitioner.dynamics.show_samples(
            #     None,
            #     None,
            #     ax=self.partitioner.animate_axes,
            #     controller=None,
            #     input_dims=self.partitioner.input_dims,
            #     zorder=1,
            #     xs=np.dstack([xt_samples_inside_backprojection_set, xt1_from_those_samples_]).transpose(0, 2, 1),
            #     colors=None
            # )

            # # Show samples from inside the backreachable set and their futures under the NN (don't necessarily end in target set)
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
