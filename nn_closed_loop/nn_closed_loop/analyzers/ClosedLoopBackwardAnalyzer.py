import nn_partition.analyzers as analyzers
import nn_closed_loop.partitioners as partitioners
import nn_closed_loop.propagators as propagators
from nn_partition.utils.utils import samples_to_range, get_sampled_outputs
import matplotlib.pyplot as plt

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

    def get_one_step_backprojection_set(self, output_constraint, input_constraint, num_partitions=None):
        backprojection_set, info = self.partitioner.get_one_step_backprojection_set(
            output_constraint, input_constraint, self.propagator, num_partitions=num_partitions
        )
        return backprojection_set, info

    def get_backprojection_set(self, output_constraint, input_constraint, t_max, num_partitions=None):
        backprojection_set, info = self.partitioner.get_backprojection_set(
            output_constraint, input_constraint, self.propagator, t_max, num_partitions=num_partitions
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
            input_constraint,
            output_constraint.get_t_max(),
            self.propagator,
            show_samples=show_samples,
            inputs_to_highlight=[
                {"dim": [0], "name": "$x$"},
                {"dim": [1], "name": "$\dot{x}$"},
            ],
            aspect=aspect,
            initial_set_color=self.initial_set_color,
            initial_set_zorder=self.initial_set_zorder,
        )
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
