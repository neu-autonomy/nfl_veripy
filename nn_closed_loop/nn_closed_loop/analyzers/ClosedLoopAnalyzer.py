import nn_partition.analyzers as analyzers
import nn_closed_loop.partitioners as partitioners
import nn_closed_loop.propagators as propagators
from nn_partition.utils.utils import samples_to_range, get_sampled_outputs
import matplotlib.pyplot as plt

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'


class ClosedLoopAnalyzer(analyzers.Analyzer):
    def __init__(self, torch_model, dynamics):
        self.torch_model = torch_model
        self.dynamics = dynamics
        analyzers.Analyzer.__init__(self, torch_model=torch_model)

    def instantiate_partitioner(self, partitioner, hyperparams):
        return partitioners.partitioner_dict[partitioner](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def instantiate_propagator(self, propagator, hyperparams):
        return propagators.propagator_dict[propagator](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        reachable_set, info = self.partitioner.get_one_step_reachable_set(
            input_constraint, output_constraint, self.propagator
        )
        return reachable_set, info

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        reachable_set, info = self.partitioner.get_reachable_set(
            input_constraint, output_constraint, self.propagator, t_max
        )
        return reachable_set, info

    def visualize(
        self,
        input_constraint,
        output_constraint,
        show=True,
        show_samples=False,
        aspect="auto",
        labels={},
        **kwargs
    ):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(
            input_constraint,
            output_constraint,
            self.propagator,
            show_samples=show_samples,
            outputs_to_highlight=[
                {"dim": [0], "name": "py"},
                {"dim": [1], "name": "pz"},
            ],
            inputs_to_highlight=[
                {"dim": [0], "name": "py"},
                {"dim": [1], "name": "pz"},
            ],
            aspect=aspect,
        )
        self.partitioner.visualize(
            kwargs.get(
                "exterior_partitions", kwargs.get("all_partitions", [])
            ),
            kwargs.get("interior_partitions", []),
            output_constraint,
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
