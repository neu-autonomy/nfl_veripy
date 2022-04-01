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

        self.reachable_set_color = 'tab:blue'
        self.reachable_set_zorder = 2
        self.initial_set_color = 'tab:red'
        self.initial_set_zorder = 2
        self.sample_zorder = 1

    @property
    def partitioner_dict(self):
        return partitioners.partitioner_dict

    @property
    def propagator_dict(self):
        return propagators.propagator_dict

    def instantiate_partitioner(self, partitioner, hyperparams):
        return self.partitioner_dict[partitioner](
            **{**hyperparams, "dynamics": self.dynamics}
        )

    def instantiate_propagator(self, propagator, hyperparams):
        return self.propagator_dict[propagator](
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
        show_trajectories=False,
        aspect="auto",
        labels={},
        inputs_to_highlight=None,
        dont_close=True,
        **kwargs
    ):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        if inputs_to_highlight is None:
            inputs_to_highlight = [
                {"dim": [0], "name": "$x_0$"},
                {"dim": [1], "name": "$x_1$"},
            ]

        self.partitioner.setup_visualization(
            input_constraint,
            output_constraint.get_t_max(),
            self.propagator,
            show_samples=show_samples,
            show_trajectories=show_trajectories,
            inputs_to_highlight=inputs_to_highlight,
            aspect=aspect,
            initial_set_color=self.initial_set_color,
            initial_set_zorder=self.initial_set_zorder,
            sample_zorder=self.sample_zorder
        )
        self.partitioner.visualize(
            kwargs.get(
                "exterior_partitions", kwargs.get("all_partitions", [])
            ),
            kwargs.get("interior_partitions", []),
            output_constraint,
            kwargs.get("iteration", None),
            reachable_set_color=self.reachable_set_color,
            reachable_set_zorder=self.reachable_set_zorder
        )

        # self.partitioner.animate_axes.legend(
        #     bbox_to_anchor=(0, 1.02, 1, 0.2),
        #     loc="lower left",
        #     mode="expand",
        #     borderaxespad=0,
        #     ncol=1,
        # )

        import numpy as np
        import nn_closed_loop.constraints as constraints
        x0 = np.array(
            [  # (num_inputs, 2)
                [-5.5, -4.5],  # x0min, x0max
                [1-0.5, 1+0.5],  # x1min, x1max
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
        #         [-0.25, 0.25],  # x0min, x0max
        #         [-0.25, 0.25],  # x1min, x1max
        #         [0.99,1.01],
        #         [1.24,1.26]
        #     ]
        # )
        x0_constraint = constraints.LpConstraint(
            range=x0, p=np.inf
        )
        input_dims = [x["dim"] for x in inputs_to_highlight]
        self.dynamics.show_trajectories(
            output_constraint.get_t_max() * self.dynamics.dt,
            x0_constraint,
            input_dims=input_dims,
            ax=self.partitioner.animate_axes,
            controller=self.propagator.network,
        ) 

        initial_constraint = constraints.LpConstraint(x0)
        self.partitioner.plot_reachable_sets(
            initial_constraint,
            input_dims,
            reachable_set_color='k',
            reachable_set_zorder=10,
            reachable_set_ls='-'
        )
        xf = np.array(
            [  # (num_inputs, 2)
                [-1, 1],  # x0min, x0max
                [-1, 1],  # x1min, x1max
            ]
        )
        xf_constraint = constraints.LpConstraint(
            range=xf, p=np.inf
        )
        self.partitioner.plot_reachable_sets(
            xf_constraint,
            input_dims,
            reachable_set_color='tab:red',
            reachable_set_zorder=10,
            reachable_set_ls='-'
        )

        self.partitioner.animate_fig.tight_layout()

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])

        if show:
            plt.show()
        elif not dont_close:
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
