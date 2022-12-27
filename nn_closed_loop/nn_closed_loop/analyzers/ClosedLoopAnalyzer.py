from matplotlib.pyplot import plot
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
        self.initial_set_color = 'k'
        self.initial_set_zorder = 2
        self.target_set_color = 'tab:red'
        self.target_set_zorder = 2
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

    def get_one_step_reachable_set(self, initial_set):
        # initial_set: constraints.LpConstraint(range=(num_states, 2))
        # reachable_set: constraints.LpConstraint(range=(num_states, 2))
        reachable_set, info = self.partitioner.get_one_step_reachable_set(
            initial_set, self.propagator
        )
        return reachable_set, info

    def get_reachable_set(self, initial_set, t_max):
        # initial_set: constraints.LpConstraint(range=(num_states, 2))
        # reachable_set: constraints.LpConstraint(range=(num_timesteps, num_states, 2))
        reachable_set, info = self.partitioner.get_reachable_set(
            initial_set, self.propagator, t_max
        )
        return reachable_set, info

    def visualize(
        self,
        initial_set,
        reachable_sets,
        target_constraint=None,
        show=True,
        show_samples=False,
        show_trajectories=False,
        aspect="auto",
        plot_lims=None,
        labels={},
        inputs_to_highlight=None,
        dont_close=True,
        controller_name=None,
        **kwargs
    ):

        if inputs_to_highlight is None:
            inputs_to_highlight = [
                {"dim": [0], "name": "$x_0$"},
                {"dim": [1], "name": "$x_1$"},
            ]
        self.partitioner.setup_visualization(
            initial_set,
            reachable_sets.get_t_max(),
            self.propagator,
            show_samples=show_samples,
            show_trajectories=show_trajectories,
            inputs_to_highlight=inputs_to_highlight,
            aspect=aspect,
            initial_set_color=self.initial_set_color,
            initial_set_zorder=self.initial_set_zorder,
            extra_set_color=self.target_set_color,
            extra_set_zorder=self.target_set_zorder,
            sample_zorder=self.sample_zorder,
            extra_constraint=target_constraint,
            plot_lims=plot_lims,
            controller_name=controller_name
        )
        self.partitioner.visualize(
            kwargs.get(
                "exterior_partitions", kwargs.get("all_partitions", [])
            ),
            kwargs.get("interior_partitions", []),
            reachable_sets,
            kwargs.get("iteration", None),
            reachable_set_color=self.reachable_set_color,
            reachable_set_zorder=self.reachable_set_zorder,
        )

        import numpy as np
        import nn_closed_loop.constraints as constraints
        x0 = np.array(
            [  # (num_inputs, 2)
                [-5.5, -4.5],  # x0min, x0max
                [-0.5, 0.5],  # x1min, x1max
            ]
        )
        x0_constraint = constraints.LpConstraint(
            range=x0, p=np.inf
        )
        input_dims = [x["dim"] for x in inputs_to_highlight]
        if show_trajectories:
            self.dynamics.show_trajectories(
                reachable_sets.get_t_max() * self.dynamics.dt,
                initial_set,
                input_dims=input_dims,
                ax=self.partitioner.animate_axes,
                controller=self.propagator.network,
            )

        self.partitioner.animate_fig.tight_layout()

        if plot_lims is not None:
            import ast
            plot_lims_arr = np.array(
                ast.literal_eval(plot_lims)
            )
            self.partitioner.animate_axes.set_xlim(plot_lims_arr[0])
            self.partitioner.animate_axes.set_ylim(plot_lims_arr[1])


        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])

        if show:
            plt.show()
        elif not dont_close:
            plt.close()

    def get_sampled_outputs(self, input_range, N=1000):
        return get_sampled_outputs(input_range, self.propagator, N=N)

    def get_sampled_output_range(
        self, initial_set, t_max=5, num_samples=1000
    ):
        return self.partitioner.get_sampled_out_range(
            initial_set, self.propagator, t_max, num_samples
        )

    def get_output_range(self, initial_set, reachable_sets):
        return self.partitioner.get_output_range(
            initial_set, reachable_sets
        )

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = samples_to_range(sampled_outputs)
        return output_range

    def get_error(self, initial_set, reachable_sets, t_max):
        return self.partitioner.get_error(
            initial_set, reachable_sets, self.propagator, t_max
        )
