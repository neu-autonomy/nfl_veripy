import inspect

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import nfl_veripy.partitioners as partitioners
import nfl_veripy.propagators as propagators
from nfl_veripy.utils.utils import get_sampled_outputs, samples_to_range

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.size"] = "20"


class Analyzer:
    def __init__(self, torch_model):
        self.torch_model = torch_model

        self.partitioner = None
        self.propagator = None

    @property
    def partitioner_dict(self):
        return partitioners.partitioner_dict

    @property
    def propagator_dict(self):
        return propagators.propagator_dict

    @property
    def partitioner(self):
        return self._partitioner

    @partitioner.setter
    def partitioner(self, hyperparams):
        if hyperparams is None:
            return

        hyperparams_ = hyperparams.copy()
        partitioner = hyperparams_.pop("type", None)

        # Make sure we don't send any args to a partitioner that can't handle
        # them. e.g, don't give NoPartitioner a time budget
        args = inspect.getfullargspec(self.partitioner_dict[partitioner]).args
        for hyperparam in hyperparams:
            if hyperparam not in args:
                hyperparams_.pop(hyperparam, None)

        self._partitioner = self.instantiate_partitioner(
            partitioner, hyperparams_
        )

    def instantiate_partitioner(self, partitioner, hyperparams):
        return self.partitioner_dict[partitioner](**hyperparams)

    @property
    def propagator(self):
        return self._propagator

    @propagator.setter
    def propagator(self, hyperparams):
        if hyperparams is None:
            return
        hyperparams_ = hyperparams.copy()
        propagator = hyperparams_.pop("type", None)

        # Make sure we don't send any args to a propagator that can't handle
        # them.
        args = inspect.getfullargspec(self.propagator_dict[propagator]).args
        for hyperparam in hyperparams:
            if hyperparam not in args:
                hyperparams_.pop(hyperparam, None)

        self._propagator = self.instantiate_propagator(
            propagator, hyperparams_
        )
        if propagator is not None:
            self._propagator.network = self.torch_model

    def instantiate_propagator(self, propagator, hyperparams):
        return self.propagator_dict[propagator](**hyperparams)

    def get_output_range(self, input_range, verbose=False):
        output_range, info = self.partitioner.get_output_range(
            input_range, self.propagator
        )
        return output_range, info

    def visualize(
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
        **kwargs
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

    def get_sampled_outputs(self, input_range, N=1000):
        return get_sampled_outputs(input_range, self.propagator, N=N)

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range, N=int(1e4)):
        sampled_outputs = self.get_sampled_outputs(input_range, N=N)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

    def get_exact_hull(self, input_range, N=int(1e4)):
        sampled_outputs = self.get_sampled_outputs(input_range, N=N)
        return ConvexHull(sampled_outputs)

    def get_error(self, input_range, output_range, **analyzer_info):
        if self.partitioner.interior_condition == "convex_hull":
            exact_hull = self.get_exact_hull(input_range)

            error = self.partitioner.get_error(
                exact_hull, analyzer_info["estimated_hull"]
            )
        elif self.partitioner.interior_condition in ["lower_bnds", "linf"]:
            output_range_exact = self.get_exact_output_range(input_range)

            error = self.partitioner.get_error(
                output_range_exact, output_range
            )

        return error
