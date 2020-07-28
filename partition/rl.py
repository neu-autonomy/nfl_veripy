import numpy as np
import pickle

from partition.Partitioner import *
from partition.Propagator import *
from partition.Analyzer import *

np.set_printoptions(suppress=True)

save_dir = "{}/results/rl/".format(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)


class RLAnalyzer(Analyzer):
    def __init__(self, torch_model):
        Analyzer.__init__(self, torch_model=torch_model)

    def get_output_range(self, input_range):

        # num_partitions = self.partitioner.num_partitions.copy()
        # num_partitions = self.partitioner.num_partitions
        output_range, info = super().get_output_range(input_range)

        # # ASSUMPTION: the partitioner is uniform --> also run it with 1 partition so we can plot the diff --> major hack...
        # self.partitioner.num_partitions = 1
        # output_range_default, _ = super().get_output_range(input_range)
        # info["output_range_default"] = output_range_default

        # # self.partitioner.num_partitions = num_partitions.copy()
        # self.partitioner.num_partitions = num_partitions
        return output_range, info

    def visualize(self, input_range, output_range_estimate, **kwargs):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(input_range, output_range_estimate, self.propagator, show_samples=True,
            outputs_to_highlight=kwargs.get("outputs_to_highlight", None),
            inputs_to_highlight=kwargs.get("inputs_to_highlight", None))
        self.partitioner.visualize(kwargs["all_partitions"], [], output_range_estimate)

        if "output_range_default" in kwargs:
            color = 'tab:red'
            linewidth = 2
            if self.partitioner.interior_condition == "linf":
                output_range_default_ = kwargs['output_range_default'][self.partitioner.output_dims_]
                rect = Rectangle(output_range_default_[:2,0], output_range_default_[0,1]-output_range_default_[0,0], output_range_default_[1,1]-output_range_default_[1,0],
                                fc='none', linewidth=linewidth,edgecolor=color,
                                label="CROWN Bounds ({})".format(label_dict[self.partitioner.interior_condition]))
                self.partitioner.animate_axes[1].add_patch(rect)
                self.partitioner.default_patches[1].append(rect)
            elif self.partitioner.interior_condition == "lower_bnds":
                output_range_default_ = kwargs['output_range_default'][self.partitioner.output_dims_]
                output_range_estimate_ = output_range_estimate[self.partitioner.output_dims_]
                line1 = self.partitioner.animate_axes[1].axhline(output_range_default_[1,0], linewidth=linewidth,color=color,
                    label="CROWN Bounds ({})".format(label_dict[self.partitioner.interior_condition]))
                line2 = self.partitioner.animate_axes[1].axvline(output_range_default_[0,0], linewidth=linewidth,color=color)
                self.partitioner.default_lines[1].append(line1)
                self.partitioner.default_lines[1].append(line2)

                center_x = (output_range_estimate_[0,0] + output_range_default_[0,0])/2.
                center_y = (self.partitioner.animate_axes[1].get_ylim()[1] + output_range_estimate_[1,0])/2.
                self.partitioner.animate_axes[1].annotate("",
                            xy=(output_range_estimate_[0,0], center_y), xycoords='data',
                            xytext=(output_range_default_[0,0], center_y), textcoords='data',
                            arrowprops=dict(arrowstyle="simple", fc="tab:gray", ec="tab:gray"),
                            )

                center_x = (self.partitioner.animate_axes[1].get_xlim()[1] + output_range_estimate_[0,0])/2.
                center_y = (output_range_estimate_[0,0] + output_range_default_[0,0])/2.
                self.partitioner.animate_axes[1].annotate("",
                            xy=(center_x, output_range_estimate_[1,0]), xycoords='data',
                            xytext=(center_x, output_range_default_[1,0]), textcoords='data',
                            arrowprops=dict(arrowstyle="simple", fc="tab:gray", ec="tab:gray"),
                            )
            else:
                raise NotImplementedError


        self.partitioner.animate_axes[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)
        self.partitioner.animate_axes[1].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1)

        self.partitioner.animate_fig.tight_layout()

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])
        plt.show()