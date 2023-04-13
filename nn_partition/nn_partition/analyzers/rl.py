import numpy as np
import pickle

import partition

np.set_printoptions(suppress=True)

save_dir = "{}/results/rl/".format(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)


class RLAnalyzer(partition.Analyzer):
    def __init__(self, torch_model):
        partition.Analyzer.__init__(self, torch_model=torch_model)

    def get_output_range(self, input_range):

        # num_partitions = self.partitioner.num_partitions.copy()
        termination_condition_value = self.partitioner.termination_condition_value
        output_range, info = super().get_output_range(input_range)

        # ASSUMPTION: the partitioner is uniform --> also run it with 1 partition so we can plot the diff --> major hack...
        # self.partitioner.num_partitions = 1
        self.partitioner.termination_condition_value = 1
        output_range_default, _ = super().get_output_range(input_range)
        info["output_range_default"] = output_range_default

        self.partitioner.termination_condition_value = termination_condition_value
        # self.partitioner.num_partitions = num_partitions

        # q_values = np.expand_dims(output_range_default[...,0], axis=0)
        # default_action = np.argmax(q_values, axis=1)
        # q_values = np.expand_dims(output_range[...,0], axis=0)
        # new_action = np.argmax(q_values, axis=1)
        # # print("output_range:", output_range)
        # # print("output_range_default:", output_range_default)
        # print("Default Action:", default_action[0], "New Action:", new_action[0])
        # # print('---')

        return output_range, info

    def visualize(self, input_range, output_range_estimate, **kwargs):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        show_input = False
        show_output = True

        self.partitioner.setup_visualization(input_range, output_range_estimate, self.propagator, show_samples=True,
            outputs_to_highlight=kwargs.get("outputs_to_highlight", None),
            inputs_to_highlight=kwargs.get("inputs_to_highlight", None),
            show_input=show_input, show_output=show_output)
        self.partitioner.visualize(kwargs["all_partitions"], [], output_range_estimate, 
            show_input=show_input, show_output=show_output)

        # Show how estimated bounds have improved over baseline
        if "output_range_default" in kwargs:
            color = 'tab:purple'
            linewidth = 2
            if self.partitioner.interior_condition == "linf":
                # Show Estimated Rectangle
                output_range_default_ = kwargs['output_range_default'][self.partitioner.output_dims_]
                rect = Rectangle(output_range_default_[:2,0], output_range_default_[0,1]-output_range_default_[0,0], output_range_default_[1,1]-output_range_default_[1,0],
                                fc='none', linewidth=linewidth,edgecolor=color,
                                label="CROWN Bounds ({})".format(label_dict[self.partitioner.interior_condition]))
                self.partitioner.ax_output.add_patch(rect)
            elif self.partitioner.interior_condition == "lower_bnds":
                
                # Show Lower Bnds if we hadn't used the smarter strategy
                output_range_default_ = kwargs['output_range_default'][self.partitioner.output_dims_]
                output_range_estimate_ = output_range_estimate[self.partitioner.output_dims_]
                line1 = self.partitioner.ax_output.axhline(output_range_default_[1,0], linewidth=linewidth,color=color,
                    label="CROWN ({})".format(label_dict[self.partitioner.interior_condition]))
                line2 = self.partitioner.ax_output.axvline(output_range_default_[0,0], linewidth=linewidth,color=color)
                # self.partitioner.ax_output.lines.append(line1)
                # self.partitioner.ax_output.lines.append(line2)

                # Arrow pointing from bad alg to good alg (axis 0)
                center_x = (output_range_estimate_[0,0] + output_range_default_[0,0])/2.
                center_y = (self.partitioner.ax_output.get_ylim()[1] + output_range_estimate_[1,0])/2.
                self.partitioner.ax_output.annotate("",
                            xy=(output_range_estimate_[0,0], center_y), xycoords='data',
                            xytext=(output_range_default_[0,0], center_y), textcoords='data',
                            arrowprops=dict(arrowstyle="simple", fc="tab:gray", ec="tab:gray"),
                            )

                # Arrow pointing from bad alg to good alg (axis 1)
                center_x = (self.partitioner.ax_output.get_xlim()[1] + output_range_estimate_[0,0])/2.
                center_y = (output_range_estimate_[0,0] + output_range_default_[0,0])/2.
                self.partitioner.ax_output.annotate("",
                            xy=(center_x, output_range_estimate_[1,0]), xycoords='data',
                            xytext=(center_x, output_range_default_[1,0]), textcoords='data',
                            arrowprops=dict(arrowstyle="simple", fc="tab:gray", ec="tab:gray"),
                            )
            else:
                raise NotImplementedError


        if show_input:
            self.partitioner.animate_axes[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=1)
        if show_output:
            output_dims = [(0,),(1,)]
            output_range_default = kwargs['output_range_default'][self.partitioner.output_dims_]
            scale = 0.05
            x_off = max((output_range_default[output_dims[0]+(1,)] - output_range_default[output_dims[0]+(0,)])*(scale), 1e-5)
            y_off = max((output_range_default[output_dims[1]+(1,)] - output_range_default[output_dims[1]+(0,)])*(scale), 1e-5)
            self.partitioner.ax_output.set_xlim(output_range_default[output_dims[0]+(0,)] - x_off, output_range_default[output_dims[0]+(1,)]+x_off)
            self.partitioner.ax_output.set_ylim(output_range_default[output_dims[1]+(0,)] - y_off, output_range_default[output_dims[1]+(1,)]+y_off)

            self.partitioner.ax_output.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=1)

        self.partitioner.animate_fig.tight_layout()

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])
        plt.show()