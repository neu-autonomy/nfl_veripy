import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from partition.network_utils import get_sampled_outputs, samples_to_range

class Analyzer:
    def __init__(self, torch_model):
        self.torch_model = torch_model

        self.partitioner = None
        self.propagator = None

    @property
    def partitioner(self):
        return self._partitioner

    @partitioner.setter
    def partitioner(self, partitioner):
        self._partitioner = partitioner

    @property
    def propagator(self):
        return self._propagator

    @propagator.setter
    def propagator(self, propagator):
        self._propagator = propagator
        if propagator is not None:
            self._propagator.network = self.torch_model

    def get_output_range(self, input_range):
        output_range, info = self.partitioner.get_output_range(input_range, self.propagator)
        return output_range, info

    def visualize(self, input_range, output_range_estimate, **kwargs):
        # def visualize_partitions(sampled_outputs, estimated_output_range, input_range, output_range_sim=None, interior_M=None, M=None):
        fig, axes = plt.subplots(1,2)

        num_input_dimensions_to_plot = 2
        input_shape = input_range.shape[:-1]
        lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
        flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
        flat_dims.sort()
        input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
        input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])

        if "all_partitions" in kwargs:
        # if interior_M is not None and len(interior_M) > 0:
            for (input_range_, output_range_) in kwargs["all_partitions"]:
                rect = Rectangle(output_range_[:2,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
                        fc='none', linewidth=1,edgecolor='m')
                axes[1].add_patch(rect)

                input_range__ = input_range_[input_dims_]
                rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
                        fc='none', linewidth=1,edgecolor='m')
                axes[0].add_patch(rect)
        # if M is not None and len(M) > 0:
        #     for (input_range_, output_range_) in M:
        #         rect = Rectangle(output_range_[:,0], output_range_[0,1]-output_range_[0,0], output_range_[1,1]-output_range_[1,0],
        #                 fc='none', linewidth=1,edgecolor='b')
        #         axes[1].add_patch(rect)

        #         rect = Rectangle(input_range_[:,0], input_range_[0,1]-input_range_[0,0], input_range_[1,1]-input_range_[1,0],
        #                 fc='none', linewidth=1,edgecolor='b')
        #         axes[0].add_patch(rect)

        # Make a rectangle for the estimated boundaries
        rect = Rectangle(output_range_estimate[:2,0], output_range_estimate[0,1]-output_range_estimate[0,0], output_range_estimate[1,1]-output_range_estimate[1,0],
                        fc='none', linewidth=2,edgecolor='g')
        axes[1].add_patch(rect)

        # Make a rectangle for the Exact boundaries
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range_exact = self.samples_to_range(sampled_outputs)
        print("Sampled (exact) output range: \n", output_range_exact)
        axes[1].scatter(sampled_outputs[:,0], sampled_outputs[:,1], c='k', zorder=2)
        rect = Rectangle(output_range_exact[:2,0], output_range_exact[0,1]-output_range_exact[0,0], output_range_exact[1,1]-output_range_exact[1,0],
                        fc='none', linewidth=1,edgecolor='k')
        axes[1].add_patch(rect)

        scale = 0.05
        x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
        y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
        axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
        axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
        axes[0].set_xlabel("Input Dim: {}".format(input_dims[0]))
        axes[0].set_ylabel("Input Dim: {}".format(input_dims[1]))

        if "save_name" in kwargs and kwargs["save_name"] is not None:
            plt.savefig(kwargs["save_name"])
        plt.show()

    def get_sampled_outputs(self, input_range, N=1000):
        return get_sampled_outputs(input_range, self.propagator, N=N)

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

if __name__ == '__main__':
    # Import all deps
    from partition.models import model_xiang_2020_robot_arm, model_simple, model_dynamics
    import numpy as np
    from partition.Partitioner import *
    from partition.Propagator import *

    np.random.seed(seed=0)
    partitioner_dict = {
        "None": NoPartitioner,
        "Uniform": UniformPartitioner,
        "SimGuided": SimGuidedPartitioner,
        "GreedySimGuided": GreedySimGuidedPartitioner,
    }
    propagator_dict = {
        "IBP": IBPPropagator,
        "CROWN": CROWNPropagator,
        "CROWN (LIRPA)": CROWNAutoLIRPAPropagator,
        "IBP (LIRPA)": IBPAutoLIRPAPropagator,
        "CROWN-IBP (LIRPA)": CROWNIBPAutoLIRPAPropagator,
        "SDP": SDPPropagator,
    }

    # Choose experiment settings
    ##############
    # LSTM
    ###############
    # ## A disastrous hack...
    # import sys, os, auto_LiRPA
    # sequence_path = os.path.dirname(os.path.dirname(auto_LiRPA.__file__))+'/examples/sequence'
    # sys.path.append(sequence_path)
    # from lstm import LSTM
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    # parser.add_argument("--norm", type=int, default=2)
    # parser.add_argument("--eps", type=float, default=0.1)
    # parser.add_argument("--num_epochs", type=int, default=20)  
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--num_slices", type=int, default=8)
    # parser.add_argument("--hidden_size", type=int, default=64)
    # parser.add_argument("--num_classes", type=int, default=10) 
    # parser.add_argument("--input_size", type=int, default=784)
    # parser.add_argument("--lr", type=float, default=5e-3)
    # parser.add_argument("--dir", type=str, default=sequence_path+"/model", help="directory to load or save the model")
    # parser.add_argument("--num_epochs_warmup", type=int, default=1, help="number of epochs for the warmup stage when eps is linearly increased from 0 to the full value")
    # parser.add_argument("--log_interval", type=int, default=10, help="interval of printing the log during training")
    # args = parser.parse_args()   
    # torch_model = LSTM(args).to(args.device)
    # input_shape = (8,98)
    # input_range = np.zeros(input_shape+(2,))
    # input_range[-1,0:1,1] = 0.01
    # num_partitions = np.ones(input_shape, dtype=int)
    # partitioner = "SimGuided"
    # partitioner_hyperparams = {"tolerance_eps": 0.001}
    # # partitioner = "Uniform"
    # # num_partitions[-1,0] = 4
    # # partitioner_hyperparams = {"num_partitions": num_partitions}
    # propagator = "IBP (LIRPA)"
    # propagator_hyperparams = {}

    ##############
    # Simple FF network
    ###############
    # torch_model = model_dynamics()
    # input_range = np.array([ # (num_inputs, 2)
    #                   [np.pi/3, 2*np.pi/3], # x0min, x0max
    #                   [np.pi/3, 2*np.pi/3], # x1min, x1max
    #                   [np.pi/3, np.pi/3], # x1min, x1max
    #                   [np.pi/3, np.pi/3], # x1min, x1max
    #                   [0, 0], # amin, amax
    # ])
    torch_model = model_xiang_2020_robot_arm()
    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3], # x1min, x1max
    ])
    # partitioner = "Uniform"
    # partitioner_hyperparams = {"num_partitions": [4,4,1,1,1]}
    # partitioner = "SimGuided"
    partitioner = "GreedySimGuided"
    partitioner_hyperparams = {
        "tolerance_eps": 0.02,
        "interior_condition": "lower_bnds",
        # "interior_condition": "linf",
        # "interior_condition": "convex_hull",
        "make_animation": True,
        "show_animation": True,
    }
    # propagator = "SDP"
    propagator = "IBP (LIRPA)"
    propagator_hyperparams = {"input_shape": input_range.shape[:-1]}

    # Run analysis & generate a plot
    analyzer = Analyzer(torch_model)
    analyzer.partitioner = partitioner_dict[partitioner](**partitioner_hyperparams)
    analyzer.propagator = propagator_dict[propagator](**propagator_hyperparams)
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    print("Estimated output_range:\n", output_range)
    print("Number of propagator calls:", analyzer_info["num_propagator_calls"])
    print("Number of partitions:", analyzer_info["num_partitions"])
    analyzer.visualize(input_range, output_range, **analyzer_info)
    print("done.")

