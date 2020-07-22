import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from partition.network_utils import get_sampled_outputs, samples_to_range

from partition.Partitioner import NoPartitioner, UniformPartitioner, SimGuidedPartitioner, GreedySimGuidedPartitioner, AdaptiveSimGuidedPartitioner
from partition.Propagator import IBPPropagator, CROWNPropagator, CROWNAutoLIRPAPropagator, IBPAutoLIRPAPropagator, CROWNIBPAutoLIRPAPropagator, SDPPropagator
partitioner_dict = {
    "None": NoPartitioner,
    "Uniform": UniformPartitioner,
    "SimGuided": SimGuidedPartitioner,
    "GreedySimGuided": GreedySimGuidedPartitioner,
    "AdaptiveSimGuided": AdaptiveSimGuidedPartitioner,
}
propagator_dict = {
    "IBP": IBPPropagator,
    "CROWN": CROWNPropagator,
    "CROWN (LIRPA)": CROWNAutoLIRPAPropagator,
    "IBP (LIRPA)": IBPAutoLIRPAPropagator,
    "CROWN-IBP (LIRPA)": CROWNIBPAutoLIRPAPropagator,
    "SDP": SDPPropagator,
}

class Analyzer:
    def __init__(self, torch_model):
        self.torch_model = torch_model

        self.partitioner = None
        self.propagator = None

    @property
    def partitioner(self):
        return self._partitioner

    @partitioner.setter
    def partitioner(self, hyperparams):
        if hyperparams is None: return
        hyperparams_ = hyperparams.copy()
        partitioner = hyperparams_.pop('type', None)
        self._partitioner = partitioner_dict[partitioner](**hyperparams_)

    @property
    def propagator(self):
        return self._propagator

    @propagator.setter
    def propagator(self, hyperparams):
        if hyperparams is None: return
        hyperparams_ = hyperparams.copy()
        propagator = hyperparams_.pop('type', None)
        self._propagator = propagator_dict[propagator](**hyperparams_)
        if propagator is not None:
            self._propagator.network = self.torch_model

    def get_output_range(self, input_range):
        output_range, info = self.partitioner.get_output_range(input_range, self.propagator)
        return output_range, info

    def visualize(self, input_range, output_range_estimate, **kwargs):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(input_range, output_range_estimate, output_range_exact, self.propagator)
        self.partitioner.visualize(kwargs["exterior_partitions"], kwargs["interior_partitions"], output_range_estimate)

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

    np.random.seed(seed=0)

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
    torch_model = model_xiang_2020_robot_arm()
    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3], # x1min, x1max
    ])
    # partitioner = "Uniform"
    # partitioner_hyperparams = {"num_partitions": [4,4,1,1,1]}
    partitioner_hyperparams = {
        "type": "GreedySimGuided",
        "tolerance_eps": 0.02,
        # "interior_condition": "lower_bnds",
        "interior_condition": "linf",
        # "interior_condition": "convex_hull",
        "make_animation": False,
        "show_animation": False,
    }
    propagator_hyperparams = {
        "type": "IBP (LIRPA)",
        "input_shape": input_range.shape[:-1],
    }

    # Run analysis & generate a plot
    analyzer = Analyzer(torch_model)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    print("Estimated output_range:\n", output_range)
    print("Number of propagator calls:", analyzer_info["num_propagator_calls"])
    print("Number of partitions:", analyzer_info["num_partitions"])
    analyzer.visualize(input_range, output_range, **analyzer_info)
    print("done.")
