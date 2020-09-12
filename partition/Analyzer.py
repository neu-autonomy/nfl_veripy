import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from partition.network_utils import get_sampled_outputs, samples_to_range
import os
import time

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = '20'

from partition.Partitioner import NoPartitioner, UniformPartitioner, SimGuidedPartitioner, GreedySimGuidedPartitioner, AdaptiveSimGuidedPartitioner, UnGuidedPartitioner
from partition.Propagator import IBPPropagator, CROWNPropagator, CROWNAutoLIRPAPropagator, IBPAutoLIRPAPropagator, CROWNIBPAutoLIRPAPropagator, SDPPropagator, FastLinAutoLIRPAPropagator, ExhaustiveAutoLIRPAPropagator
partitioner_dict = {
    "None": NoPartitioner,
    "Uniform": UniformPartitioner,
    "SimGuided": SimGuidedPartitioner,
    "GreedySimGuided": GreedySimGuidedPartitioner,
    "AdaptiveSimGuided": AdaptiveSimGuidedPartitioner,
    "UnGuided": UnGuidedPartitioner,
}
propagator_dict = {
    "IBP": IBPPropagator,
    "CROWN": CROWNPropagator,
    "CROWN_LIRPA": CROWNAutoLIRPAPropagator,
    "IBP_LIRPA": IBPAutoLIRPAPropagator,
    "CROWN-IBP_LIRPA": CROWNIBPAutoLIRPAPropagator,
    "FastLin_LIRPA": FastLinAutoLIRPAPropagator,
    "Exhaustive_LIRPA": ExhaustiveAutoLIRPAPropagator,
    "SDP": SDPPropagator,
}

save_dir = "{}/results/analyzer/".format(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)

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

    def get_output_range(self, input_range, verbose=False):
        output_range, info = self.partitioner.get_output_range(input_range, self.propagator)
        return output_range, info

    def visualize(self, input_range, output_range_estimate, show=True, show_samples=True, show_legend=True, show_input=True, show_output=True, title=None, **kwargs):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(input_range, output_range_estimate, self.propagator, show_samples=show_samples, inputs_to_highlight=kwargs.get('inputs_to_highlight', None), outputs_to_highlight=kwargs.get('outputs_to_highlight', None),
            show_input=show_input, show_output=show_output)
        self.partitioner.visualize(kwargs.get("exterior_partitions", kwargs.get("all_partitions", [])), kwargs.get("interior_partitions", []), output_range_estimate,
            show_input=show_input, show_output=show_output)

        if show_legend:
            if show_input:
                self.partitioner.input_axis.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=1)
            if show_output:
                self.partitioner.output_axis.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=2)

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

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

    def get_exact_hull(self, input_range, N=int(1e5)):
        from scipy.spatial import ConvexHull
        sampled_outputs = self.get_sampled_outputs(input_range, N=N)
        return ConvexHull(sampled_outputs)

if __name__ == '__main__':
    # Import all deps
    from partition.models import model_xiang_2020_robot_arm, model_simple, model_dynamics, random_model
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
    torch_model, model_info = model_xiang_2020_robot_arm()
    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3], # x1min, x1max
    ])

   # neurons = [2,50,2]
  #  torch_model, model_info = random_model(activation='relu', neurons=neurons, seed=0)
  #  input_range = np.zeros((model_info['model_neurons'][0],2))
   # input_range[:,1] = 1.

    # partitioner = "Uniform"
    # partitioner_hyperparams = {"num_partitions": [4,4,1,1,1]}
    partitioner_hyperparams = {
        "num_simulations": int(10000),
        # "type": "Uniform",
         "type": "SimGuided",
         #"type": "GreedySimGuided",
        #"type": "AdaptiveSimGuided",
        # "type": "UnGuided",

        # "termination_condition_type": "verify",
        # "termination_condition_value": [np.array([1., 0.]), np.array([100.])],

       # "termination_condition_type": "input_cell_size",
       # "termination_condition_value": 100,
      # "termination_condition_type": "num_propagator_calls",
      # "termination_condition_value": 0.05,
      #   "termination_condition_type": "pct_improvement",
       #  "termination_condition_value": 0.001,
       
       "termination_condition_type": "time_budget",

       #  "termination_condition_type": "pct_error",
         "termination_condition_value": 2,
        # "num_partitions": 1,

       # "interior_condition": "lower_bnds",
        #"interior_condition": "linf",
        "interior_condition": "convex_hull",
        # "interior_condition": "linf",
        # "interior_condition": "convex_hull",
        "make_animation": False,
        "show_animation": False,
        # "show_output": False,
    }
    propagator_hyperparams = {
       "type": "IBP_LIRPA",
      "type": "CROWN_LIRPA",
        "input_shape": input_range.shape[:-1],
    }

    # Run analysis & generate a plot
    analyzer = Analyzer(torch_model)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    t_start = time.time()
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    t_end = time.time()
    computation_time = t_end - t_start
  #  print(analyzer_info)
    np.random.seed(seed=0)
   # output_range_exact = analyzer.get_exact_output_range(input_range)
    #if analyzer.partitioner["interior_condition"] == "convex_hull":
   #else:
    if  partitioner_hyperparams["interior_condition"] == "convex_hull":
        exact_hull = analyzer.get_exact_hull(input_range)

        error = analyzer.partitioner.get_error(exact_hull, analyzer_info["estimated_hull"])
    if  partitioner_hyperparams["interior_condition"] in ["lower_bnds", "linf"]:
        output_range_exact = analyzer.get_exact_output_range(input_range)

        error = analyzer.partitioner.get_error(output_range_exact, output_range)


   # output_range_exact = analyzer.get_exact_output_range(input_range)

   # error = analyzer.partitioner.get_error(output_range_exact, output_range)
    print("Estimated output_range:\n", output_range)
    # print("True output_range:\n", output_range_exact)
    print("\n")
    print("Number of propagator calls:", analyzer_info["num_propagator_calls"])
    print("Error: ", error)
    print("Number of partitions:", analyzer_info["num_partitions"])
    print("Computation time:",analyzer_info["computation_time"] )
    print("Number of iteration :",analyzer_info["num_iteration"] )

    pars = '_'.join([str(key)+"_"+str(value) for key, value in sorted(partitioner_hyperparams.items(), key=lambda kv: kv[0]) if key not in ["make_animation", "show_animation", "type"]])
    pars2 = '_'.join([str(key)+"_"+str(value) for key, value in sorted(propagator_hyperparams.items(), key=lambda kv: kv[0]) if key not in ["input_shape", "type"]])
    analyzer_info["save_name"] = save_dir+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+".pdf"

    title = "# Partitions: {}, Error: {}".format(str(analyzer_info['num_partitions']), str(round(error, 3)))
    analyzer.visualize(input_range, output_range, show_legend=False, show_input=True, show_output=False, title=title, **analyzer_info)
    # title = "# Partitions: {}, Error: {}".format(str(analyzer_info["num_partitions"]), str(round(error, 3)))
    # analyzer.visualize(input_range, output_range, show_legend=False, show_input=True, show_output=False, title=title, **analyzer_info)

    print("done.")
