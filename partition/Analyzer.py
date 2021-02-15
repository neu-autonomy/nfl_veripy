import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from partition.network_utils import get_sampled_outputs, samples_to_range
import os
import time
import argparse

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

    def visualize(self, input_range, output_range_estimate, show=True, show_samples=True, show_legend=True, show_input=True, show_output=True, title=None, labels={}, **kwargs):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(input_range, output_range_estimate, self.propagator, show_samples=show_samples, inputs_to_highlight=kwargs.get('inputs_to_highlight', None), outputs_to_highlight=kwargs.get('outputs_to_highlight', None),
            show_input=show_input, show_output=show_output, labels=labels)
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

    def get_exact_hull(self, input_range, N=int(1e7)):
        from scipy.spatial import ConvexHull
        sampled_outputs = self.get_sampled_outputs(input_range, N=N)
        return ConvexHull(sampled_outputs)

if __name__ == '__main__':
    # Import all deps
    from partition.models import model_xiang_2020_robot_arm, model_simple, model_dynamics, random_model
    import numpy as np

    np.random.seed(seed=0)

    parser = argparse.ArgumentParser(description='Analyze a NN.')
    parser.add_argument('--model', default='robot_arm',
                        help='which NN to analyze (default: robot_arm)')
    parser.add_argument('--activation', default='tanh',
                        help='nonlinear activation fn in NN (default: tanh)')
    parser.add_argument('--partitioner', default='GreedySimGuided',
                        help='which partitioner to use (default: GreedySimGuided)')
    parser.add_argument('--propagator', default='CROWN_LIRPA',
                        help='which propagator to use (default: CROWN_LIRPA)')
    parser.add_argument('--term_type', default='time_budget',
                        help='type of condition to terminate (default: time_budget)')
    parser.add_argument('--term_val', default=2., type=float,
                        help='value of condition to terminate (default: 2)')
    parser.add_argument('--interior_condition', default='lower_bnds',
                        help='type of bound to optimize for (default: lower_bnds)')
    parser.add_argument('--num_simulations', default=1e4,
                        help='how many MC samples to begin with (default: 1e4)')
    parser.add_argument('--save_plot', default=True, type=bool,
                        help='whether to save the visualization (default: True)')
    parser.add_argument('--show_plot', default=False, type=bool,
                        help='whether to show the visualization (default: False)')
    parser.add_argument('--show_input', default=False, type=bool,
                        help='whether to show the input partition in the plot (default: False)')
    parser.add_argument('--input_plot_labels', default=["Input", None], type=list,
                        help='x and y labels on input partition plot (default: ["Input", None])')
    parser.add_argument('--output_plot_labels', default=["Output", None], type=list,
                        help='x and y labels on output partition plot (default: ["Output", None])')
    parser.add_argument('--input_plot_aspect', default="auto",
                        help='aspect ratio on input partition plot (default: auto)')
    parser.add_argument('--output_plot_aspect', default="auto",
                        help='aspect ratio on output partition plot (default: auto)')

    args = parser.parse_args()

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
    if args.model == 'robot_arm':
        torch_model, model_info = model_xiang_2020_robot_arm(activation=args.activation)
        input_range = np.array([ # (num_inputs, 2)
                          [np.pi/3, 2*np.pi/3], # x0min, x0max
                          [np.pi/3, 2*np.pi/3], # x1min, x1max
        ])
    elif args.model == 'random_weights':
         neurons = [2,50,2]
         torch_model, model_info = random_model(activation=args.activation, neurons=neurons, seed=0)
         input_range = np.zeros((model_info['model_neurons'][0],2))
         input_range[:,1] = 1.
    else:
        raise NotImplementedError

    partitioner_hyperparams = {
        "num_simulations": args.num_simulations,
        "type": args.partitioner,

        # "termination_condition_type": "verify",
        # "termination_condition_value": [np.array([1., 0.]), np.array([100.])],

       # "termination_condition_type": "input_cell_size",
       # "termination_condition_value": 100,
      # "termination_condition_type": "num_propagator_calls",
      # "termination_condition_value": 0.05,
      #   "termination_condition_type": "pct_improvement",
       #  "termination_condition_value": 0.001,
       
       "termination_condition_type": args.term_type,
        "termination_condition_value": args.term_val,

       #  "termination_condition_type": "pct_error",
        # "num_partitions": 1,

        "interior_condition": args.interior_condition,
        "make_animation": False,
        "show_animation": False,
        # "show_output": False,
    }
    propagator_hyperparams = {
        "type": args.propagator,
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
    print("\n")
    print("{}+{}".format(partitioner_hyperparams["type"], propagator_hyperparams["type"]) )
   # print("Estimated output_range:\n", output_range)
    # print("True output_range:\n", output_range_exact)
    print("Number of propagator calls:", analyzer_info["num_propagator_calls"])
    print("Error: ", error)
    print("Number of partitions:", analyzer_info["num_partitions"])
    print("Computation time:",analyzer_info["computation_time"] )
    print("Number of iteration :",analyzer_info["num_iteration"] )
    print("Error (inloop) :",analyzer_info["estimation_error"] )
  #  print(output_range , analyzer_info["estimated_hull"] )

    if args.save_plot:
        # Ugly logic to embed parameters in filename:
        pars = '_'.join([str(key)+"_"+str(value) for key, value in sorted(partitioner_hyperparams.items(), key=lambda kv: kv[0]) if key not in ["make_animation", "show_animation", "type"]])
        pars2 = '_'.join([str(key)+"_"+str(value) for key, value in sorted(propagator_hyperparams.items(), key=lambda kv: kv[0]) if key not in ["input_shape", "type"]])
        model_str = args.model + '_' + args.activation + '_'
        analyzer_info["save_name"] = save_dir+model_str+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars
        if len(pars2) > 0:
            analyzer_info["save_name"] = analyzer_info["save_name"] + "_" + pars2
        analyzer_info["save_name"] = analyzer_info["save_name"] + ".png"
        
        # Plot settings
        labels = {"input": args.input_plot_labels, "output": args.output_plot_labels}
        aspects = {"input": args.input_plot_aspect, "output": args.output_plot_aspect}
        
        # Generate the plot & save
        analyzer.visualize(input_range, output_range, show=args.show_plot, show_samples=True, show_legend=False, show_input=args.show_input, show_output=True, title=None, labels=labels, aspects=aspects, **analyzer_info)
    
    print("done.")
