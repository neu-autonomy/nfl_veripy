import numpy as np
from partition.Analyzer import Analyzer
import torch

# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from partition.network_utils import get_sampled_outputs, samples_to_range
# import os

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'

from closed_loop.ClosedLoopPartitioner import ClosedLoopNoPartitioner
from closed_loop.ClosedLoopPropagator import ClosedLoopCROWNPropagator, ClosedLoopIBPPropagator

# save_dir = "{}/results/analyzer/".format(os.path.dirname(os.path.abspath(__file__)))
# os.makedirs(save_dir, exist_ok=True)

class ClosedLoopAnalyzer(Analyzer):
    def __init__(self, torch_model, At=None, bt=None, ct=None):
        self.torch_model = torch_model
        Analyzer.__init__(self, torch_model=torch_model)

        self.At = At
        self.bt = bt
        self.ct = ct

        # All possible partitioners, propagators
        self.partitioner_dict = {
            "None": ClosedLoopNoPartitioner,
        }
        self.propagator_dict = {
            "CROWN": ClosedLoopCROWNPropagator,
            "IBP": ClosedLoopIBPPropagator,
        }

    def get_output_range(self, input_range):
        output_range, info = self.partitioner.get_output_range(input_range, self.propagator)
        return output_range, info

    def visualize(self, input_range, output_range_estimate, show=True, show_samples=False, **kwargs):
        raise NotImplementedError
        # # sampled_outputs = self.get_sampled_outputs(input_range)
        # # output_range_exact = self.samples_to_range(sampled_outputs)

        # self.partitioner.setup_visualization(input_range, output_range_estimate, self.propagator, show_samples=show_samples)
        # self.partitioner.visualize(kwargs.get("exterior_partitions", kwargs.get("all_partitions", [])), kwargs.get("interior_partitions", []), output_range_estimate)

        # self.partitioner.animate_axes[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #         mode="expand", borderaxespad=0, ncol=1)
        # self.partitioner.animate_axes[1].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #         mode="expand", borderaxespad=0, ncol=1)

        # self.partitioner.animate_fig.tight_layout()

        # if "save_name" in kwargs and kwargs["save_name"] is not None:
        #     plt.savefig(kwargs["save_name"])

        # if show:
        #     plt.show()
        # else:
        #     plt.close()

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
    import numpy as np

    np.random.seed(seed=0)

    ##############
    # Simple FF network
    ###############
    from closed_loop.nn import load_model
    torch_model = load_model()
    
    ##############
    # Dynamics: Double integrator
    ##############
    At = np.array([[1, 1],[0, 1]])
    bt = np.array([[0.5], [1]])
    ct = np.array([0., 0.]).T
    num_states, num_inputs = bt.shape
    # Min/max control inputs
    u_min = -1; u_max = 1

    init_state_range = np.array([ # (num_inputs, 2)
                      [2.5, 3.0], # x0min, x0max
                      [-0.25, 0.25], # x1min, x1max
    ])
    goal_state_range = np.array([
                          [-0.25, 0.25],
                          [-0.25, 0.25]            
    ])

    # Sampling time for simulation
    dt = 1.0

    # LQR-MPC parameters
    # Q = np.eye(2)
    # R = 1
    # Pinf = solve_discrete_are(At, bt, Q, R)
    #
    ##############

    from closed_loop.utils import init_state_range_to_polytope, get_polytope_A
    A_inputs, b_inputs = init_state_range_to_polytope(init_state_range)

    t_max = 4
    A_out = get_polytope_A(9)

    # all_A_out.append(all_A_out[0])
    # all_bs = reachLP_n(t_max, model, A_inputs, b_inputs, At, bt, ct, A_out)
    # all_all_bs.append(all_bs)

    # # SDP (pre-solved)
    # sdp_output_polytope_A = get_polytope_A(9)
    # all_bs = reachSDP_n(t_max, model, A_inputs, b_inputs, At, bt, ct, sdp_output_polytope_A, u_min=u_min, u_max=u_max)
    # # sdp_all_bs_small = all_bs.copy()
    # sdp_all_bs_large = all_bs.copy()
    # # sdp_all_bs_small_unbounded = all_bs.copy()
    # # sdp_all_bs_large_unbounded = all_bs.copy()

    # sdp_all_bs = sdp_all_bs_large
    # # sdp_all_bs = sdp_all_bs_small
    # # sdp_all_bs = sdp_all_bs_small_unbounded
    # # sdp_all_bs = sdp_all_bs_large_unbounded

    # all_A_out.append(sdp_output_polytope_A)
    # all_all_bs.append(sdp_all_bs)

    # run_simulation(At, bt, ct, dt,
    #            t_max, init_state_range, goal_state_range,
    #            u_min, u_max, num_states,
    #            collect_data=False,
    #            show_bounds=True, all_bs=all_all_bs, A_in=all_A_out, bnd_colors=['g','r','c','r'],
    #            model=model,
    #            save_plot=False,
    #         num_samples = 1000, clip_control=True, show_dataset=False)

    partitioner_hyperparams = {
        "type": "None",
        # "make_animation": False,
        # "show_animation": False,
    }
    propagator_hyperparams = {
        "type": "IBP",
        "input_shape": init_state_range.shape[:-1],
    }

    # Run analysis & generate a plot
    analyzer = ClosedLoopAnalyzer(torch_model, At, bt, ct)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    all_bs = analyzer.get_reachable_set(t_max, A_inputs, b_inputs, A_out, u_limits=[u_min, u_max])
    print("all_bs:", all_bs)
    # output_range, analyzer_info = analyzer.get_output_range(input_range)
    # print("Estimated output_range:\n", output_range)
    # print("Number of propagator calls:", analyzer_info["num_propagator_calls"])
    # print("Number of partitions:", analyzer_info["num_partitions"])

    # pars = '_'.join([str(key)+"_"+str(value) for key, value in partitioner_hyperparams.items() if key not in ["make_animation", "show_animation", "type"]])
    # pars2 = '_'.join([str(key)+"_"+str(value) for key, value in propagator_hyperparams.items() if key not in ["input_shape", "type"]])
    # analyzer_info["save_name"] = save_dir+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+".png"

    # analyzer.visualize(input_range, output_range, **analyzer_info)

    print("--- done. ---")
