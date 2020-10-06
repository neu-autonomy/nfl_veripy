import numpy as np
from partition.Analyzer import Analyzer
import torch

import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from partition.network_utils import get_sampled_outputs, samples_to_range
# import os

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'

from closed_loop.ClosedLoopPartitioner import ClosedLoopNoPartitioner, ClosedLoopUniformPartitioner
from closed_loop.ClosedLoopPropagator import ClosedLoopCROWNPropagator, ClosedLoopIBPPropagator, ClosedLoopFastLinPropagator, ClosedLoopSDPPropagator
from closed_loop.ClosedLoopConstraints import PolytopeInputConstraint, LpInputConstraint, PolytopeOutputConstraint, LpOutputConstraint, EllipsoidInputConstraint, EllipsoidOutputConstraint

# save_dir = "{}/results/analyzer/".format(os.path.dirname(os.path.abspath(__file__)))
# os.makedirs(save_dir, exist_ok=True)

class ClosedLoopAnalyzer(Analyzer):
    def __init__(self, torch_model, dynamics):
        self.torch_model = torch_model
        self.dynamics = dynamics
        Analyzer.__init__(self, torch_model=torch_model)

        # All possible partitioners, propagators
        self.partitioner_dict = {
            "None": ClosedLoopNoPartitioner,
            "Uniform": ClosedLoopUniformPartitioner,
        }
        self.propagator_dict = {
            "CROWN": ClosedLoopCROWNPropagator,
            "IBP": ClosedLoopIBPPropagator,
            "FastLin": ClosedLoopFastLinPropagator,
            "SDP": ClosedLoopSDPPropagator,
        }

    def instantiate_partitioner(self, partitioner, hyperparams):
        # dynamics = {"At": self.dynamics.At, "bt": self.dynamics.bt, "ct": self.dynamics.ct}
        # return self.partitioner_dict[partitioner](**{**hyperparams, **dynamics})
        return self.partitioner_dict[partitioner](**{**hyperparams, "dynamics": self.dynamics})

    def instantiate_propagator(self, propagator, hyperparams):
        # dynamics = {"At": self.dynamics.At, "bt": self.dynamics.bt, "ct": self.dynamics.ct}
        return self.propagator_dict[propagator](**{**hyperparams, "dynamics": self.dynamics})

    def get_one_step_reachable_set(self, input_constraint, output_constraint):
        reachable_set, info = self.partitioner.get_one_step_reachable_set(input_constraint, output_constraint, self.propagator)
        return reachable_set, info

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        reachable_set, info = self.partitioner.get_reachable_set(input_constraint, output_constraint, self.propagator, t_max)
        return reachable_set, info

    def visualize(self, input_constraint, output_constraint, show=True, show_samples=False, **kwargs):
        # sampled_outputs = self.get_sampled_outputs(input_range)
        # output_range_exact = self.samples_to_range(sampled_outputs)

        self.partitioner.setup_visualization(input_constraint, output_constraint, self.propagator, show_samples=show_samples)
        self.partitioner.visualize(kwargs.get("exterior_partitions", kwargs.get("all_partitions", [])), kwargs.get("interior_partitions", []), output_constraint)
        
        # self.partitioner.animate_axes.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #         mode="expand", borderaxespad=0, ncol=1)

        self.partitioner.animate_fig.tight_layout()

        # if "save_name" in kwargs and kwargs["save_name"] is not None:
        #     plt.savefig(kwargs["save_name"])

        if show:
            plt.show()
        else:
            plt.close()

    #def get_sampled_outputs(self, input_range, N=1000):
      #  return get_sampled_outputs(input_range, self.propagator, N=N)
    def get_sampled_output_range(self, input_constraint, t_max =5, num_samples =1000):
        return  self.partitioner.get_sampled_out_range(input_constraint, self.propagator, t_max, num_samples)
   


    def get_output_range(self, input_constraint, output_constraint):
        return self.partitioner.get_output_range(input_constraint, output_constraint)

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def get_exact_output_range(self, input_range):
        sampled_outputs = self.get_sampled_outputs(input_range)
        output_range = self.samples_to_range(sampled_outputs)
        return output_range

    def  get_error(self, input_constraint,output_constraint, t_max):
        return self.partitioner.get_error(input_constraint,output_constraint , self.propagator, t_max)

if __name__ == '__main__':
    # Import all deps
    import numpy as np

    np.random.seed(seed=0)

    # system = 'quadrotor'
    system = 'double_integrator_mpc'

    ##############
    # Simple FF network
    ###############
    from closed_loop.nn import load_model
    if system == 'double_integrator_mpc':
        torch_model = load_model(name='double_integrator_mpc')
    elif system == 'quadrotor':
        torch_model = load_model(name='quadrotor_small')
    else:
        raise NotImplementedError
    
    ##############
    # Dynamics
    ##############
    if system == 'double_integrator_mpc':
        from closed_loop.Dynamics import DoubleIntegrator
        dynamics = DoubleIntegrator()
        init_state_range = np.array([ # (num_inputs, 2)
                          [2.5, 3.0], # x0min, x0max
                          [-0.25, 0.25], # x1min, x1max
        ])
        t_max = 1
    elif system == 'quadrotor':
        from closed_loop.Dynamics import Quadrotor
        dynamics = Quadrotor()
        init_state_range = np.array([ # (num_inputs, 2)
                      [4.65,4.65,2.95,0.94,-0.01,-0.01],
                      [4.75,4.75,3.05,0.96,0.01,0.01]
        ]).T
        t_max = 0.1
    else:
        raise NotImplementedError

    # all_output_constraint.append(all_output_constraint[0])
    # all_bs = reachLP_n(t_max, model, input_constraint, At, bt, ct, output_constraint)
    # all_all_bs.append(all_bs)

    # # SDP (pre-solved)
    # sdp_output_polytope_A = get_polytope_A(9)
    # all_bs = reachSDP_n(t_max, model, input_constraint, At, bt, ct, sdp_output_polytope_A, u_min=u_min, u_max=u_max)
    # # sdp_all_bs_small = all_bs.copy()
    # sdp_all_bs_large = all_bs.copy()
    # # sdp_all_bs_small_unbounded = all_bs.copy()
    # # sdp_all_bs_large_unbounded = all_bs.copy()

    # sdp_all_bs = sdp_all_bs_large
    # # sdp_all_bs = sdp_all_bs_small
    # # sdp_all_bs = sdp_all_bs_small_unbounded
    # # sdp_all_bs = sdp_all_bs_large_unbounded

    # all_output_constraint.append(sdp_output_polytope_A)
    # all_all_bs.append(sdp_all_bs)

    partitioner_hyperparams = {
        "type": "None",
        # "type": "Uniform",
       # "num_partitions": np.array([4,4]),
        # "num_partitions": np.array([4,4,1,1,1,1]),
        # "make_animation": False,
        # "show_animation": False,
    }
    propagator_hyperparams = {
        "type": "SDP",
        # "type": "IBP",
        # "type": "CROWN",
      #  "type": "FastLin",
        "input_shape": init_state_range.shape[:-1],
    }

    # Run analysis & generate a plot
   # print(torch_model,dynamics)

    analyzer = ClosedLoopAnalyzer(torch_model, dynamics)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    # ## Polytope Boundaries
    # from closed_loop.utils import init_state_range_to_polytope, get_polytope_A
    # A_inputs, b_inputs = init_state_range_to_polytope(init_state_range)
    # if system == 'quadrotor': A_out = A_inputs
    # else: A_out = get_polytope_A(8)
    # input_constraint = PolytopeInputConstraint(A_inputs, b_inputs)
    # output_constraint = PolytopeOutputConstraint(A_out)

    ### LP-Ball Boundaries
    input_constraint = LpInputConstraint(range=init_state_range, p=np.inf)
    output_constraint = LpOutputConstraint(p=np.inf)

    # ### Ellipsoid Boundaries
    # input_constraint = EllipsoidInputConstraint(
    #     center=np.mean(init_state_range, axis=1),
    #     shape=np.diag((init_state_range[:,1]-init_state_range[:,0])**2)
    # )
    # output_constraint = EllipsoidOutputConstraint()

    output_constraint, analyzer_info = analyzer.get_reachable_set(input_constraint, output_constraint, t_max=t_max)
    # print("output_constraint:", output_constraint)
    # output_range, analyzer_info = analyzer.get_output_range(input_range)
    # print("Estimated output_range:\n", output_range)
    # print("Number of propagator calls:", analyzer_info["num_propagator_calls"])
    # print("Number of partitions:", analyzer_info["num_partitions"])
    
    # pars = '_'.join([str(key)+"_"+str(value) for key, value in partitioner_hyperparams.items() if key not in ["make_animation", "show_animation", "type"]])
    # pars2 = '_'.join([str(key)+"_"+str(value) for key, value in propagator_hyperparams.items() if key not in ["input_shape", "type"]])
    # analyzer_info["save_name"] = save_dir+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+".png"
   # print("output constraint:", output_constraint)
   # print("Analyzer:", analyzer_info)
  #  print('estimated output rang', analyzer.get_output_range(input_constraint, output_constraint))
  #  print('sampled output range', analyzer.get_sampled_output_range(input_constraint,t_max=5, num_samples=1000))
    # error, avg_error = analyzer.get_error(input_constraint,output_constraint)
    # print('Final step approximation error:{:.2f}\nAverage approximation error: {:.2f}'.format(error, avg_error))
    #error, avg_error = analyzer.get_error(input_constraint,output_constraint)
   # print('Final step approximation error:{:.2f}\nAverage approximation error: {:.2f}'.format(error, avg_error))
    analyzer.visualize(input_constraint, output_constraint, show_samples=True, **analyzer_info)
 
    print("--- done. ---")
