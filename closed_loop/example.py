import numpy as np
import closed_loop.dynamics as dynamics
import closed_loop.analyzers as analyzers
import closed_loop.constraints as constraints
from closed_loop.utils.nn import load_model

if __name__ == '__main__':
    # Import all deps

    np.random.seed(seed=0)

    # system = 'quadrotor'
    system = 'double_integrator_mpc'

    ##############
    # Simple FF network
    ###############
    if system == 'double_integrator_mpc':
        torch_model = load_model(name='double_integrator_mpc')
    elif system == 'quadrotor':
        torch_model = load_model(name='quadrotor')
    else:
        raise NotImplementedError
    
    ##############
    # Dynamics
    ##############
    if system == 'double_integrator_mpc':
        dynamics = dynamics.DoubleIntegrator()
        # dynamics = DoubleIntegratorOutputFeedback()
        init_state_range = np.array([ # (num_inputs, 2)
                          [2.5, 3.0], # x0min, x0max
                          [-0.25, 0.25], # x1min, x1max
        ])
        t_max = 5
    elif system == 'quadrotor':
        # dynamics = Quadrotor()
        dynamics = QuadrotorOutputFeedback()
        init_state_range = np.array([ # (num_inputs, 2)
                      [4.65,4.65,2.95,0.94,-0.01,-0.01],
                      [4.75,4.75,3.05,0.96,0.01,0.01]
        ]).T
        t_max = 1.5
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
        "type": "Uniform",
        "num_partitions": np.array([4,4]),
       # "num_partitions": np.array([4,4,1,1,1,1]),
        # "make_animation": False,
        # "show_animation": False,
       # "type": "ProbPartition",
      #  "num_partitions": np.array([10])
    }
    propagator_hyperparams = {
        # "type": "SDP",
        # "type": "IBP",
        "type": "CROWN",
      #  "type": "FastLin",
        "input_shape": init_state_range.shape[:-1],
    }

    # Run analysis & generate a plot
   # print(torch_model,dynamics)

    analyzer = analyzers.ClosedLoopAnalyzer(torch_model, dynamics)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    print(analyzer)
    print(analyzer.partitioner)

    # # ## Polytope Boundaries
    # from closed_loop.utils import init_state_range_to_polytope, get_polytope_A
    # A_inputs, b_inputs = init_state_range_to_polytope(init_state_range)
    # if system == 'quadrotor': A_out = A_inputs
    # else: A_out = get_polytope_A(8)
    # input_constraint = PolytopeInputConstraint(A_inputs, b_inputs)
    # output_constraint = PolytopeOutputConstraint(A_out)

    ## LP-Ball Boundaries
    input_constraint = constraints.LpInputConstraint(range=init_state_range, p=np.inf)
    output_constraint = constraints.LpOutputConstraint(p=np.inf)

    # ### Ellipsoid Boundaries
    # input_constraint = EllipsoidInputConstraint(
    #     center=np.mean(init_state_range, axis=1),
    #     shape=np.diag((init_state_range[:,1]-init_state_range[:,0])**2)
    # )
    # output_constraint = EllipsoidOutputConstraint()

    # # Estimate time of running the calculations
    # import time
    # num_calls = 5
    # times = np.empty(num_calls)
    # for num in range(num_calls):
    #     t_start = time.time()
    #     output_constraint, analyzer_info = analyzer.get_reachable_set(input_constraint, output_constraint, t_max=t_max)
    #     t_end = time.time()
    #     t = t_end - t_start
    #     times[num] = t
    # print("All times: {}".format(times))
    # print("Avg time: {}".format(times.mean()))
    
    output_constraint, analyzer_info, prob_list = analyzer.get_reachable_set(input_constraint, output_constraint, t_max=0.8)
    # print("output_constraint:", output_constraint.range)
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
    # error, avg_error = analyzer.get_error(input_constraint,output_constraint, t_max=t_max)
    # print('Final step approximation error:{:.2f}\nAverage approximation error: {:.2f}'.format(error, avg_error))
    #error, avg_error = analyzer.get_error(input_constraint,output_constraint)
   # print('Final step approximation error:{:.2f}\nAverage approximation error: {:.2f}'.format(error, avg_error))
   # import pdb
    #pdb.set_trace()
    analyzer.visualize(input_constraint, output_constraint, show_samples=True, prob_list =prob_list ,**analyzer_info)
    print("--- done. ---")