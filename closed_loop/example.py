import numpy as np
import closed_loop.dynamics as dynamics
import closed_loop.analyzers as analyzers
import closed_loop.constraints as constraints
from closed_loop.utils.nn import load_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze a closed loop system w/ NN controller.')
    parser.add_argument('--system', default='double_integrator_mpc',
                        choices=["double_integrator_mpc", "quarotor"],
                        help='which system to analyze (default: double_integrator_mpc)')
    
    parser.add_argument('--state_feedback', dest='state_feedback', action='store_true',
                        help='whether to save the visualization')
    parser.add_argument('--output_feedback', dest='state_feedback', action='store_false')
    parser.set_defaults(state_feedback=True)

    parser.add_argument('--partitioner', default='Uniform',
                        choices=["None", "Uniform"],
                        help='which partitioner to use (default: Uniform)')
    parser.add_argument('--propagator', default='IBP',
                        choices=["IBP", "CROWN", "FastLin", "SDP"],
                        help='which propagator to use (default: IBP)')
    
    parser.add_argument('--num_partitions', default=np.array([4,4]),
                        help='how many cells per dimension to use (default: [4,4])')
    parser.add_argument('--boundaries', default="lp",
                        choices=["lp", "polytope"],
                        help='what shape of convex set to bound reachable sets (default: lp)')

    # parser.add_argument('--term_type', default='time_budget',
    #                     choices=["time_budget", "verify", "input_cell_size", "num_propagator_calls", "pct_improvement", "pct_error"],
    #                     help='type of condition to terminate (default: time_budget)')
    # parser.add_argument('--term_val', default=2., type=float,
    #                     help='value of condition to terminate (default: 2)')
    # parser.add_argument('--interior_condition', default='lower_bnds',
    #                     choices=["lower_bnds", "linf", "convex_hull"],
    #                     help='type of bound to optimize for (default: lower_bnds)')
    # parser.add_argument('--num_simulations', default=1e4,
    #                     help='how many MC samples to begin with (default: 1e4)')
    
    # parser.add_argument('--save_plot', dest='save_plot', action='store_true',
    #                     help='whether to save the visualization')
    # parser.add_argument('--skip_save_plot', dest='feature', action='store_false')
    # parser.set_defaults(save_plot=True)
    
    # parser.add_argument('--show_plot', dest='show_plot', action='store_true',
    #                     help='whether to show the visualization')
    # parser.add_argument('--skip_show_plot', dest='show_plot', action='store_false')
    # parser.set_defaults(show_plot=False)
    
    # parser.add_argument('--show_input', dest='show_input', action='store_true',
    #                     help='whether to show the input partition in the plot')
    # parser.add_argument('--skip_show_input', dest='show_input', action='store_false')
    # parser.set_defaults(show_input=True)
    
    # parser.add_argument('--show_output', dest='show_output', action='store_true',
    #                     help='whether to show the output set in the plot')
    # parser.add_argument('--skip_show_output', dest='show_output', action='store_false')
    # parser.set_defaults(show_output=True)

    # parser.add_argument('--input_plot_labels', metavar='N', default=["Input", None], type=str, nargs='+',
    #                     help='x and y labels on input partition plot (default: ["Input", None])')
    # parser.add_argument('--output_plot_labels', metavar='N', default=["Output", None], type=str, nargs='+',
    #                     help='x and y labels on output partition plot (default: ["Output", None])')
    # parser.add_argument('--input_plot_aspect', default="auto",
    #                     choices=["auto", "equal"],
    #                     help='aspect ratio on input partition plot (default: auto)')
    # parser.add_argument('--output_plot_aspect', default="auto",
    #                     choices=["auto", "equal"],
    #                     help='aspect ratio on output partition plot (default: auto)')

    args = parser.parse_args()


    np.random.seed(seed=0)

    # Load NN
    torch_model = load_model(name=args.system)
    
    ##############
    # Dynamics
    ##############
    if args.system == 'double_integrator_mpc':
        if args.state_feedback:
            dyn = dynamics.DoubleIntegrator()
        else:
            dyn = dynamics.DoubleIntegratorOutputFeedback()
        init_state_range = np.array([ # (num_inputs, 2)
                          [2.5, 3.0], # x0min, x0max
                          [-0.25, 0.25], # x1min, x1max
        ])
        t_max = 5
    elif args.system == 'quadrotor':
        if args.state_feedback:
            dyn = dynamics.Quadrotor()
        else:
            dyn = dynamics.QuadrotorOutputFeedback()
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
        "type": args.partitioner,
        "num_partitions": args.num_partitions,
        # "make_animation": False,
        # "show_animation": False,
    }
    propagator_hyperparams = {
        "type": args.propagator,
        "input_shape": init_state_range.shape[:-1],
    }

    # Run analysis & generate a plot

    analyzer = analyzers.ClosedLoopAnalyzer(torch_model, dyn)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    if args.boundaries == "polytope":
        from closed_loop.utils import init_state_range_to_polytope, get_polytope_A
        A_inputs, b_inputs = init_state_range_to_polytope(init_state_range)
        if system == 'quadrotor': A_out = A_inputs
        else: A_out = get_polytope_A(8)
        input_constraint = PolytopeInputConstraint(A_inputs, b_inputs)
        output_constraint = PolytopeOutputConstraint(A_out)
    elif args.boundaries == "lp":
        input_constraint = constraints.LpInputConstraint(range=init_state_range, p=np.inf)
        output_constraint = constraints.LpOutputConstraint(p=np.inf)
    else:
        raise NotImplementedError
    
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