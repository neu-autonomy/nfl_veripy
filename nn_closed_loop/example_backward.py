from nn_closed_loop.constraints.ClosedLoopConstraints import LpConstraint
import numpy as np
import nn_closed_loop.dynamics as dynamics
import nn_closed_loop.analyzers as analyzers
import nn_closed_loop.constraints as constraints
from nn_closed_loop.utils.nn import load_controller
from nn_closed_loop.utils.utils import range_to_polytope, plot_time_data
import os
import argparse


def main(args):
    np.random.seed(seed=0)
    stats = {}
    inputs_to_highlight=None

    # Dynamics
    if args.system == "double_integrator":
        if args.state_feedback:
            dyn = dynamics.DoubleIntegrator()
        else:
            raise NotImplementedError
            dyn = dynamics.DoubleIntegratorOutputFeedback()
        if args.final_state_range is None:
            final_state_range = np.array(
                [  # (num_inputs, 2)
                    [4.5, 5.0],  # x0min, x0max
                    [-0.25, 0.25],  # x1min, x1max
                ]
            )
        else:
            raise NotImplementedError
            import ast

            init_state_range = np.array(
                ast.literal_eval(args.init_state_range)
            )
    elif args.system == "ground_robot":
        if args.state_feedback:
            dyn = dynamics.GroundRobotSI()
        else:
            raise NotImplementedError
        if args.final_state_range is None:
            final_state_range = np.array(
                [  # (num_inputs, 2)
                    [-1., 1.],  # x0min, x0max
                    [-1., 1.],  # x1min, x1max
                ]
            )
        else:
            import ast

            final_state_range = np.array(
                ast.literal_eval(args.final_state_range)
            )
    elif args.system == "ground_robot_DI":
        if args.state_feedback:
            dyn = dynamics.GroundRobotDI()
        else:
            raise NotImplementedError
        if args.final_state_range is None:
            final_state_range = np.array(
                [  # (num_inputs, 2)
                    [-1, 1],  # x0min, x0max
                    [-1, 1],  # x1min, x1max
                    [-0.01, 0.01],
                    [-0.01, 0.01],
                ]
            )
        else:
            import ast

            final_state_range = np.array(
                ast.literal_eval(args.final_state_range)
            )
    elif args.system == "discrete_quadrotor":
        inputs_to_highlight = [
            {"dim": [0], "name": "$x$"},
            {"dim": [1], "name": "$y$"},
            {"dim": [2], "name": "$z$"},
        ]
        if args.state_feedback:
            dyn = dynamics.DiscreteQuadrotor()
        else:
            raise NotImplementedError
        if args.final_state_range is None:
            final_state_range = np.array(
                [  # (num_inputs, 2)
                    [-1, 1],  # x0min, x0max
                    [-1, 1],  # x1min, x1max
                    [1.5, 3.5],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                ]
            )
        else:
            import ast

            final_state_range = np.array(
                ast.literal_eval(args.final_state_range)
            )
    elif args.system == "4_double_integrators":
        if args.state_feedback:
            dyn = dynamics.DoubleIntegratorx4()
        else:
            raise NotImplementedError
        if args.final_state_range is None:
            final_state_range = np.array(
                [  # (num_inputs, 2)
                    [-0.5, 0.5],  # x0min, x0max
                    [-0.5, 0.5],  # x1min, x1max
                    [-0.5, 0.5],  # x2min, x2max
                    [-0.5, 0.5],  # x3min, x3max
                    [-0.01, 0.01],
                    [-0.01, 0.01],
                    [-0.01, 0.01],
                    [-0.01, 0.01],
                ]
            )
        else:
            import ast

            final_state_range = np.array(
                ast.literal_eval(args.final_state_range)
            )
    elif args.system == "quadrotor":
        inputs_to_highlight = [
            {"dim": [0], "name": "$x$"},
            {"dim": [1], "name": "$y$"},
            {"dim": [2], "name": "$z$"},
        ]
        if args.state_feedback:
            dyn = dynamics.Quadrotor()
        else:
            dyn = dynamics.QuadrotorOutputFeedback()
        if args.final_state_range is None:
            final_state_range = np.array(
                [  # (num_inputs, 2)
                    [-5-0.25, -0.25, 2, -0.01, -0.01, -0.01],
                    [-5+0.25, 0.25, 2.5, 0.01, 0.01, 0.01],
                ]
            ).T
        else:
            import ast

            final_state_range = np.array(
                ast.literal_eval(args.final_state_range)
            )
    else:
            raise NotImplementedError
            import ast

    if args.num_partitions is None:
        num_partitions = np.array([1, 1, 1, 1])
    else:
        import ast

        num_partitions = np.array(
            ast.literal_eval(args.num_partitions)
        )

    if args.init_state_range is None:
        init_constraint = None
    else:
        import ast
        init_state_range = np.array(
            ast.literal_eval(args.init_state_range)
        )
        init_constraint = LpConstraint(init_state_range)


    partitioner_hyperparams = {
        "type": args.partitioner,
        "num_partitions": num_partitions,
    }
    propagator_hyperparams = {
        "type": args.propagator,
        "input_shape": final_state_range.shape[:-1],
    }

    # Load NN control policy
    controller = load_controller(
        system=dyn.__class__.__name__,
        model_name=args.controller,
    )

    controller_name=None
    if args.show_policy:
        controller_name = vars(args)['controller']

    # Set up analyzer (+ partitioner + propagator)
    analyzer = analyzers.ClosedLoopBackwardAnalyzer(controller, dyn)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    # Set up initial state set (and placeholder for reachable sets)
    if args.boundaries == "polytope":
        A_out, b_out = range_to_polytope(final_state_range)
        output_constraint1 = constraints.PolytopeConstraint(
            A=A_out, b=[b_out]
        )
        input_constraint = constraints.PolytopeConstraint(None, None)
        output_constraint = [output_constraint1]
    elif args.boundaries == "lp":
        output_constraint1 = constraints.LpConstraint(
            range=final_state_range, p=np.inf
        )
        input_constraint = constraints.LpConstraint(p=np.inf, range=None)
        final_state_range2 = np.array(
                [  # (num_inputs, 2)
                    [6.5, 7.0],  # x0min, x0max
                    [-0.25, 0.25],  # x1min, x1max
                ]
            )
        output_constraint2 = constraints.LpConstraint(
            range=final_state_range2, p=np.inf
        )
        # output_constraint = [output_constraint2,output_constraint1]
        output_constraint = [output_constraint1]
    else:
        raise NotImplementedError

    if args.estimate_runtime:
        # Run the analyzer N times to compute an estimated runtime
        import time

        num_calls = 5
        times = np.empty(num_calls)
        final_errors = np.empty(num_calls)
        avg_errors = np.empty(num_calls, dtype=np.ndarray)
        all_errors = np.empty(num_calls, dtype=np.ndarray)
        output_constraints = np.empty(num_calls, dtype=object)
        for num in range(num_calls):
            print('call: {}'.format(num))
            t_start = time.time()
            input_constraint_list, analyzer_info_list = analyzer.get_backprojection_set(
                output_constraint, input_constraint, t_max=args.t_max, num_partitions=num_partitions, overapprox=args.overapprox, refined=args.refined
            )
            t_end = time.time()
            t = t_end - t_start
            times[num] = t
            backprojection_sets = input_constraint_list[0]
            target_set = output_constraint[0]
            # final_error, avg_error, all_error = analyzer.get_backprojection_error(target_set, backprojection_sets, t_max=args.t_max)

            # final_errors[num] = final_error
            # avg_errors[num] = avg_error
            # all_errors[num] = all_error
            output_constraints[num] = output_constraint

        stats['runtimes'] = times
        # stats['final_step_errors'] = final_errors
        # stats['avg_errors'] = avg_errors
        # stats['all_errors'] = all_errors
        stats['output_constraints'] = output_constraints

        print("All times: {}".format(times))
        print("Avg time: {} +/- {}".format(times.mean(), times.std()))
        # print("Final Error: {}".format(final_errors[-1]))
    else:
        # Run analysis once
        # Run analysis & generate a plot
        import time
        t_start = time.time()
        input_constraint_list, analyzer_info_list = analyzer.get_backprojection_set(
            output_constraint, input_constraint, t_max=args.t_max, num_partitions=num_partitions, overapprox=args.overapprox, refined=args.refined
        )
        t_end = time.time()
        print(t_end-t_start)
        
        # plot_time_data(analyzer_info_list[0])
        
    controller_name=None
    if args.show_policy:
            controller_name = vars(args)['controller']

    if args.save_plot:
        save_dir = "{}/results/examples_backward/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        os.makedirs(save_dir, exist_ok=True)

        # Ugly logic to embed parameters in filename:
        pars = "_".join(
            [
                str(key) + "_" + str(value)
                for key, value in sorted(
                    partitioner_hyperparams.items(), key=lambda kv: kv[0]
                )
                if key
                not in [
                    "make_animation",
                    "show_animation",
                    "type",
                    "num_partitions",
                ]
            ]
        )
        pars2 = "_".join(
            [
                str(key) + "_" + str(value)
                for key, value in sorted(
                    propagator_hyperparams.items(), key=lambda kv: kv[0]
                )
                if key not in ["input_shape", "type"]
            ]
        )
        analyzer_info_list[0]["save_name"] = (
            save_dir
            + args.system
            + pars
            + "_"
            + partitioner_hyperparams["type"]
            + "_"
            + propagator_hyperparams["type"]
            + "_"
            # + "tmax"
            # + "_"
            # + str(round(args.t_max, 1))
            # + "_"
            + args.boundaries
            + "_"
            + str(args.num_polytope_facets)
            + "_"
            + "partitions"
            + "_"
            + np.array2string(num_partitions, separator='_')[1:-1]
        )
        if len(pars2) > 0:
            analyzer_info_list[0]["save_name"] = (
                analyzer_info_list[0]["save_name"] + "_" + pars2
            )
        analyzer_info_list[0]["save_name"] = analyzer_info_list[0]["save_name"] + ".png"
    # import pdb; pdb.set_trace()
    if args.show_plot or args.save_plot:
        analyzer.visualize(
            input_constraint_list,
            output_constraint,
            analyzer_info_list,
            show_samples=args.show_samples,
            show_trajectories=args.show_trajectories,
            show_convex_hulls=args.show_convex_hulls,
            inputs_to_highlight=inputs_to_highlight,
            show=args.show_plot,
            labels=args.plot_labels,
            aspect=args.plot_aspect,
            plot_lims=args.plot_lims,
            initial_constraint=[init_constraint],
            controller_name=controller_name,
            show_BReach=args.show_BReach
        )

    return stats, analyzer_info_list


def setup_parser():

    parser = argparse.ArgumentParser(
        description="Backward analyze a closed loop system w/ NN controller."
    )
    parser.add_argument(
        "--system",
        default="double_integrator",
        choices=["double_integrator", "quadrotor", "ground_robot", "ground_robot_DI", "4_double_integrators", "discrete_quadrotor"],
        help="which system to analyze (default: double_integrator_mpc)",
    )
    parser.add_argument(
        "--controller",
        default="default",
        help="which NN controller to load (e.g., sine_wave_controller for ground_robot) (default: default)",
    )
    parser.add_argument(
        "--init_state_range",
        default=None,
        help="2*num_states values (default: None)",
    )
    parser.add_argument(
        "--final_state_range",
        default=None,
        help="2*num_states values (default: None)",
    )

    parser.add_argument(
        "--state_feedback",
        dest="state_feedback",
        action="store_true",
        help="whether to save the visualization",
    )
    parser.add_argument(
        "--output_feedback", dest="state_feedback", action="store_false"
    )
    parser.set_defaults(state_feedback=True)

    parser.add_argument(
        "--partitioner",
        default="None",
        choices=["None"],
        help="which partitioner to use (work in progress for backward...)",
    )
    parser.add_argument(
        "--propagator",
        default="IBP",
        choices=["IBP", "CROWN", "FastLin", "SDP", "CROWNLP", "CROWNNStep"],
        help="which propagator to use (default: IBP)",
    )

    parser.add_argument(
        "--num_partitions",
        default=None,
        help="how many cells per dimension to use (default: None)",
    )
    parser.add_argument(
        "--boundaries",
        default="lp",
        choices=["lp", "polytope"],
        help="what shape of convex set to bound reachable sets (default: lp)",
    )
    parser.add_argument(
        "--num_polytope_facets",
        default=8,
        type=int,
        help="how many facets on constraint polytopes (default: 8)",
    )
    parser.add_argument(
        "--t_max",
        default=1.,
        type=float,
        help="seconds into future to compute reachable sets (default: 2.)",
    )

    parser.add_argument(
        "--estimate_runtime", dest="estimate_runtime", action="store_true"
    )
    parser.set_defaults(estimate_runtime=False)

    parser.add_argument(
        "--save_plot",
        dest="save_plot",
        action="store_true",
        help="whether to save the visualization",
    )
    parser.add_argument(
        "--skip_save_plot", dest="feature", action="store_false"
    )
    parser.set_defaults(save_plot=True)

    parser.add_argument(
        "--show_plot",
        dest="show_plot",
        action="store_true",
        help="whether to show the visualization",
    )
    parser.add_argument(
        "--skip_show_plot", dest="show_plot", action="store_false"
    )
    parser.set_defaults(show_plot=False)

    parser.add_argument(
        "--plot_labels",
        metavar="N",
        default=["x_0", "x_1"],
        type=str,
        nargs="+",
        help='x and y labels on input plot (default: ["Input", None])',
    )
    parser.add_argument(
        "--plot_aspect",
        default="auto",
        choices=["auto", "equal"],
        help="aspect ratio on input partition plot (default: auto)",
    )
    parser.add_argument(
        "--plot_lims",
        default=None,
        help='x and y lims on plot (default: None)',
    )

    parser.add_argument(
        "--overapprox",
        dest="overapprox",
        action="store_true",
        help="whether compute an overapproximation or underapproximation",
    )
    parser.add_argument(
        "--underapprox", dest="overapprox", action="store_false"
    )
    parser.set_defaults(overapprox=False)
    parser.add_argument(
        "--refined",
        dest="refined",
        action="store_true",
        help="whether to use extended set of constraints for BReach-LP",
    )
    parser.set_defaults(refined=False)
    parser.add_argument(
        "--show_BReach",
        dest="show_BReach",
        action="store_true",
        help="whether to show results of BReach-LP when using ReBReach-LP",
    )
    parser.set_defaults(show_BReach=False)
    parser.add_argument(
        "--show_policy",
        dest="show_policy",
        action="store_true",
        help="Displays policy as a function of state (only valid for ground_robot and ground_robot_DI)"
    )
    parser.add_argument(
        "--show_trajectories",
        dest="show_trajectories",
        action="store_true",
        help="Show trajectories starting from initial condition"
    )
    parser.add_argument(
        "--show_samples",
        dest="show_samples",
        action="store_true",
        help="Show samples starting from initial condition"
    )
    parser.add_argument(
        "--show_convex_hulls",
        dest="show_convex_hulls",
        action="store_true",
        help="Show convex hulls of true backprojection sets"
    )
    return parser


if __name__ == "__main__":

    parser = setup_parser()

    args = parser.parse_args()

    main(args)
