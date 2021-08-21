import numpy as np
import nn_closed_loop.dynamics as dynamics
import nn_closed_loop.analyzers as analyzers
import nn_closed_loop.constraints as constraints
from nn_closed_loop.utils.nn import load_controller, load_controller_unity
from nn_closed_loop.utils.utils import (
    range_to_polytope,
    get_polytope_A,
)
import os
import argparse


def main(args):
    np.random.seed(seed=0)
    stats = {}

    # Load NN control policy
    controller = load_controller(name=args.system)

    # Dynamics
    if args.system == "double_integrator":
        inputs_to_highlight = [
            {"dim": [0], "name": "$x_0$"},
            {"dim": [1], "name": "$x_1$"},
        ]
        if args.state_feedback:
            dyn = dynamics.DoubleIntegrator()
        else:
            dyn = dynamics.DoubleIntegratorOutputFeedback()
        if args.init_state_range is None:
            init_state_range = np.array(
                [  # (num_inputs, 2)
                    [2.5, 3.0],  # x0min, x0max
                    [-0.25, 0.25],  # x1min, x1max
                ]
            )
        else:
            import ast

            init_state_range = np.array(
                ast.literal_eval(args.init_state_range)
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
        if args.init_state_range is None:
            init_state_range = np.array(
                [  # (num_inputs, 2)
                    [4.65, 4.65, 2.95, 0.94, -0.01, -0.01],
                    [4.75, 4.75, 3.05, 0.96, 0.01, 0.01],
                ]
            ).T
        else:
            import ast

            init_state_range = np.array(
                ast.literal_eval(args.init_state_range)
            )
    elif args.system == "duffing":
        inputs_to_highlight = [
            {"dim": [0], "name": "$x_0$"},
            {"dim": [1], "name": "$x_1$"},
        ]
        dyn = dynamics.Duffing()
        init_state_range = np.array(
            [  # (num_inputs, 2)
                [2.45, 2.55],  # x0min, x0max 
                [1.45, 1.55],  # x1min, x1max
            ]
        )
    elif args.system == "iss":
        inputs_to_highlight = [
            {"dim": [0], "name": "$x_0$"},
            {"dim": [1], "name": "$x_1$"},
        ]
        dyn = dynamics.ISS()
        init_state_range = 100 * np.ones((dyn.n, 2))
        init_state_range[:, 0] = init_state_range[:, 0] - 0.5
        init_state_range[:, 1] = init_state_range[:, 1] + 0.5
    elif args.system == "unity":
        inputs_to_highlight = [
            {"dim": [0], "name": "$x$"},
            {"dim": [1], "name": "$y$"},
        ]
        dyn = dynamics.Unity(args.nx, args.nu)
        if args.init_state_range is None:
            init_state_range = np.vstack([-np.ones(args.nx), np.ones(args.nx)]).T
        else:
            import ast

            init_state_range = np.array(
                ast.literal_eval(args.init_state_range)
            )
        controller = load_controller_unity(args.nx, args.nu)
    else:
        raise NotImplementedError

    if args.num_partitions is None:
        num_partitions = np.array([4, 4])
    else:
        import ast

        num_partitions = np.array(
            ast.literal_eval(args.num_partitions)
        )

    partitioner_hyperparams = {
        "type": args.partitioner,
        "num_partitions": num_partitions,
        "make_animation": args.make_animation,
        "show_animation": args.show_animation,
    }
    propagator_hyperparams = {
        "type": args.propagator,
        "input_shape": init_state_range.shape[:-1],
    }
    if args.propagator == "SDP":
        propagator_hyperparams["cvxpy_solver"] = args.cvxpy_solver

    # Set up analyzer (+ parititoner + propagator)
    analyzer = analyzers.ClosedLoopAnalyzer(controller, dyn)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    # Set up initial state set (and placeholder for reachable sets)
    if args.boundaries == "polytope":
        A_inputs, b_inputs = range_to_polytope(init_state_range)
        if args.system == "quadrotor":
            A_out = A_inputs
        else:
            A_out = get_polytope_A(args.num_polytope_facets)
        input_constraint = constraints.PolytopeConstraint(
            A_inputs, b_inputs
        )
        output_constraint = constraints.PolytopeConstraint(A_out)
    elif args.boundaries == "lp":
        input_constraint = constraints.LpConstraint(
            range=init_state_range, p=np.inf
        )
        output_constraint = constraints.LpConstraint(p=np.inf)
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
            output_constraint, analyzer_info = analyzer.get_reachable_set(
                input_constraint, output_constraint, t_max=args.t_max
            )
            t_end = time.time()
            t = t_end - t_start
            times[num] = t

            final_error, avg_error, all_error = analyzer.get_error(input_constraint, output_constraint, t_max=args.t_max)
            final_errors[num] = final_error
            avg_errors[num] = avg_error
            all_errors[num] = all_error
            output_constraints[num] = output_constraint

        stats['runtimes'] = times
        stats['final_step_errors'] = final_errors
        stats['avg_errors'] = avg_errors
        stats['all_errors'] = all_errors
        stats['output_constraints'] = output_constraints

        print("All times: {}".format(times))
        print("Avg time: {} +/- {}".format(times.mean(), times.std()))
    else:
        # Run analysis once
        output_constraint, analyzer_info = analyzer.get_reachable_set(
            input_constraint, output_constraint, t_max=args.t_max
        )

    if args.estimate_error:
        final_error, avg_error, errors = analyzer.get_error(input_constraint, output_constraint, t_max=args.t_max)
        print('Final step approximation error:{:.2f}\nAverage approximation error: {:.2f}\nAll errors: {}'.format(final_error, avg_error, errors))

    if args.save_plot:
        save_dir = "{}/results/examples/".format(
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
        analyzer_info["save_name"] = (
            save_dir
            + args.system
            + pars
            + "_"
            + partitioner_hyperparams["type"]
            + "_"
            + propagator_hyperparams["type"]
            + "_"
            + "tmax"
            + "_"
            + str(round(args.t_max, 1))
            + "_"
            + args.boundaries
            + "_"
            + str(args.num_polytope_facets)
        )
        if len(pars2) > 0:
            analyzer_info["save_name"] = (
                analyzer_info["save_name"] + "_" + pars2
            )
        analyzer_info["save_name"] = analyzer_info["save_name"] + ".png"

    if args.show_plot or args.save_plot:
        analyzer.visualize(
            input_constraint,
            output_constraint,
            show_samples=True,
            show=args.show_plot,
            labels=args.plot_labels,
            aspect=args.plot_aspect,
            iteration=None,
            inputs_to_highlight=inputs_to_highlight,
            **analyzer_info
        )

    return stats, analyzer_info


def setup_parser():

    parser = argparse.ArgumentParser(
        description="Analyze a closed loop system w/ NN controller."
    )
    parser.add_argument(
        "--system",
        default="double_integrator",
        choices=["double_integrator", "quadrotor", "duffing", "iss"],
        help="which system to analyze (default: double_integrator)",
    )
    parser.add_argument(
        "--init_state_range",
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
        "--cvxpy_solver",
        default="default",
        choices=["MOSEK", "default"],
        help="which solver to use with cvxpy (default: default)",
    )
    parser.add_argument(
        "--partitioner",
        default="Uniform",
        choices=["None", "Uniform", "SimGuided", "GreedySimGuided", "UnGuided"],
        help="which partitioner to use (default: Uniform)",
    )
    parser.add_argument(
        "--propagator",
        default="IBP",
        choices=["IBP", "CROWN", "FastLin", "SDP", "CROWNLP", "SeparableCROWN", "SeparableIBP", "SeparableSGIBP"],
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
        default=2.0,
        type=float,
        help="seconds into future to compute reachable sets (default: 2.)",
    )

    parser.add_argument(
        "--estimate_runtime", dest="estimate_runtime", action="store_true"
    )
    parser.set_defaults(estimate_runtime=False)

    parser.add_argument(
        "--estimate_error", dest="estimate_error", action="store_true"
    )
    parser.add_argument(
        "--skip_estimate_error", dest="estimate_error", action="store_false"
    )
    parser.set_defaults(estimate_error=True)

    parser.add_argument(
        "--save_plot",
        dest="save_plot",
        action="store_true",
        help="whether to save the visualization",
    )
    parser.add_argument(
        "--skip_save_plot", dest="save_plot", action="store_false"
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
        "--make_animation",
        dest="make_animation",
        action="store_true",
        help="whether to animate the partitioning process",
    )
    parser.add_argument(
        "--skip_make_animation", dest="make_animation", action="store_false"
    )
    parser.set_defaults(make_animation=False)
    parser.add_argument(
        "--show_animation",
        dest="show_animation",
        action="store_true",
        help="whether to show animation of the partitioning process",
    )
    parser.add_argument(
        "--skip_show_animation", dest="show_animation", action="store_false"
    )
    parser.set_defaults(show_animation=False)
    parser.add_argument(
        "--nx",
        default=2,
        help="number of states - only used for scalability expt (default: 2)",
    )
    parser.add_argument(
        "--nu",
        default=2,
        help="number of control inputs - only used for scalability expt (default: 2)",
    )

    return parser


if __name__ == "__main__":

    parser = setup_parser()

    args = parser.parse_args()

    main(args)
