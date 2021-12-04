import numpy as np
import nn_closed_loop.dynamics as dynamics
import nn_closed_loop.analyzers as analyzers
import nn_closed_loop.constraints as constraints
from nn_closed_loop.utils.nn import load_controller
from nn_closed_loop.utils.utils import range_to_polytope
import os
import argparse


def main(args):
    np.random.seed(seed=0)
    stats = {}

    # Load NN control policy
    controller = load_controller(name=args.system)

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
                    [2.5, 3.0],  # x0min, x0max
                    [-0.25, 0.25],  # x1min, x1max
                ]
            )
        else:
            raise NotImplementedError
            import ast

            init_state_range = np.array(
                ast.literal_eval(args.init_state_range)
            )
    else:
        raise NotImplementedError

    if args.num_partitions is None:
        num_partitions = np.array([2, 2])
    else:
        import ast

        num_partitions = np.array(
            ast.literal_eval(args.num_partitions)
        )

    partitioner_hyperparams = {
        "type": args.partitioner,
        "num_partitions": num_partitions,
    }
    propagator_hyperparams = {
        "type": args.propagator,
        "input_shape": final_state_range.shape[:-1],
    }

    # Set up analyzer (+ partitioner + propagator)
    analyzer = analyzers.ClosedLoopBackwardAnalyzer(controller, dyn)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    # Set up initial state set (and placeholder for reachable sets)
    if args.boundaries == "polytope":
        A_out, b_out = range_to_polytope(final_state_range)
        output_constraint = constraints.PolytopeConstraint(
            A=A_out, b=[b_out]
        )
        input_constraint = constraints.PolytopeConstraint(None, None)
    elif args.boundaries == "lp":
        output_constraint = constraints.LpConstraint(
            range=final_state_range, p=np.inf
        )
        input_constraint = constraints.LpConstraint(p=np.inf, range=None)
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
            input_constraint, analyzer_info = analyzer.get_backprojection_set(
                output_constraint, input_constraint, t_max=None, num_partitions=num_partitions
            )
            t_end = time.time()
            t = t_end - t_start
            times[num] = t

        stats['runtimes'] = times

        print("All times: {}".format(times))
        print("Avg time: {} +/- {}".format(times.mean(), times.std()))
    else:
        # Run analysis once
        # Run analysis & generate a plot
        input_constraint, analyzer_info = analyzer.get_backprojection_set(
            output_constraint, input_constraint, t_max=None, num_partitions=num_partitions
        )

    # print(input_constraint.A, input_constraint.b)
    # error, avg_error = analyzer.get_error(input_constraint,output_constraint, t_max=args.t_max)
    # print('Final step approximation error:{:.2f}\nAverage approximation error: {:.2f}'.format(error, avg_error))

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
        analyzer_info["save_name"] = (
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
            analyzer_info["save_name"] = (
                analyzer_info["save_name"] + "_" + pars2
            )
        analyzer_info["save_name"] = analyzer_info["save_name"] + ".png"

    if args.show_plot or args.save_plot:
        analyzer.visualize(
            input_constraint[0],
            output_constraint,
            show_samples=True,
            show=args.show_plot,
            labels=args.plot_labels,
            aspect=args.plot_aspect,
            plot_lims=args.plot_lims,
            **analyzer_info
        )

    return stats


def setup_parser():

    parser = argparse.ArgumentParser(
        description="Backward analyze a closed loop system w/ NN controller."
    )
    parser.add_argument(
        "--system",
        default="double_integrator",
        choices=["double_integrator", "quadrotor"],
        help="which system to analyze (default: double_integrator_mpc)",
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
        choices=["IBP", "CROWN", "FastLin", "SDP", "CROWNLP"],
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
    # parser.add_argument(
    #     "--t_max",
    #     default=2.0,
    #     type=float,
    #     help="seconds into future to compute reachable sets (default: 2.)",
    # )

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

    return parser


if __name__ == "__main__":

    parser = setup_parser()

    args = parser.parse_args()

    main(args)
