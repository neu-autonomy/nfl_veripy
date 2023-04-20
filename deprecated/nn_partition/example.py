import numpy as np
import nn_partition.analyzers as analyzers
from nn_partition.models.models import (
    model_xiang_2020_robot_arm,
    random_model,
)
import argparse
import os


def main(args):

    np.random.seed(seed=0)
    stats = {}

    # Setup NN
    if args.model == "robot_arm":
        torch_model, model_info = model_xiang_2020_robot_arm(
            activation=args.activation
        )
        if args.input_range is None:
            input_range = np.array(
                [  # (num_inputs, 2)
                    [np.pi / 3, 2 * np.pi / 3],  # x0min, x0max
                    [np.pi / 3, 2 * np.pi / 3],  # x1min, x1max
                ]
            )
        else:
            import ast

            input_range = np.array(
                ast.literal_eval(args.input_range)
            )
    elif args.model == "random_weights":
        neurons = [2, 50, 2]
        torch_model, model_info = random_model(
            activation=args.activation, neurons=neurons, seed=0
        )
        if args.input_range is None:
            input_range = np.zeros((model_info["model_neurons"][0], 2))
            input_range[:, 1] = 1.0
        else:
            import ast

            input_range = np.array(
                ast.literal_eval(args.input_range)
            )
    else:
        raise NotImplementedError

    partitioner_hyperparams = {
        "num_simulations": args.num_simulations,
        "type": args.partitioner,
        "termination_condition_type": args.term_type,
        "termination_condition_value": args.term_val,
        "interior_condition": args.interior_condition,
        "make_animation": args.make_animation,
        "show_animation": args.show_animation,
        "show_input": args.show_input,
        "show_output": args.show_output,
        "num_partitions": args.num_partitions,
    }
    propagator_hyperparams = {
        "type": args.propagator,
        "input_shape": input_range.shape[:-1],
    }

    # Run analysis & generate a plot
    analyzer = analyzers.Analyzer(torch_model)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams

    if args.estimate_runtime:
        raise NotImplementedError
    else:
        output_range, analyzer_info = analyzer.get_output_range(input_range)

    if args.estimate_error:

        error = analyzer.get_error(input_range, output_range, **analyzer_info)
        stats['error'] = error

    stats['output_range'] = output_range

    # Generate a visualization of the input/output mapping
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
                if key not in ["make_animation", "show_animation", "type"]
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
        model_str = args.model + "_" + args.activation + "_"
        analyzer_info["save_name"] = (
            save_dir
            + model_str
            + partitioner_hyperparams["type"]
            + "_"
            + propagator_hyperparams["type"]
            + "_"
            + pars
        )
        if len(pars2) > 0:
            analyzer_info["save_name"] = (
                analyzer_info["save_name"] + "_" + pars2
            )
        analyzer_info["save_name"] = analyzer_info["save_name"] + ".png"

    if args.save_plot or args.show_plot:
        # Plot shape/label settings
        labels = {
            "input": [
                label if label != "None" else None
                for label in args.input_plot_labels
            ],
            "output": [
                label if label != "None" else None
                for label in args.output_plot_labels
            ],
        }
        aspects = {
            "input": args.input_plot_aspect,
            "output": args.output_plot_aspect,
        }

        # Generate the plot & save
        analyzer.visualize(
            input_range,
            output_range,
            show=args.show_plot,
            show_samples=True,
            show_legend=False,
            show_input=args.show_input,
            show_output=args.show_output,
            title=None,
            labels=labels,
            aspects=aspects,
            **analyzer_info
        )

    print("done.")
    return stats, analyzer_info


def setup_parser():

    parser = argparse.ArgumentParser(description="Analyze a NN.")
    parser.add_argument(
        "--model",
        default="robot_arm",
        choices=["robot_arm", "random_weights"],
        help="which NN to analyze (default: robot_arm)",
    )
    parser.add_argument(
        "--activation",
        default="tanh",
        choices=["tanh", "sigmoid", "relu"],
        help="nonlinear activation fn in NN (default: tanh)",
    )
    parser.add_argument(
        "--partitioner",
        default="GreedySimGuided",
        choices=[
            "None",
            "Uniform",
            "SimGuided",
            "GreedySimGuided",
            "AdaptiveGreedySimGuided",
            "UnGuided",
        ],
        help="which partitioner to use (default: GreedySimGuided)",
    )
    parser.add_argument(
        "--propagator",
        default="CROWN_LIRPA",
        choices=[
            "IBP",
            "CROWN",
            "CROWN_LIRPA",
            "IBP_LIRPA",
            "CROWN-IBP_LIRPA",
            "FastLin_LIRPA",
            "Exhaustive_LIRPA",
            "SDP",
        ],
        help="which propagator to use (default: CROWN_LIRPA)",
    )
    parser.add_argument(
        "--input_range",
        default=None,
        help="2*num_inputs values (default: None)",
    )
    parser.add_argument(
        "--term_type",
        default="time_budget",
        choices=[
            "time_budget",
            "verify",
            "input_cell_size",
            "num_propagator_calls",
            "pct_improvement",
            "pct_error",
        ],
        help="type of condition to terminate (default: time_budget)",
    )
    parser.add_argument(
        "--term_val",
        default=2.0,
        type=float,
        help="value of condition to terminate (default: 2)",
    )
    parser.add_argument(
        "--interior_condition",
        default="lower_bnds",
        choices=["lower_bnds", "linf", "convex_hull"],
        help="type of bound to optimize for (default: lower_bnds)",
    )
    parser.add_argument(
        "--num_simulations",
        default=1e4,
        help="how many MC samples to begin with (default: 1e4)",
    )
    parser.add_argument(
        "--num_partitions",
        default=16,
        type=int,
        help="if uniform, how many cells to split into (default: 16)",
    )

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
        "--show_input",
        dest="show_input",
        action="store_true",
        help="whether to show the input partition in the plot",
    )
    parser.add_argument(
        "--skip_show_input", dest="show_input", action="store_false"
    )
    parser.set_defaults(show_input=True)

    parser.add_argument(
        "--show_output",
        dest="show_output",
        action="store_true",
        help="whether to show the output set in the plot",
    )
    parser.add_argument(
        "--skip_show_output", dest="show_output", action="store_false"
    )
    parser.set_defaults(show_output=True)

    parser.add_argument(
        "--input_plot_labels",
        metavar="N",
        default=["Input", None],
        type=str,
        nargs="+",
        help='x and y labels on input plot (default: ["Input", None])',
    )
    parser.add_argument(
        "--output_plot_labels",
        metavar="N",
        default=["Output", None],
        type=str,
        nargs="+",
        help='x and y labels on output plot (default: ["Output", None])',
    )
    parser.add_argument(
        "--input_plot_aspect",
        default="auto",
        choices=["auto", "equal"],
        help="aspect ratio on input plot (default: auto)",
    )
    parser.add_argument(
        "--output_plot_aspect",
        default="auto",
        choices=["auto", "equal"],
        help="aspect ratio on output plot (default: auto)",
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
        "--estimate_error", dest="estimate_error", action="store_true"
    )
    parser.add_argument(
        "--skip_estimate_error", dest="estimate_error", action="store_false"
    )
    parser.set_defaults(estimate_error=True)

    parser.add_argument(
        "--estimate_runtime", dest="estimate_runtime", action="store_true"
    )
    parser.set_defaults(estimate_runtime=False)

    return parser


if __name__ == "__main__":
    np.random.seed(seed=0)

    parser = setup_parser()

    args = parser.parse_args()

    main(args)
