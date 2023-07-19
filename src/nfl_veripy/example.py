"""Runs a closed-loop reachability experiment according to a param file."""

from nfl_veripy.utils.utils import suppress_unecessary_logs

suppress_unecessary_logs()  # needs to happen before other imports

import argparse  # noqa: E402
import ast  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import nfl_veripy.analyzers as analyzers  # noqa: E402
import nfl_veripy.constraints as constraints  # noqa: E402
import nfl_veripy.dynamics as dynamics  # noqa: E402
from nfl_veripy.utils.nn import load_controller  # noqa: E402

dir_path = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def main_forward(params: dict) -> Tuple[Dict, Dict]:
    """Runs a forward reachability analysis experiment according to params."""
    np.random.seed(seed=0)
    stats = {}

    dyn = dynamics.get_dynamics_instance(
        params["system"]["type"], params["system"]["feedback"]
    )

    controller = load_controller(
        system=dyn.__class__.__name__,
        model_name=params["system"]["controller"],
    )

    # Set up analyzer (+ parititoner + propagator)
    analyzer = analyzers.ClosedLoopAnalyzer(controller, dyn)
    analyzer.partitioner = params["analysis"]["partitioner"]
    analyzer.propagator = params["analysis"]["propagator"]

    initial_state_range = np.array(
        ast.literal_eval(params["analysis"]["initial_state_range"])
    )
    initial_state_set = constraints.state_range_to_constraint(
        initial_state_range, params["analysis"]["propagator"]["boundary_type"]
    )

    if params["analysis"]["estimate_runtime"]:
        # Run the analyzer N times to compute an estimated runtime
        times = np.empty(params["analysis"]["num_calls"])
        final_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        avg_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_reachable_sets = np.empty(
            params["analysis"]["num_calls"], dtype=object
        )
        for num in range(params["analysis"]["num_calls"]):
            logger.info(f"call: {num}")
            t_start = time.time()
            reachable_sets, analyzer_info = analyzer.get_reachable_set(
                initial_state_set, t_max=params["analysis"]["t_max"]
            )
            t_end = time.time()
            times[num] = t_end - t_start

            if num == 0:
                final_error, avg_error, all_error = analyzer.get_error(
                    initial_state_set,
                    reachable_sets,
                    t_max=params["analysis"]["t_max"],
                )
                final_errors[num] = final_error
                avg_errors[num] = avg_error
                all_errors[num] = all_error
                all_reachable_sets[num] = reachable_sets

        stats["runtimes"] = times
        stats["final_step_errors"] = final_errors
        stats["avg_errors"] = avg_errors
        stats["all_errors"] = all_errors
        stats["reachable_sets"] = all_reachable_sets

        logger.info(f"All times: {times}")
        logger.info(f"Avg time: {times.mean()} +/- {times.std()}")
    else:
        # Run analysis once
        t_start = time.time()
        reachable_sets, analyzer_info = analyzer.get_reachable_set(
            initial_state_set, t_max=params["analysis"]["t_max"]
        )
        t_end = time.time()
        logger.info(f"Runtime: {t_end - t_start} sec.")
        stats["reachable_sets"] = reachable_sets

    if params["analysis"]["estimate_error"]:
        final_error, avg_error, errors = analyzer.get_error(
            initial_state_set,
            reachable_sets,
            t_max=params["analysis"]["t_max"],
        )
        logger.info(f"Final step approximation error: {final_error}")
        logger.info(f"Avg errors: {avg_error}")
        logger.info(f"All errors: {errors}")

    if params["visualization"]["save_plot"]:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = f"{this_file_dir}/results/examples/"
        os.makedirs(save_dir, exist_ok=True)

        # Ugly logic to embed parameters in filename:
        pars = "_".join(
            [
                str(key) + "_" + str(value)
                for key, value in sorted(
                    params["analysis"]["partitioner"].items(),
                    key=lambda kv: kv[0],
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
                    params["analysis"]["propagator"].items(),
                    key=lambda kv: kv[0],
                )
                if key not in ["input_shape", "type"]
            ]
        )
        analyzer_info["save_name"] = (
            save_dir
            + dyn.name
            + pars
            + "_"
            + params["analysis"]["partitioner"]["type"]
            + "_"
            + params["analysis"]["propagator"]["type"]
            + "_"
            + "tmax"
            + "_"
            + str(round(params["analysis"]["t_max"], 1))
            + "_"
            + params["analysis"]["propagator"]["boundary_type"]
        )
        if len(pars2) > 0:
            analyzer_info["save_name"] = (
                analyzer_info["save_name"] + "_" + pars2
            )
        analyzer_info["save_name"] = analyzer_info["save_name"] + ".png"

    if (
        params["visualization"]["show_plot"]
        or params["visualization"]["save_plot"]
    ):
        analyzer.visualize(
            initial_state_set,
            reachable_sets,
            show_samples=params["visualization"]["show_samples"],
            show_trajectories=params["visualization"]["show_trajectories"],
            show=params["visualization"]["show_plot"],
            axis_dims=[[a] for a in params["visualization"]["plot_dims"]],
            axis_labels=params["visualization"]["plot_axis_labels"],
            aspect=params["visualization"]["plot_aspect"],
            plot_lims=params["visualization"]["plot_lims"],
            iteration=None,
            controller_name=None,
            **analyzer_info,
        )

    return stats, analyzer_info


def main_backward(params: dict) -> tuple[dict, dict]:
    """Runs a backward reachability analysis experiment according to params."""

    np.random.seed(seed=0)
    stats = {}

    if params["system"]["feedback"] != "FullState":
        raise ValueError(
            "Currently only support state feedback for backward reachability."
        )

    dyn = dynamics.get_dynamics_instance(
        params["system"]["type"], params["system"]["feedback"]
    )

    controller = load_controller(
        system=dyn.__class__.__name__,
        model_name=params["system"]["controller"],
    )

    # Set up analyzer (+ parititoner + propagator)
    analyzer = analyzers.ClosedLoopBackwardAnalyzer(controller, dyn)
    analyzer.partitioner = params["analysis"]["partitioner"]
    analyzer.propagator = params["analysis"]["propagator"]

    final_state_range = np.array(
        ast.literal_eval(params["analysis"]["final_state_range"])
    )
    target_set = constraints.state_range_to_constraint(
        final_state_range, params["analysis"]["propagator"]["boundary_type"]
    )

    if params["analysis"]["estimate_runtime"]:
        # Run the analyzer N times to compute an estimated runtime

        times = np.empty(params["analysis"]["num_calls"])
        final_errors = np.empty(params["analysis"]["num_calls"])
        avg_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_errors = np.empty(
            params["analysis"]["num_calls"], dtype=np.ndarray
        )
        all_backprojection_sets = np.empty(
            params["analysis"]["num_calls"], dtype=object
        )
        target_sets = np.empty(params["analysis"]["num_calls"], dtype=object)
        for num in range(params["analysis"]["num_calls"]):
            logger.info(f"call: {num}")
            t_start = time.time()
            (
                backprojection_sets,
                analyzer_info,
            ) = analyzer.get_backprojection_set(
                target_set,
                t_max=params["analysis"]["t_max"],
                overapprox=params["analysis"]["overapprox"],
            )
            t_end = time.time()
            times[num] = t_end - t_start

            if num == 0:
                (
                    final_error,
                    avg_error,
                    all_error,
                ) = analyzer.get_backprojection_error(
                    target_set,
                    backprojection_sets,
                    t_max=params["analysis"]["t_max"],
                )

                final_errors[num] = final_error
                avg_errors[num] = avg_error
                all_errors[num] = all_error
                all_backprojection_sets[num] = backprojection_sets
                target_sets[num] = target_sets

        stats["runtimes"] = times
        stats["final_step_errors"] = final_errors
        stats["avg_errors"] = avg_errors
        stats["all_errors"] = all_errors
        stats["all_backprojection_sets"] = all_backprojection_sets
        stats["target_sets"] = target_sets
        stats["avg_runtime"] = times.mean()

        logger.info(f"All times: {times}")
        logger.info(f"Avg time: {times.mean()} +/- {times.std()}")
        logger.info(f"Final Error: {final_errors[-1]}")
        logger.info(f"Avg Error: {avg_errors[-1]}")
    else:
        # Run analysis once
        # Run analysis & generate a plot
        backprojection_sets, analyzer_info = analyzer.get_backprojection_set(
            target_set,
            t_max=params["analysis"]["t_max"],
            overapprox=params["analysis"]["overapprox"],
        )
        stats["backprojection_sets"] = backprojection_sets

    controller_name = None
    if params["visualization"]["show_policy"]:
        controller_name = params["system"]["controller"]

    if params["visualization"]["save_plot"]:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = f"{this_file_dir}/results/examples_backward/"
        os.makedirs(save_dir, exist_ok=True)

        # Ugly logic to embed parameters in filename:
        pars = "_".join(
            [
                str(key) + "_" + str(value)
                for key, value in sorted(
                    params["analysis"]["partitioner"].items(),
                    key=lambda kv: kv[0],
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
                    params["analysis"]["propagator"].items(),
                    key=lambda kv: kv[0],
                )
                if key not in ["input_shape", "type"]
            ]
        )
        analyzer_info["save_name"] = (
            save_dir
            + params["system"]["type"]
            + pars
            + "_"
            + params["analysis"]["partitioner"]["type"]
            + "_"
            + params["analysis"]["propagator"]["type"]
            + "_"
            # + "tmax"
            # + "_"
            # + str(round(params["analysis"]["t_max"], 1))
            # + "_"
            + params["analysis"]["propagator"]["boundary_type"]
            + "_"
            # + str(params["analysis"]["propagator"]["num_polytope_facets"])
            # + "_"
            + "partitions"
            + "_"
            # + np.array2string(num_partitions, separator='_')[1:-1]
        )
        if len(pars2) > 0:
            analyzer_info["save_name"] = (
                analyzer_info["save_name"] + "_" + pars2
            )
        analyzer_info["save_name"] = analyzer_info["save_name"] + ".png"

    if params["analysis"].get("initial_state_range", None) is None:
        initial_state_set = None
    else:
        initial_state_range = np.array(
            ast.literal_eval(params["analysis"]["initial_state_range"])
        )
        initial_state_set = constraints.LpConstraint(initial_state_range)

    if (
        params["visualization"]["show_plot"]
        or params["visualization"]["save_plot"]
    ):
        analyzer.visualize(
            backprojection_sets,
            target_set,
            analyzer_info,
            show=params["visualization"]["show_plot"],
            show_samples=params["visualization"]["show_samples"],
            show_samples_from_cells=params["visualization"][
                "show_samples_from_cells"
            ],
            show_trajectories=params["visualization"]["show_trajectories"],
            show_convex_hulls=params["visualization"]["show_convex_hulls"],
            axis_dims=params["visualization"]["plot_dims"],
            axis_labels=params["visualization"]["plot_axis_labels"],
            aspect=params["visualization"]["plot_aspect"],
            plot_lims=params["visualization"]["plot_lims"],
            initial_constraint=initial_state_set,
            controller_name=controller_name,
            show_BReach=params["visualization"]["show_BReach"],
        )

    return stats, analyzer_info


def setup_parser() -> dict:
    """Load yaml config file with experiment params."""
    parser = argparse.ArgumentParser(
        description="Analyze a closed loop system w/ NN controller."
    )

    parser.add_argument(
        "--config",
        type=str,
        help=(
            "Absolute or relative path to yaml file describing experiment"
            " configuration. Note: if this arg starts with 'example_configs/',"
            " the configs in the installed package will be used (ignoring the"
            " pwd)."
        ),
    )

    args = parser.parse_args()

    if args.config.startswith("example_configs/"):
        # Use the config files in the pip-installed package
        param_filename = f"{dir_path}/_static/{args.config}"
    else:
        # Use the absolute/relative path provided in args.config
        param_filename = f"{args.config}"

    with open(param_filename, mode="r", encoding="utf-8") as file:
        params = yaml.load(file, yaml.Loader)

    return params


if __name__ == "__main__":
    print("~~ test change ~~")
    experiment_params = setup_parser()

    if experiment_params["analysis"]["reachability_direction"] == "forward":
        main_forward(experiment_params)
    if experiment_params["analysis"]["reachability_direction"] == "backward":
        main_backward(experiment_params)
