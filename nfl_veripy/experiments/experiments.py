import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import nfl_veripy.analyzers as analyzers
import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import nfl_veripy.examples.example as ex
from nfl_veripy.utils.nn import load_controller

results_dir = "{}/results/logs/".format(
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(results_dir, exist_ok=True)


class Experiment:
    def __init__(self):
        self.info = {
            ("CROWN", "Uniform"): {
                "name": "Reach-LP-Partition",
                "color": "tab:green",
                "ls": "-",
            },
            ("CROWNNStep", "None"): {
                "name": "Reach-LP-N-Step",
                "color": "black",
                "ls": "--",
            },
            ("CROWN", "None"): {
                "name": "Reach-LP",
                "color": "tab:green",
                "ls": "--",
            },
            ("OVERT", "None"): {
                "name": "OVERT",
                "color": "tab:purple",
                "ls": "-",
            },
            ("SDP", "Uniform"): {
                "name": "Reach-SDP-Partition",
                "color": "tab:red",
                "ls": "-",
            },
            ("SDP", "None"): {
                "name": "Reach-SDP~\cite{hu2020reach}",
                "color": "tab:red",
                "ls": "--",
            },
            ("SeparableCROWN", "None"): {
                "name": "CL-CROWN",
            },
            ("SeparableSGIBP", "None"): {
                "name": "CL-SG-IBP~\cite{xiang2020reachable}",
            },
        }


class CompareMultipleCombos(Experiment):
    def __init__(self):
        self.filename = results_dir + "alg_error_{dt}_table.pkl"
        Experiment.__init__(self)

    def run(self):
        dt = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        parser = ex.setup_parser()
        args = parser.parse_args()

        args.save_plot = False
        args.show_plot = False
        args.make_animation = False
        args.show_animation = False
        args.init_state_range = "[[2.5, 3.0], [-0.25, 0.25]]"
        args.state_feedback = True
        args.boundaries = "lp"
        args.system = "double_integrator"
        args.t_max = 5
        args.estimate_runtime = True

        expts = [
            {
                "partitioner": "UnGuided",
                "propagator": "CROWN",
            },
            {
                "partitioner": "SimGuided",
                "propagator": "CROWN",
            },
            {
                "partitioner": "GreedySimGuided",
                "propagator": "CROWN",
            },
        ]

        df = pd.DataFrame()

        for expt in expts:
            for key, value in expt.items():
                setattr(args, key, value)
            stats, info = ex.main(args)

            for i, runtime in enumerate(stats["runtimes"]):
                df = df.append(
                    {
                        **expt,
                        "run": i,
                        "runtime": runtime,
                        "final_step_error": stats["final_step_errors"][i],
                        "avg_error": stats["avg_errors"][i],
                        "output_constraint": stats["output_constraints"][i],
                    },
                    ignore_index=True,
                )
        df.to_pickle(self.filename.format(dt=dt))

    def plot(self):
        raise NotImplementedError


class CompareRuntimeVsErrorTable(Experiment):
    def __init__(self):
        self.filename = results_dir + "runtime_vs_error_{dt}_table.pkl"
        Experiment.__init__(self)

    def run(self):
        dt = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        parser = ex.setup_parser()
        args = parser.parse_args()

        args.save_plot = False
        args.show_plot = False
        args.make_animation = False
        args.show_animation = False
        args.init_state_range = "[[2.5, 3.0], [-0.25, 0.25]]"
        args.state_feedback = True
        args.boundaries = "lp"
        args.system = "double_integrator"
        args.t_max = 5
        args.estimate_runtime = True

        expts = [
            # {
            #     'partitioner': 'None',
            #     'propagator': 'SeparableCROWN',
            # },
            # {
            #     'partitioner': 'None',
            #     'propagator': 'SeparableSGIBP',
            # },
            {
                "partitioner": "None",
                "propagator": "CROWN",
            },
            {
                "partitioner": "None",
                "propagator": "OVERT",
            },
            # {
            #     'partitioner': 'Uniform',
            #     'num_partitions': "[4, 4]",
            #     'propagator': 'CROWN',
            # },
            # {
            #     'partitioner': 'None',
            #     'propagator': 'SDP',
            #     'cvxpy_solver': 'MOSEK',
            # },
            # {
            #     'partitioner': 'Uniform',
            #     'num_partitions': "[4, 4]",
            #     'propagator': 'SDP',
            #     'cvxpy_solver': 'MOSEK',
            # },
        ]

        df = pd.DataFrame()

        for expt in expts:
            for key, value in expt.items():
                setattr(args, key, value)
            stats, info = ex.main(args)

            for i, runtime in enumerate(stats["runtimes"]):
                df = df.append(
                    {
                        **expt,
                        "run": i,
                        "runtime": runtime,
                        "final_step_error": stats["final_step_errors"][i],
                        "avg_error": stats["avg_errors"][i],
                        "output_constraint": stats["output_constraints"][i],
                        "all_errors": stats["all_errors"][i],
                    },
                    ignore_index=True,
                )
        df.to_pickle(self.filename.format(dt=dt))

    def grab_latest_groups(self):
        # Grab latest file as pandas dataframe
        list_of_files = glob.glob(self.filename.format(dt="*"))
        latest_filename = max(list_of_files, key=os.path.getctime)
        df = pd.read_pickle(latest_filename)

        # df will have every trial, so group by which prop/part was used
        groupby = ["propagator", "partitioner"]
        grouped = df.groupby(groupby)
        return grouped, latest_filename

    def plot(self):
        grouped, filename = self.grab_latest_groups()

        # Setup table columns
        rows = []
        rows.append(["Algorithm", "Runtime [s]", "Error"])

        tuples = []
        tuples += [("SeparableCROWN", "None"), ("SeparableSGIBP", "None")]
        tuples += [
            (prop, part)
            for part in ["None", "Uniform"]
            for prop in ["SDP", "CROWN", "OVERT"]
        ]

        # Go through each combination of prop/part we want in the table
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue

            name = self.info[prop_part_tuple]["name"]

            mean_runtime = group["runtime"].mean()
            std_runtime = group["runtime"].std()
            runtime_str = "${:.3f} \pm {:.3f}$".format(
                mean_runtime, std_runtime
            )

            final_step_error = group["final_step_error"].mean()

            # Add the entries to the table for that prop/part
            row = []
            row.append(name)
            row.append(runtime_str)
            row.append(round(final_step_error))

            rows.append(row)

        # print as a human-readable table and as a latex table
        print(tabulate(rows, headers="firstrow"))
        print()
        print(tabulate(rows, headers="firstrow", tablefmt="latex_raw"))

    def plot_error_vs_timestep(self):
        grouped, filename = self.grab_latest_groups()

        fig, ax = plt.subplots(1, 1)

        # Go through each combination of prop/part we want in the table
        for propagator in ["SDP", "CROWN", "OVERT"]:
            for partitioner in ["None", "Uniform"]:
                prop_part_tuple = (propagator, partitioner)
                try:
                    group = grouped.get_group(prop_part_tuple)
                except KeyError:
                    continue

                all_errors = group["all_errors"].iloc[0]
                t_max = all_errors.shape[0]
                label = self.info[prop_part_tuple]["name"]

                # replace citation with the ref number in this plot
                label = label.replace("~\\cite{hu2020reach}", " [22]")

                plt.plot(
                    np.arange(1, t_max + 1),
                    all_errors,
                    color=self.info[prop_part_tuple]["color"],
                    ls=self.info[prop_part_tuple]["ls"],
                    label=label,
                )
        plt.legend()

        ax.set_yscale("log")
        plt.xlabel("Time Steps")
        plt.ylabel("Approximation Error")
        plt.tight_layout()

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace("table", "timestep").replace(
            "pkl", "png"
        )
        plt.savefig(fig_filename)

        # plt.show()

    def plot_reachable_sets(self):
        grouped, filename = self.grab_latest_groups()

        dyn = dynamics.DoubleIntegrator()
        controller = load_controller(system=dyn.__class__.__name__)

        init_state_range = np.array(
            [  # (num_inputs, 2)
                [2.5, 3.0],  # x0min, x0max
                [-0.25, 0.25],  # x1min, x1max
            ]
        )

        partitioner_hyperparams = {
            "type": "None",
        }
        propagator_hyperparams = {
            "type": "CROWN",
            "input_shape": init_state_range.shape[:-1],
        }

        # Set up analyzer (+ parititoner + propagator)
        analyzer = analyzers.ClosedLoopAnalyzer(controller, dyn)
        analyzer.partitioner = partitioner_hyperparams
        analyzer.propagator = propagator_hyperparams

        input_constraint = constraints.LpConstraint(
            range=init_state_range, p=np.inf
        )

        inputs_to_highlight = [
            {"dim": [0], "name": "$\mathbf{x}_0$"},
            {"dim": [1], "name": "$\mathbf{x}_1$"},
        ]

        t_max = 5

        analyzer.partitioner.setup_visualization(
            input_constraint,
            t_max,
            analyzer.propagator,
            show_samples=True,
            inputs_to_highlight=inputs_to_highlight,
            aspect="auto",
            initial_set_color=analyzer.initial_set_color,
            initial_set_zorder=analyzer.initial_set_zorder,
            sample_zorder=analyzer.sample_zorder,
        )

        analyzer.partitioner.linewidth = 1

        # Go through each combination of prop/part we want in the table
        for propagator in ["SDP", "CROWN", "OVERT"]:
            for partitioner in ["None", "Uniform"]:
                prop_part_tuple = (propagator, partitioner)
                try:
                    group = grouped.get_group(prop_part_tuple)
                except KeyError:
                    continue

                output_constraint = group["output_constraint"].iloc[0]

                analyzer.partitioner.visualize(
                    [],
                    [],
                    output_constraint,
                    None,
                    reachable_set_color=self.info[prop_part_tuple]["color"],
                    reachable_set_ls=self.info[prop_part_tuple]["ls"],
                    reachable_set_zorder=analyzer.reachable_set_zorder,
                )

                analyzer.partitioner.default_patches = (
                    analyzer.partitioner.animate_axes.patches.copy()
                )
                analyzer.partitioner.default_lines = (
                    analyzer.partitioner.animate_axes.lines.copy()
                )

        # Add shaded regions for verification
        goal_arr = np.array(
            [
                [-0.5, 0.5],
                [-0.25, 0.25],
            ]
        )
        dims = analyzer.partitioner.input_dims
        color = "None"
        fc_color = "lightblue"
        linewidth = 1
        ls = "-"
        rect = constraints.make_rect_from_arr(
            goal_arr, dims, color, linewidth, fc_color, ls, zorder=0
        )
        analyzer.partitioner.animate_axes.add_patch(rect)

        avoid_arr = np.array(
            [
                analyzer.partitioner.animate_axes.get_xlim(),
                [0.35, analyzer.partitioner.animate_axes.get_ylim()[1]],
            ]
        )
        dims = analyzer.partitioner.input_dims
        color = "None"
        fc_color = "wheat"
        linewidth = 1
        ls = "-"
        rect = constraints.make_rect_from_arr(
            avoid_arr, dims, color, linewidth, fc_color, ls, zorder=0
        )
        analyzer.partitioner.animate_axes.add_patch(rect)

        plt.tight_layout()

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace("table", "reachable").replace(
            "pkl", "png"
        )
        plt.savefig(fig_filename)

        # plt.show()

    def plot_error_vs_runtime(self):
        grouped, filename = self.grab_latest_groups()

        # Go through each combination of prop/part we want in the table
        for propagator in ["CROWN", "CROWNNStep"]:
            for partitioner in ["None", "Uniform"]:
                prop_part_tuple = (propagator, partitioner)
                try:
                    group = grouped.get_group(prop_part_tuple)
                except KeyError:
                    continue

                import pdb

                pdb.set_trace()
                error = group["final_step_error"].iloc[0]
                label = self.info[prop_part_tuple]["name"]
                runtime = group["runtime"].mean()

                print(propagator, partitioner)
                print(error)
                print(runtime)
                print("--")


class CompareLPvsCF(Experiment):
    def __init__(self, system):
        self.system = system
        Experiment.__init__(self)

    def run(self):
        rows = []
        rows.append(["", "1", "4", "16"])

        propagator_names = {"CROWNLP": "L.P.", "CROWN": "C.F."}
        t_max = {"quadrotor": 2, "double_integrator": 5}
        partitions = {
            "quadrotor": ["[1,1,1,1,1,1]", "[2,2,1,1,1,1]", "[2,2,2,2,1,1]"],
            "double_integrator": ["[1,1]", "[2,2]", "[4,4]"],
        }

        parser = ex.setup_parser()

        for propagator in ["CROWNLP", "CROWN"]:
            row = [propagator_names[propagator]]
            for num_partitions in partitions[self.system]:
                args = parser.parse_args()
                args.partitioner = "Uniform"
                args.propagator = propagator
                args.system = self.system
                args.state_feedback = True
                args.t_max = t_max[self.system]
                args.num_partitions = num_partitions
                args.estimate_runtime = True

                stats, info = ex.main(args)

                mean_runtime = stats["runtimes"].mean()
                std_runtime = stats["runtimes"].std()
                runtime_str = "${:.3f} \pm {:.3f}$".format(
                    mean_runtime, std_runtime
                )
                row.append(runtime_str)
            rows.append(row)

        self.data = rows

    def plot(self):
        if hasattr(self, "data"):
            rows = self.data
        else:
            # Grab from specific pkl file
            raise NotImplementedError

        print(tabulate(rows, headers="firstrow"))
        print()
        print(tabulate(rows, headers="firstrow", tablefmt="latex_raw"))


class NxScalability(Experiment):
    def __init__(self, state_or_control="state"):
        self.filename = results_dir + "runtime_vs_num_{x_or_u}_{dt}_table.pkl"
        self.state_or_control = state_or_control
        Experiment.__init__(self)

    def run(self):
        dt = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        parser = ex.setup_parser()
        args = parser.parse_args()

        args.save_plot = False
        args.show_plot = False
        args.make_animation = False
        args.show_animation = False
        # args.init_state_range = "[[2.5, 3.0], [-0.25, 0.25]]"
        args.state_feedback = True
        args.boundaries = "lp"
        args.system = "unity"
        args.t_max = 5
        args.estimate_runtime = True

        expts = [
            # {
            #     'partitioner': 'None',
            #     'propagator': 'CROWN',
            # },
            {
                "partitioner": "Uniform",
                "propagator": "CROWN",
            },
            # {
            #     'partitioner': 'None',
            #     'propagator': 'CROWNNStep',
            # },
        ]

        nxs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]

        df = pd.DataFrame()

        for nx in nxs:
            for expt in expts:
                for key, value in expt.items():
                    setattr(args, key, value)

                print(
                    "Prop: {}, Part: {}, nx: {}".format(
                        expt["propagator"], expt["partitioner"], nx
                    )
                )

                args.num_partitions = "2"

                if nx > 10 and expt["partitioner"] == "Uniform":
                    continue

                if self.state_or_control == "state":
                    args.nx = nx
                elif self.state_or_control == "control":
                    args.nu = nx
                stats, info = ex.main(args)

                for i, runtime in enumerate(stats["runtimes"]):
                    df = df.append(
                        {
                            **expt,
                            "run": i,
                            "runtime": runtime,
                            "final_step_error": stats["final_step_errors"][i],
                            "avg_error": stats["avg_errors"][i],
                            "output_constraint": stats["output_constraints"][
                                i
                            ],
                            "all_errors": stats["all_errors"][i],
                            "nx": nx,
                        },
                        ignore_index=True,
                    )
        df.to_pickle(self.filename.format(x_or_u=self.state_or_control, dt=dt))

    def plot(self):
        # Grab latest file as pandas dataframe
        list_of_files = glob.glob(
            self.filename.format(x_or_u=self.state_or_control, dt="*")
        )
        latest_filename = max(list_of_files, key=os.path.getctime)
        df = pd.read_pickle(latest_filename)

        max_runtime_to_plot = 40.0

        groups = df.groupby(["propagator", "partitioner"])

        plt.clf()

        for (prop, part), df_ in groups:
            grp = df_.groupby(["nx"])
            # stat = 'final_step_error'
            stat = "runtime"
            runtime_mean_series = grp[stat].mean()
            runtime_std_series = grp[stat].std()

            color = self.info[(prop, part)]["color"]
            ls = self.info[(prop, part)]["ls"]
            label = self.info[(prop, part)]["name"]

            inds = runtime_mean_series < max_runtime_to_plot

            runtime_mean_series = runtime_mean_series[inds]
            runtime_std_series = runtime_std_series[inds]

            plt.plot(
                runtime_mean_series.index.to_numpy(),
                runtime_mean_series.to_numpy(),
                color=color,
                linestyle=ls,
                label=label,
            )
            plt.gca().fill_between(
                runtime_mean_series.index.to_numpy(),
                runtime_mean_series.to_numpy() - runtime_std_series.to_numpy(),
                runtime_mean_series.to_numpy(),
                alpha=0.2,
                color=color,
                linestyle=ls,
            )
            plt.gca().fill_between(
                runtime_mean_series.index.to_numpy(),
                runtime_mean_series.to_numpy(),
                runtime_mean_series.to_numpy() + runtime_std_series.to_numpy(),
                alpha=0.2,
                color=color,
                linestyle=ls,
            )

        if self.state_or_control == "state":
            plt.xlabel("Number of States, $n_x$")
        elif self.state_or_control == "control":
            plt.xlabel("Number of Control Inputs, $n_u$")
        # plt.ylabel(stat)
        plt.ylabel("Computation Time [s]")
        plt.legend(prop={"size": 14})
        plt.tight_layout()

        # Save plot with similar name to pkl file that contains data
        filename = latest_filename
        fig_filename = filename.replace(
            "table", "runtime_" + self.state_or_control
        ).replace("pkl", "png")
        plt.savefig(fig_filename)


if __name__ == "__main__":
    # Like Fig 3 in ICRA21 paper
    c = CompareRuntimeVsErrorTable()
    # c.run()
    # c.plot()  # 3A: table
    # c.plot_reachable_sets()  # 3B: overlay reachable sets
    # c.plot_error_vs_timestep()  # 3C: error vs timestep
    c.plot_error_vs_runtime()

    # c = CompareLPvsCF(system="double_integrator")
    # c.run()
    # c.plot()

    # c = CompareLPvsCF(system="quadrotor")
    # c.run()
    # c.plot()

    # See how runtime scales with number of states
    # c = NxScalability("state")
    # c.run()
    # c.plot()

    # See how runtime scales with number of control inputs
    # c = NxScalability("control")
    # c.run()
    # c.plot()

    # WIP...
    # c = CompareMultipleCombos()
    # c.run()
    # c.plot()
