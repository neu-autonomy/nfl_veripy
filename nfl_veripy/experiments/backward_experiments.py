import argparse
import datetime
import glob
import os

import example_backward as ex
import matplotlib.pyplot as plt
import nfl_veripy.analyzers as analyzers
import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import numpy as np
import pandas as pd
from nfl_veripy.utils.nn import load_controller
from tabulate import tabulate

results_dir = "{}/results/logs/".format(
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(results_dir, exist_ok=True)


class Experiment:
    def __init__(self):
        # self.info = {
        #     ('CROWN', False): {
        #         'name': 'BReach-LP',
        #         'color': 'tab:orange',
        #         'ls': '-',
        #     },
        #     ('CROWNNStep', False): {
        #         'name': 'ReBReach-LP',
        #         'color': 'tab:blue',
        #         'ls': '-',
        #     },
        #     ('CROWN', True): {
        #         'name': 'Hybrid',
        #         'color': 'navy',
        #         'ls': '--',
        #     },
        # }
        self.info = {
            "uniform": {
                "name": "Uniform",
                "color": "tab:orange",
                "ls": "-",
            },
            "guided": {
                "name": "Guided",
                "color": "tab:blue",
                "ls": "-",
            },
        }


class ErrorVsPartitions(Experiment):
    def __init__(self):
        self.filename = results_dir + "runtime_vs_error_{dt}_table.pkl"
        self.baseline_filename = (
            results_dir + "runtime_vs_error_{dt}_table_baseline.pkl"
        )
        Experiment.__init__(self)

    def run(self):
        dt = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        parser = ex.setup_parser()
        args = parser.parse_args()

        args.save_plot = False
        args.show_plot = False
        args.make_animation = False
        args.show_animation = False
        args.state_feedback = True
        args.boundaries = "lp"
        args.system = "double_integrator"
        args.t_max = 5
        args.estimate_runtime = True
        args.overapprox = True
        args.partitioner = "None"
        args.propagator = "CROWN"
        args.refined = True

        expts = [
            {
                "num_partitions": "[2,2]",
                "partition_heuristic": "uniform",
            },
            # # {
            # #     'num_partitions': "[3,2]",
            # #     'partition_heuristic': 'uniform',
            # # },
            # {
            #     'num_partitions': "[3,3]",
            #     'partition_heuristic': 'uniform',
            # },
            # # {
            # #     'num_partitions': "[4,3]",
            # #     'partition_heuristic': 'uniform',
            # # },
            # {
            #     'num_partitions': "[4,4]",
            #     'partition_heuristic': 'uniform',
            # },
            # # {
            # #     'num_partitions': "[5,4]",
            # #     'partition_heuristic': 'uniform',
            # # },
            # {
            #     'num_partitions': "[5,5]",
            #     'partition_heuristic': 'uniform',
            # },
            # # {
            # #     'num_partitions': "[6,5]",
            # #     'partition_heuristic': 'uniform',
            # # },
            # {
            #     'num_partitions': "[6,6]",
            #     'partition_heuristic': 'uniform',
            # },
            # # {
            # #     'num_partitions': "[7,6]",
            # #     'partition_heuristic': 'uniform',
            # # },
            # {
            #     'num_partitions': "[7,7]",
            #     'partition_heuristic': 'uniform',
            # },
            # # {
            # #     'num_partitions': "[8,7]",
            # #     'partition_heuristic': 'uniform',
            # # },
            # {
            #     'num_partitions': "[8,8]",
            #     'partition_heuristic': 'uniform',
            # },
            # {
            #     'num_partitions': "[10,10]",
            #     'partition_heuristic': 'uniform',
            # },
            # {
            #     'num_partitions': "[12,12]",
            #     'partition_heuristic': 'uniform',
            # },
            {
                "num_partitions": "4",
                "partition_heuristic": "guided",
            },
            # {
            #     'num_partitions': "5",
            #     'partition_heuristic': 'guided',
            # },
            # {
            #     'num_partitions': "7",
            #     'partition_heuristic': 'guided',
            # },
            {
                "num_partitions": "9",
                "partition_heuristic": "guided",
            },
            # {
            #     'num_partitions': "13",
            #     'partition_heuristic': 'guided',
            # },
            {
                "num_partitions": "16",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "25",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "36",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "49",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "64",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "81",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "100",
                "partition_heuristic": "guided",
            },
            {
                "num_partitions": "144",
                "partition_heuristic": "guided",
            },
        ]
        baselines = [
            {
                "num_partitions": "[25, 25]",
                "partition_heuristic": "uniform",
            },
        ]

        df = pd.DataFrame()
        df_b = pd.DataFrame()

        # Generate 5 random numbers between 10 and 30
        np.random.seed(1)
        rand_len = 1
        rand_list = np.random.uniform(low=0, high=5, size=(rand_len,))
        print(rand_list)
        # rand_list = [0,1]

        for expt in expts:
            avg_runtime_avg = 0
            avg_error_avg = 0
            for shift in rand_list:
                for key, value in expt.items():
                    setattr(args, key, value)
                setattr(
                    args,
                    "final_state_range",
                    "[[{},{}],[-0.25,0.25]]".format(4.5 + shift, 5.0 + shift),
                )
                stats, info = ex.main(args)

                avg_runtime_avg += stats["avg_runtime"] / rand_len
                avg_error_avg += stats["final_step_errors"][0] / rand_len

            # import pdb; pdb.set_trace()

            for i, runtime in enumerate(stats["runtimes"]):
                stats["final_step_errors"][i] = avg_error_avg
                stats["avg_runtime"] = avg_runtime_avg
                df = df.append(
                    {
                        **expt,
                        "run": i,
                        "runtime": runtime,
                        "final_step_error": stats["final_step_errors"][i],
                        "avg_error": stats["avg_errors"][i],
                        "output_constraint": stats["output_constraints"][i],
                        "all_errors": stats["all_errors"][i],
                        "avg_runtime": stats["avg_runtime"],
                    },
                    ignore_index=True,
                )
        df.to_pickle(self.filename.format(dt=dt))

        # for baseline in baselines:
        #     avg_runtime_avg = 0
        #     avg_error_avg = 0
        #     for shift in rand_list:
        #         for key, value in baseline.items():
        #             setattr(args, key, value)
        #         setattr(args,'final_state_range', '[[{},{}],[-0.25,0.25]]'.format(4.5+shift, 5.0+shift))
        #         stats, info = ex.main(args)

        #         avg_runtime_avg += stats['avg_runtime']/rand_len
        #         avg_error_avg += stats['final_step_errors'][0]/rand_len
        #         # import pdb; pdb.set_trace()

        #     for i, runtime in enumerate(stats['runtimes']):
        #         stats['final_step_errors'][i] = avg_error_avg
        #         stats['avg_runtime'] = avg_runtime_avg
        #         df_b = df_b.append({
        #             **baseline,
        #             'run': i,
        #             'runtime': runtime,
        #             'final_step_error': stats['final_step_errors'][i],
        #             'avg_error': stats['avg_errors'][i],
        #             'output_constraint': stats['output_constraints'][i],
        #             'all_errors': stats['all_errors'][i],
        #             'avg_runtime': stats['avg_runtime']
        #             }, ignore_index=True)
        # df_b.to_pickle(self.baseline_filename.format(dt=dt))

    def grab_latest_groups(self):
        # Grab latest file as pandas dataframe
        list_of_files = glob.glob(self.filename.format(dt="*"))
        latest_filename = max(list_of_files, key=os.path.getctime)
        df = pd.read_pickle(latest_filename)

        # df will have every trial, so group by which prop/part was used
        groupby = ["partition_heuristic"]
        grouped = df.groupby(groupby)

        list_of_files_b = glob.glob(self.baseline_filename.format(dt="*"))
        latest_filename_b = max(list_of_files_b, key=os.path.getctime)
        df_b = pd.read_pickle(latest_filename_b)

        # df will have every trial, so group by which prop/part was used
        grouped_b = df_b.groupby(groupby)

        return grouped, latest_filename, grouped_b

    def plot(self):
        grouped, filename = self.grab_latest_groups()
        # Setup table columns
        rows = []
        rows.append(["Algorithm", "Runtime [s]", "Final Step Error"])

        tuples = []
        tuples += [("CROWN", "None"), ("CROWNNStep", "None")]

        # Go through each combination of prop/part we want in the table
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue

            # import pdb; pdb.set_trace()

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
            row.append("{:.2f}".format(final_step_error))

            rows.append(row)

        # print as a human-readable table and as a latex table
        print(tabulate(rows, headers="firstrow"))
        print()
        print(tabulate(rows, headers="firstrow", tablefmt="latex_raw"))

    def plot_error_vs_timestep(self):
        grouped, filename, grouped_b = self.grab_latest_groups()

        fig, ax = plt.subplots(1, 1)

        # Go through each combination of prop/part we want in the table
        for partitioner in ["uniform", "guided"]:
            prop_part_tuple = partitioner
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue

            all_errors = group["final_step_error"]

            all_runtimes = group["avg_runtime"]
            label = self.info[prop_part_tuple]["name"]

            import pdb

            pdb.set_trace()
            # replace citation with the ref number in this plot
            # label = label.replace('~\\cite{hu2020reach}', ' [22]')

            plt.plot(
                all_runtimes,
                all_errors,
                color=self.info[prop_part_tuple]["color"],
                ls=self.info[prop_part_tuple]["ls"],
                label=label,
            )

        for partitioner in ["uniform"]:
            prop_part_tuple = partitioner
            try:
                group_b = grouped_b.get_group(prop_part_tuple)
            except KeyError:
                continue

            ref_errors = [
                group_b["final_step_error"][0],
                group_b["final_step_error"][0],
            ]
            all_runtimes = [0.5, 5]
            label = "lower bound"
            # replace citation with the ref number in this plot
            # label = label.replace('~\\cite{hu2020reach}', ' [22]')

            #     # import pdb; pdb.set_trace()
            plt.plot(
                all_runtimes,
                ref_errors,
                color="k",
                ls="--",
            )

        plt.legend()

        ax.set_yscale("log")
        plt.xlabel("Time (s)")
        plt.yticks([2, 3, 4, 5, 10, 50], [2, 3, 4, 5, 10, 50], fontsize=16)
        plt.ylabel("Approximation Error")
        plt.tight_layout()
        ax.grid(which="major", color="#CCCCCC", linewidth=0.8)
        # Show the minor grid as well. Style it in very light gray as a thin,
        # dotted line.
        ax.grid(which="minor", color="#CCCCCC", linestyle=":", linewidth=0.5)

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace("table", "timestep").replace(
            "pkl", "png"
        )
        plt.savefig(fig_filename)

        plt.show()


# class CompareMultipleCombos(Experiment):
#     def __init__(self):
#         self.filename = results_dir + 'alg_error_{dt}_table.pkl'
#         Experiment.__init__(self)

#     def run(self):
#         dt = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

#         parser = ex.setup_parser()
#         args = parser.parse_args()

#         args.save_plot = False
#         args.show_plot = False
#         args.make_animation = False
#         args.show_animation = False
#         args.init_state_range = "[[2.5, 3.0], [-0.25, 0.25]]"
#         args.state_feedback = True
#         args.boundaries = "lp"
#         args.system = "double_integrator"
#         args.t_max = 5
#         args.estimate_runtime = True

#         expts = [
#             {
#                 'partitioner': 'UnGuided',
#                 'propagator': 'CROWN',
#             },
#             {
#                 'partitioner': 'SimGuided',
#                 'propagator': 'CROWN',
#             },
#             {
#                 'partitioner': 'GreedySimGuided',
#                 'propagator': 'CROWN',
#             },
#         ]

#         df = pd.DataFrame()

#         for expt in expts:
#             for key, value in expt.items():
#                 setattr(args, key, value)
#             stats, info = ex.main(args)

#             for i, runtime in enumerate(stats['runtimes']):
#                 df = df.append({
#                     **expt,
#                     'run': i,
#                     'runtime': runtime,
#                     'final_step_error': stats['final_step_errors'][i],
#                     'avg_error': stats['avg_errors'][i],
#                     'output_constraint': stats['output_constraints'][i],
#                     }, ignore_index=True)
#         df.to_pickle(self.filename.format(dt=dt))

#     def plot(self):
#         raise NotImplementedError


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
        args.init_state_range = "[[4.5, 5.0], [-0.25, 0.25]]"
        args.state_feedback = True
        args.boundaries = "lp"
        args.system = "double_integrator"
        args.t_max = 5
        args.estimate_runtime = True
        args.overapprox = True
        args.partition_heuristic = "uniform"
        args.all_lps = True
        args.slow_cvxpy = True

        expts = [
            {
                "partitioner": "None",
                "num_partitions": "[4, 4]",
                "propagator": "CROWN",
                "refined": False,
            },
            {
                "partitioner": "None",
                "num_partitions": "[4, 4]",
                "propagator": "CROWNNStep",
                "refined": False,
            },
            {
                "partitioner": "None",
                "num_partitions": "[4, 4]",
                "propagator": "CROWN",
                "refined": True,
            },
        ]

        df = pd.DataFrame()

        for expt in expts:
            for key, value in expt.items():
                setattr(args, key, value)
            stats, info = ex.main(args)
            # import pdb; pdb.set_trace()

            for i, runtime in enumerate(stats["runtimes"]):
                df = df.append(
                    {
                        **expt,
                        "run": i,
                        "runtime": runtime,
                        "final_step_error": stats["final_step_errors"][i],
                        "avg_error": stats["avg_errors"][i],
                        "input_constraint": stats["input_constraints"][i],
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
        groupby = ["propagator", "refined"]
        grouped = df.groupby(groupby)
        return grouped, latest_filename

    def plot(self):
        grouped, filename = self.grab_latest_groups()

        # Setup table columns
        rows = []
        rows.append(["Algorithm", "Runtime [s]", "Final Step Error"])

        tuples = []
        tuples += [("CROWN", False), ("CROWN", True), ("CROWNNStep", False)]

        # Go through each combination of prop/part we want in the table
        for prop_part_tuple in tuples:
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue

            # import pdb; pdb.set_trace()

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
            row.append("{:.2f}".format(final_step_error))

            rows.append(row)

        # print as a human-readable table and as a latex table
        print(tabulate(rows, headers="firstrow"))
        print()
        print(tabulate(rows, headers="firstrow", tablefmt="latex_raw"))

    def plot_error_vs_timestep(self):
        grouped, filename = self.grab_latest_groups()

        fig, ax = plt.subplots(1, 1)

        tuples = [("CROWN", False), ("CROWNNStep", False), ("CROWN", True)]

        for prop_part_tuple in tuples:
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

        # Go through each combination of prop/part we want in the table
        # for propagator in ['CROWN', 'CROWNNStep']:
        #     for partitioner in ['None']:
        #         prop_part_tuple = (propagator, partitioner)
        #         try:
        #             group = grouped.get_group(prop_part_tuple)
        #         except KeyError:
        #             continue

        #         all_errors = group['all_errors'].iloc[0]
        #         t_max = all_errors.shape[0]
        #         label = self.info[prop_part_tuple]['name']

        #         # replace citation with the ref number in this plot
        #         label = label.replace('~\\cite{hu2020reach}', ' [22]')

        #         plt.plot(
        #             np.arange(1, t_max+1),
        #             all_errors,
        #             color=self.info[prop_part_tuple]['color'],
        #             ls=self.info[prop_part_tuple]['ls'],
        #             label=label,
        #         )
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
        controller = load_controller(system="DoubleIntegrator")

        final_state_range = np.array(
            [  # (num_inputs, 2)
                [4.5, 5.0],  # x0min, x0max
                [-0.25, 0.25],  # x1min, x1max
            ]
        )

        partitioner_hyperparams = {
            "type": "None",
        }
        propagator_hyperparams = {
            "type": "CROWN",
            "input_shape": final_state_range.shape[:-1],
        }

        # Set up analyzer (+ parititoner + propagator)
        analyzer = analyzers.ClosedLoopBackwardAnalyzer(controller, dyn)
        analyzer.partitioner = partitioner_hyperparams
        analyzer.propagator = propagator_hyperparams

        output_constraint = constraints.LpConstraint(
            range=final_state_range, p=np.inf
        )

        inputs_to_highlight = [
            {"dim": [0], "name": "$x$"},
            {"dim": [1], "name": "$\dot{x}$"},
        ]

        t_max = 5

        # analyzer.partitioner.setup_visualization(
        #     input_constraint,
        #     t_max,
        #     analyzer.propagator,
        #     show_samples=True,
        #     inputs_to_highlight=inputs_to_highlight,
        #     aspect="auto",
        #     initial_set_color=analyzer.initial_set_color,
        #     initial_set_zorder=analyzer.initial_set_zorder,
        #     # sample_zorder=analyzer.sample_zorder
        # )
        analyzer.partitioner.setup_visualization(
            output_constraint,
            t_max,
            analyzer.propagator,
            show_samples=False,
            # show_samples=show_samples,
            inputs_to_highlight=inputs_to_highlight,
            plot_lims=np.array([[-3.8, 5.64], [-0.64, 2.5]]),
            # aspect=False,
            initial_set_color="tab:red",
            # initial_set_zorder=self.estimated_backprojection_set_zorder,
            # extra_constraint = initial_constraint,
            # extra_set_color=self.initial_set_color,
            # extra_set_zorder=self.initial_set_zorder,
        )

        analyzer.partitioner.linewidth = 2

        # Go through each combination of prop/part we want in the table
        tuples = [("CROWN", False), ("CROWNNStep", False), ("CROWN", True)]

        for prop_part_tuple in tuples:
            # for propagator in ['SDP', 'CROWN']:
            #     for partitioner in ['None', 'Uniform']:
            # prop_part_tuple = (propagator, partitioner)
            # import pdb; pdb.set_trace()
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue

            input_constraint = [
                group["input_constraint"].iloc[i] for i in range(t_max)
            ]
            # import pdb; pdb.set_trace()
            for cons in input_constraint:
                analyzer.partitioner.visualize(
                    [],
                    [],
                    cons,
                    reachable_set_color=self.info[prop_part_tuple]["color"],
                    reachable_set_ls=self.info[prop_part_tuple]["ls"],
                    # reachable_set_zorder=analyzer.reachable_set_zorder
                )

                analyzer.partitioner.default_patches = (
                    analyzer.partitioner.animate_axes.patches.copy()
                )
                analyzer.partitioner.default_lines = (
                    analyzer.partitioner.animate_axes.lines.copy()
                )

                try:
                    # import pdb; pdb.set_trace()
                    analyzer.plot_true_backprojection_sets(
                        input_constraint[-1],
                        # backreachable_set,
                        output_constraint,
                        t_max=t_max,
                        color="darkgreen",
                        zorder=10,  # self.true_backprojection_set_zorder,
                        linestyle="-",
                        show_samples=False,
                    )
                except:
                    pass

        # Add shaded regions for verification
        # goal_arr = np.array([
        #     [-0.5, 0.5],
        #     [-0.25, 0.25],
        # ])
        # dims = analyzer.partitioner.input_dims
        # color = "None"
        # fc_color = "lightblue"
        # linewidth = 1
        # ls = '-'
        # rect = constraints.make_rect_from_arr(goal_arr, dims, color, linewidth, fc_color, ls, zorder=0)
        # analyzer.partitioner.animate_axes.add_patch(rect)

        # avoid_arr = np.array([
        #     analyzer.partitioner.animate_axes.get_xlim(),
        #     [0.35, analyzer.partitioner.animate_axes.get_ylim()[1]],
        # ])
        # dims = analyzer.partitioner.input_dims
        # color = "None"
        # fc_color = "wheat"
        # linewidth = 1
        # ls = '-'
        # rect = constraints.make_rect_from_arr(avoid_arr, dims, color, linewidth, fc_color, ls, zorder=0)
        # analyzer.partitioner.animate_axes.add_patch(rect)

        plt.tight_layout()
        # analyzer.partitioner.animate_axis.set_xlim(-3.8, 5.64)
        # analyzer.partitioner.animate_axis.set_ylim(-0.64, 2.5)

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace("table", "reachable").replace(
            "pkl", "png"
        )
        plt.savefig(fig_filename)

        plt.show()


class NNScalability(Experiment):
    def __init__(self):
        self.filename = results_dir + "runtime_vs_error_{dt}_table.pkl"
        self.baseline_filename = (
            results_dir + "runtime_vs_error_{dt}_table_baseline.pkl"
        )
        Experiment.__init__(self)

    def run(self):
        dt = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        parser = ex.setup_parser()
        args = parser.parse_args()

        args.save_plot = False
        args.show_plot = False
        args.make_animation = False
        args.show_animation = False
        args.state_feedback = True
        args.boundaries = "lp"
        args.system = "discrete_quadrotor"
        args.t_max = 6
        args.num_partitions = "750"
        args.estimate_runtime = True
        args.overapprox = True
        args.partitioner = "None"
        args.propagator = "CROWN"
        args.refined = False
        args.init_state_range = "[[-5.25,-4.75],[-.25,.25],[2.25,2.75],[0.95,0.99],[-0.01,0.01],[-0.01,0.01]]"
        args.final_state_range = (
            "[[-1,1],[-1,1],[1.5,3.5],[-1,1],[-1,1],[-1,1]]"
        )

        expts = [
            {
                "partition_heuristic": "guided",
                "controller": "discrete_quad_avoid_origin_maneuver_2",
            },
            {
                "partition_heuristic": "guided",
                "controller": "discrete_quad_avoid_origin_maneuver_40_40",
            },
            {
                "partition_heuristic": "guided",
                "controller": "discrete_quad_avoid_origin_maneuver_50_50",
            },
            {
                "partition_heuristic": "guided",
                "controller": "discrete_quad_avoid_origin_maneuver_60_60",
            },
            {
                "partition_heuristic": "guided",
                "controller": "discrete_quad_avoid_origin_maneuver_80_80",
            },
            {
                "partition_heuristic": "guided",
                "controller": "discrete_quad_avoid_origin_maneuver_128_128",
            },
            # {
            #     'partition_heuristic': 'guided',
            #     'controller': 'discrete_quad_avoid_origin_maneuver_128_128'
            # }
        ]

        df = pd.DataFrame()

        for expt in expts:
            controller_name = expt["controller"]
            if controller_name == "discrete_quad_avoid_origin_maneuver_2":
                num_nodes = 40
            else:
                import re

                num_nodes = sum(
                    [int(s) for s in re.findall(r"\d+", controller_name)]
                )
            for key, value in expt.items():
                setattr(args, key, value)
            stats, info = ex.main(args)

            avg_runtime = stats["avg_runtime"]
            avg_error = stats["final_step_errors"][0]

            # import pdb; pdb.set_trace()

            for i, runtime in enumerate(stats["runtimes"]):
                stats["final_step_errors"][i] = avg_error
                stats["avg_runtime"] = avg_runtime
                df = df.append(
                    {
                        **expt,
                        "run": i,
                        "runtime": runtime,
                        "final_step_error": stats["final_step_errors"][i],
                        "avg_error": stats["avg_errors"][i],
                        "all_errors": stats["all_errors"][i],
                        "avg_runtime": stats["avg_runtime"],
                        "num_nodes": num_nodes,
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
        groupby = ["controller"]
        grouped = df.groupby(groupby)

        return grouped, latest_filename

    # def plot(self):
    #     grouped, filename = self.grab_latest_groups()
    #     # Setup table columns
    #     rows = []
    #     rows.append(["Algorithm", "Runtime [s]", "Final Step Error"])

    #     tuples = []
    #     tuples += [('CROWN', 'None'), ('CROWNNStep', 'None')]

    #     # Go through each combination of prop/part we want in the table
    #     for prop_part_tuple in tuples:
    #         try:
    #             group = grouped.get_group(prop_part_tuple)
    #         except KeyError:
    #             continue

    #         # import pdb; pdb.set_trace()

    #         name = self.info[prop_part_tuple]['name']

    #         mean_runtime = group['runtime'].mean()
    #         std_runtime = group['runtime'].std()
    #         runtime_str = "${:.3f} \pm {:.3f}$".format(mean_runtime, std_runtime)

    #         final_step_error = group['final_step_error'].mean()

    #         # Add the entries to the table for that prop/part
    #         row = []
    #         row.append(name)
    #         row.append(runtime_str)
    #         row.append("{:.2f}".format(final_step_error))

    #         rows.append(row)

    #     # print as a human-readable table and as a latex table
    #     print(tabulate(rows, headers='firstrow'))
    #     print()
    #     print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))

    def time_vs_num_nodes(self):
        grouped, filename = self.grab_latest_groups()

        fig, ax = plt.subplots(1, 1)

        all_runtimes = []
        num_nodes = []

        # Go through each combination of prop/part we want in the table
        for controller in [
            "discrete_quad_avoid_origin_maneuver_2",
            "discrete_quad_avoid_origin_maneuver_40_40",
            "discrete_quad_avoid_origin_maneuver_50_50",
            "discrete_quad_avoid_origin_maneuver_60_60",
            "discrete_quad_avoid_origin_maneuver_80_80",
            "discrete_quad_avoid_origin_maneuver_100_100",
            "discrete_quad_avoid_origin_maneuver_128_128",
        ]:
            # import pdb; pdb.set_trace()
            prop_part_tuple = controller
            try:
                group = grouped.get_group(prop_part_tuple)
            except KeyError:
                continue

            all_errors = group["final_step_error"]
            num_nodes.append(np.mean(group["num_nodes"]))

            all_runtimes.append(np.mean(group["avg_runtime"]))

            # import pdb; pdb.set_trace()
            # replace citation with the ref number in this plot
            # label = label.replace('~\\cite{hu2020reach}', ' [22]')

        from matplotlib.patches import Rectangle

        ax.add_patch(Rectangle((30, 20), 70, 45, alpha=0.1, color="g"))
        # ax.add_patch(Rectangle((100, 20), 170, 45, alpha=0.1, color='r'))

        plt.plot(num_nodes, all_runtimes, color="tab:blue", marker="o")

        plt.plot(
            [100, 100],
            [20, 65],
            color="tab:red",
        )

        # ax.set_yscale('log')
        plt.xlabel("Number of Neurons")
        # plt.yticks([2, 3, 4, 5, 10, 50],[2, 3, 4, 5, 10, 50], fontsize=16)
        plt.ylabel("Computation Time (s)")
        plt.tight_layout()
        ax.grid(which="major", color="#CCCCCC", linewidth=0.8)
        ax.set_xlim((30, 270))
        ax.set_ylim((20, 65))
        # Show the minor grid as well. Style it in very light gray as a thin,
        # dotted line.
        ax.grid(which="minor", color="#CCCCCC", linestyle=":", linewidth=0.5)

        # Save plot with similar name to pkl file that contains data
        fig_filename = filename.replace("table", "timestep").replace(
            "pkl", "png"
        )
        plt.savefig(fig_filename)

        plt.show()


if __name__ == "__main__":
    # Like Fig 3 in ICRA21 paper
    # c = CompareRuntimeVsErrorTable()
    ###c = ErrorVsPartitions()
    # c.run()
    # c.plot()  # 3A: table
    # c.plot_reachable_sets()  # 3B: overlay reachable sets
    ###c.plot_error_vs_timestep()  # 3C: error vs timestep
    c = NNScalability()
    # c.run()
    c.time_vs_num_nodes()

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
