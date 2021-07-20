import nn_closed_loop.example as ex
import numpy as np
from tabulate import tabulate
import pandas as pd
import datetime
import os
import glob

results_dir = "{}/results/logs/".format(
    os.path.dirname(os.path.abspath(__file__))
)
os.makedirs(results_dir, exist_ok=True)


class Experiment:
    def __init__(self):
        return


class CompareMultipleCombos(Experiment):
    def __init__(self):
        self.filename = results_dir + 'alg_error_table_{dt}.pkl'

    def run(self):
        dt = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

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
                'partitioner': 'UnGuided',
                'propagator': 'CROWN',
            },
            {
                'partitioner': 'SimGuided',
                'propagator': 'CROWN',
            },
            {
                'partitioner': 'GreedySimGuided',
                'propagator': 'CROWN',
            },
        ]

        df = pd.DataFrame()

        for expt in expts:
            for key, value in expt.items():
                setattr(args, key, value)
            stats, info = ex.main(args)

            for i, runtime in enumerate(stats['runtimes']):
                df = df.append({
                    **expt,
                    'run': i,
                    'runtime': runtime,
                    'final_step_error': stats['final_step_errors'][i],
                    'avg_error': stats['avg_errors'][i],
                    'output_constraint': stats['output_constraints'][i],
                    }, ignore_index=True)
        df.to_pickle(self.filename.format(dt=dt))

    def plot(self):
        # Grab latest file as pandas dataframe
        list_of_files = glob.glob(self.filename.format(dt='*'))
        latest_filename = max(list_of_files, key=os.path.getctime)
        df = pd.read_pickle(latest_filename)

        # df will have every trial, so group by which prop/part was used
        groupby = ['propagator', 'partitioner']
        grouped = df.groupby(groupby)

        import pdb; pdb.set_trace()

        # human-readable names for each algorithm
        def to_name(prop_part_tuple):
            names = {
                ('CROWN', 'Uniform'): 'Reach-LP-Partition',
                ('CROWN', 'None'): 'Reach-LP',
                ('SDP', 'None'): 'Reach-SDP~\cite{hu2020reach}',
                ('SDP', 'Uniform'): 'Reach-SDP-Partition',
            }
            return names[prop_part_tuple]

        # Setup table columns
        rows = []
        rows.append(["Algorithm", "Runtime [s]", "Error"])

        # Go through each combination of prop/part we want in the table
        for propagator in ['SDP', 'CROWN']:
            for partitioner in ['None', 'Uniform']:
                prop_part_tuple = (propagator, partitioner)
                try:
                    group = grouped.get_group(prop_part_tuple)
                except KeyError:
                    continue

                name = to_name(prop_part_tuple)

                mean_runtime = group['runtime'].mean()
                std_runtime = group['runtime'].std()
                runtime_str = "${:.3f} \pm {:.3f}$".format(mean_runtime, std_runtime)

                final_step_error = group['final_step_error'].mean()

                # Add the entries to the table for that prop/part
                row = []
                row.append(name)
                row.append(runtime_str)
                row.append(final_step_error)

                rows.append(row)

        # print as a human-readable table and as a latex table
        print(tabulate(rows, headers='firstrow'))
        print()
        print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))




class CompareRuntimeVsErrorTable(Experiment):
    def __init__(self):
        self.filename = results_dir + 'runtime_vs_error_table_{dt}.pkl'

    def run(self):
        dt = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

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
                'partitioner': 'None',
                'propagator': 'CROWN',
            },
            {
                'partitioner': 'Uniform',
                'num_partitions': "[4, 4]",
                'propagator': 'CROWN',
            },
            {
                'partitioner': 'None',
                'propagator': 'SDP',
                'cvxpy_solver': 'MOSEK',
            },
            {
                'partitioner': 'Uniform',
                'num_partitions': "[4, 4]",
                'propagator': 'SDP',
                'cvxpy_solver': 'MOSEK',
            },
        ]

        df = pd.DataFrame()

        for expt in expts:
            for key, value in expt.items():
                setattr(args, key, value)
            stats, info = ex.main(args)

            for i, runtime in enumerate(stats['runtimes']):
                df = df.append({
                    **expt,
                    'run': i,
                    'runtime': runtime,
                    'final_step_error': stats['final_step_errors'][i],
                    'avg_error': stats['avg_errors'][i],
                    'output_constraint': stats['output_constraints'][i],
                    }, ignore_index=True)
        df.to_pickle(self.filename.format(dt=dt))

    def plot(self):
        # Grab latest file as pandas dataframe
        list_of_files = glob.glob(self.filename.format(dt='*'))
        latest_filename = max(list_of_files, key=os.path.getctime)
        df = pd.read_pickle(latest_filename)

        # df will have every trial, so group by which prop/part was used
        groupby = ['propagator', 'partitioner']
        grouped = df.groupby(groupby)

        # human-readable names for each algorithm
        def to_name(prop_part_tuple):
            names = {
                ('CROWN', 'Uniform'): 'Reach-LP-Partition',
                ('CROWN', 'None'): 'Reach-LP',
                ('SDP', 'None'): 'Reach-SDP~\cite{hu2020reach}',
                ('SDP', 'Uniform'): 'Reach-SDP-Partition',
            }
            return names[prop_part_tuple]

        # Setup table columns
        rows = []
        rows.append(["Algorithm", "Runtime [s]", "Error"])

        # Go through each combination of prop/part we want in the table
        for propagator in ['SDP', 'CROWN']:
            for partitioner in ['None', 'Uniform']:
                prop_part_tuple = (propagator, partitioner)
                try:
                    group = grouped.get_group(prop_part_tuple)
                except KeyError:
                    continue

                name = to_name(prop_part_tuple)

                mean_runtime = group['runtime'].mean()
                std_runtime = group['runtime'].std()
                runtime_str = "${:.3f} \pm {:.3f}$".format(mean_runtime, std_runtime)

                final_step_error = group['final_step_error'].mean()

                # Add the entries to the table for that prop/part
                row = []
                row.append(name)
                row.append(runtime_str)
                row.append(final_step_error)

                rows.append(row)

        # print as a human-readable table and as a latex table
        print(tabulate(rows, headers='firstrow'))
        print()
        print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))


class CompareLPvsCF(Experiment):
    def __init__(self, system):
        self.system = system

    def run(self):
        rows = []
        rows.append(["", "1", "4", "16"])

        propagator_names = {"CROWNLP": "L.P.", "CROWN": "C.F."}
        t_max = {"quadrotor": "2", "double_integrator": "2"}
        partitions = {
            'quadrotor': ["[1,1,1,1,1,1]", "[2,2,1,1,1,1]", "[2,2,2,2,1,1]"],
            'double_integrator': ["[1,1]", "[2,2]", "[4,4]"]
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

                mean_runtime = stats['runtimes'].mean()
                std_runtime = stats['runtimes'].std()
                runtime_str = "${:.3f} \pm {:.3f}$".format(mean_runtime, std_runtime)
                row.append(runtime_str)
            rows.append(row)
        
        self.data = rows

    def plot(self):
        if hasattr(self, "data"):
            rows = self.data
        else:
            # Grab from specific pkl file
            raise NotImplementedError

        print(tabulate(rows, headers='firstrow'))
        print()
        print(tabulate(rows, headers='firstrow', tablefmt='latex_raw'))


if __name__ == '__main__':
    
    # Like Fig 3A (table) in ICRA21 paper
    # c = CompareRuntimeVsErrorTable()
    # c.run()
    # c.plot()

    # c = CompareLPvsCF(system="double_integrator")
    # c.run()
    # c.plot()

    # c = CompareLPvsCF(system="quadrotor")
    # c.run()
    # c.plot()

    c = CompareMultipleCombos()
    c.run()
    c.plot()
