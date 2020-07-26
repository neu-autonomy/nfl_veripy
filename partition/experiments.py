from importlib import reload
import partition
import partition.Partitioner
import partition.Analyzer
import partition.Propagator
from partition.models import model_xiang_2020_robot_arm, model_gh1
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import os
import glob
import time

save_dir = "{}/results".format(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)

def experiment():
    
    # Choose the model and input range
    # torch_model = model_gh1()
    torch_model = model_xiang_2020_robot_arm()

    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3] # x1min, x1max
    ])
    
    # Select which algorithms and hyperparameters to evaluate
    partitioners = ["SimGuided", "GreedySimGuided"]
    propagators = ["CROWN_LIRPA"]
    partitioner_hyperparams_to_use = {
        "Uniform":
            {
                "num_partitions": [1,2,4,8,16,32]
            },
        "SimGuided":
            {
                "termination_condition_type": ["input_cell_size"],
                "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                "num_simulations": [1000]
            },
        "GreedySimGuided":
            {
                "termination_condition_type": ["input_cell_size"],
                # "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001],
                "num_simulations": [1000]
            },
        "AdaptiveSimGuided":
        #"BoundarySimGuided":

            {
                "tolerance_eps": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                #"tolerance_expanding_step": [0.001],
                #"k_NN": [1],
                "num_simulations": [1000],
                #"tolerance_range": [0.05]
            },
    }

    # Auto-run combinations of algorithms & hyperparams, log results to pandas dataframe
    df = pd.DataFrame()
    analyzer = partition.Analyzer.Analyzer(torch_model)
    for partitioner, propagator in itertools.product(partitioners, propagators):
        partitioner_keys = list(partitioner_hyperparams_to_use[partitioner].keys())
        partitioner_hyperparams = {"type": partitioner}
        for partitioner_vals in itertools.product(*list(partitioner_hyperparams_to_use[partitioner].values())):
            for partitioner_i in range(len(partitioner_keys)):
                partitioner_hyperparams[partitioner_keys[partitioner_i]] = partitioner_vals[partitioner_i]
            propagator_hyperparams = {"type": propagator, "input_shape": input_range.shape[:-1]}
            data_row = run_and_add_row(analyzer, input_range, partitioner_hyperparams, propagator_hyperparams)
            df = df.append(data_row, ignore_index=True)
    
    # Also record the "exact" bounds (via sampling) in the same dataframe
    output_range_exact = analyzer.get_exact_output_range(input_range)
    df = df.append({
        "output_range_estimate": output_range_exact,
        "input_range": input_range,
        "propagator": "EXACT",
        "partitioner": "EXACT",
    }, ignore_index=True)

    # Save the df in the "results" dir (so you don't have to re-run the expt)
    current_datetime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    df.to_pickle("{}/{}.pkl".format(save_dir, current_datetime))

    return df

def run_and_add_row(analyzer, input_range, partitioner_hyperparams, propagator_hyperparams):
    print("Partitioner: {},\n Propagator: {}".format(partitioner_hyperparams, propagator_hyperparams))
    np.random.seed(0)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    t_start = time.time()
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    t_end = time.time()
    pars = '_'.join([str(key)+"_"+str(value) for key, value in partitioner_hyperparams.items() if key not in ["make_animation", "show_animation", "type"]])
    pars2 = '_'.join([str(key)+"_"+str(value) for key, value in propagator_hyperparams.items() if key not in ["input_shape", "type"]])

    analyzer_info["save_name"] = save_dir+"/imgs/"+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+".png"
    analyzer.visualize(input_range, output_range, show=False, **analyzer_info)

    stats = {
        "computation_time": t_end - t_start,
        "output_range_estimate": output_range,
        "input_range": input_range,
        "propagator": type(analyzer.propagator).__name__,
        "partitioner": type(analyzer.partitioner).__name__,
    }
    data_row = {**stats, **analyzer_info, **partitioner_hyperparams, **propagator_hyperparams}
    return data_row

def add_approx_error_to_df(df):
    output_range_exact = get_exact_output_range(df)
    output_area_exact = np.product(output_range_exact[:,1] - output_range_exact[:,0])
    df['lower_bound_errors'] = ""
    df['output_area_estimate'] = ""
    df['output_area_error'] = ""
    for index, row in df.iterrows():
        lower_bnd_errors = output_range_exact[:,0] - row["output_range_estimate"][:,0]
        df.at[index, 'lower_bound_errors'] = lower_bnd_errors
        
        output_area_estimate = np.product(row["output_range_estimate"][:,1] - row["output_range_estimate"][:,0])
        df.at[index, 'output_area_estimate'] = output_area_estimate
        df.at[index, 'output_area_error'] = (output_area_estimate / output_area_exact) - 1.
        

def get_exact_output_range(df):
    row_exact = df[df["partitioner"] == "EXACT"]
    output_range_exact = row_exact["output_range_estimate"].values[0]
    return output_range_exact

def plot(df, stat):
    output_range_exact = get_exact_output_range(df)
    for partitioner in df["partitioner"].unique():
        for propagator in df["propagator"].unique():
            if propagator == "EXACT" or partitioner == "EXACT":
                continue
            df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
            plt.loglog(df_[stat].values, df_["output_area_error"],
                marker=algs[partitioner]["marker"],
                color=cm.get_cmap("tab20c")(4*algs[propagator]["color_ind"]+algs[partitioner]["color_ind"]),
                label=propagator+'-'+partitioner)

    plt.xlabel(stat)
    plt.ylabel('Approximation Error')
    plt.legend()
    plt.show()

algs ={
    "UniformPartitioner": {
        "marker": "x",
        "color_ind": 0,
        "name": "Uniform",
    },
    "SimGuidedPartitioner": {
        "marker": "o",
        "color_ind": 1,
        "name": "SimGuided",
    },
    "GreedySimGuidedPartitioner": {
        "marker": "^",
        "color_ind": 2,
        "name": "GreedySimGuided",
    },
    "AdaptiveSimGuidedPartitioner": {
        "marker": "*",
        "color_ind": 3,
        "name": "BoundarySimGuided",
    },
    "IBPPropagator": {
        "color_ind": 0,
        "name": "IBPPropagator",
    },
    "CROWNPropagator": {
        "color_ind": 1,
        "name": "CROWNPropagator",
    },
    "IBPAutoLIRPAPropagator": {
        "color_ind": 0,
        "name": "IBPPropagator",
    },
    "CROWNAutoLIRPAPropagator": {
        "color_ind": 1,
        "name": "CROWNPropagator",
    },
    "SDPPropagator": {
        "color_ind": 2,
        "name": "SDPPropagator",
    },
}

if __name__ == '__main__':

    # Run an experiment
    df = experiment()

    # If you want to plot w/o re-running the experiments, comment out the experiment line.
    if 'df' not in locals():
        # If you know the path
        latest_file = save_dir+"14-07-2020_18-56-40.pkl"

        # If you want to look up most recently made df
        list_of_files = glob.glob(save_dir+"/*.pkl")
        latest_file = max(list_of_files, key=os.path.getctime)

        df = pd.read_pickle(latest_file)

    add_approx_error_to_df(df)
    plot(df, stat="num_partitions")
    # plot(df, stat="num_propagator_calls")
    # plot(df, stat="computation_time")


