from importlib import reload
import partition
import partition.Partition
import partition.Analyzer
import partition.Propagator
from partition.xiang import model_xiang_2020_robot_arm
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

partitioner_dict = {
    "None": partition.Partition.NoPartitioner,
    "Uniform": partition.Partition.UniformPartitioner,
    "SimGuided": partition.Partition.SimGuidedPartitioner,
}
propagator_dict = {
    "IBP": partition.Propagator.IBPPropagator,
    "CROWN": partition.Propagator.CROWNPropagator,
}
def experiment():
    
    # Choose the model and input range
    torch_model = model_xiang_2020_robot_arm()
    input_range = np.array([ # (num_inputs, 2)
                      [np.pi/3, 2*np.pi/3], # x0min, x0max
                      [np.pi/3, 2*np.pi/3] # x1min, x1max
    ])
    
    # Select which algorithms and hyperparameters to evaluate
    partitioners = ["Uniform", "SimGuided"]
    propagators = ["IBP", "CROWN"]
    partitioner_hyperparams_to_use = {
        "Uniform":
            {
                "num_partitions": [1,2,4,8,16,32]
            },
        "SimGuided":
            {
                "tolerance_eps": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                "num_simulations": [1000]
            }
    }

    # Auto-run combinations of algorithms & hyperparams, log results to pandas dataframe
    df = pd.DataFrame()
    analyzer = partition.Analyzer.Analyzer(torch_model)
    for partitioner, propagator in itertools.product(partitioners, propagators):
        partitioner_keys = list(partitioner_hyperparams_to_use[partitioner].keys())
        partitioner_hyperparams = {}
        for partitioner_vals in itertools.product(*list(partitioner_hyperparams_to_use[partitioner].values())):
            for partitioner_i in range(len(partitioner_keys)):
                partitioner_hyperparams[partitioner_keys[partitioner_i]] = partitioner_vals[partitioner_i]
            propagator_hyperparams = {}
            data_row = run_and_add_row(analyzer, input_range, partitioner, propagator, partitioner_hyperparams, propagator_hyperparams)
            df = df.append(data_row, ignore_index=True)
    
    # Also record the "exact" bounds (via sampling) in the same dataframe
    output_range_exact = analyzer.get_exact_output_range(input_range)
    df = df.append({
        "output_range_estimate": output_range_exact,
        "input_range": input_range,
        "propagator": "EXACT",
        "partitioner": "EXACT",
    }, ignore_index=True)

    return df

def run_and_add_row(analyzer, input_range, partitioner, propagator, partitioner_hyperparams, propagator_hyperparams):
    print("Partitioner: {}, Propagator: {}, {}, {}".format(partitioner, propagator, partitioner_hyperparams, propagator_hyperparams))
    analyzer.partitioner = partitioner_dict[partitioner](**partitioner_hyperparams)
    analyzer.propagator = propagator_dict[propagator](**propagator_hyperparams)
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    # analyzer.visualize(input_range, output_range, **analyzer_info)

    stats = {
        "output_range_estimate": output_range,
        "input_range": input_range,
        "propagator": type(analyzer.propagator).__name__,
        "partitioner": type(analyzer.partitioner).__name__,
    }
    data_row = {**stats, **analyzer_info}
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

def plot(df):
    output_range_exact = get_exact_output_range(df)
    for partitioner in df["partitioner"].unique():
        for propagator in df["propagator"].unique():
            if propagator == "EXACT" or partitioner == "EXACT": continue
            df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
            plt.semilogx(df_["num_partitions"].values, df_["output_area_error"], label=propagator+'-'+partitioner)
    
    plt.xlabel('Num Partitions')
    plt.ylabel('Approximation Error')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()

def plot(df):
    output_range_exact = get_exact_output_range(df)
    for partitioner in df["partitioner"].unique():
        for propagator in df["propagator"].unique():
            if propagator == "EXACT" or partitioner == "EXACT": continue
            df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
            plt.semilogx(df_["num_propagator_calls"].values, df_["output_area_error"], '-'+algs[partitioner]["marker"], label=propagator+'-'+partitioner)
    
    plt.xlabel('Num Propagtor Calls')
    plt.ylabel('Approximation Error')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()

def plot2(df):
    output_range_exact = get_exact_output_range(df)
    for partitioner in df["partitioner"].unique():
        for propagator in df["propagator"].unique():
            if propagator == "EXACT" or partitioner == "EXACT": continue
            df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
            plt.semilogx(df_["num_partitions"].values, df_["output_area_error"],'-'+algs[partitioner]["marker"], label=propagator+'-'+partitioner)
    
    plt.xlabel('Num Partitions')
    plt.ylabel('Approximation Error')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()

algs ={
    "UniformPartitioner": {
        "marker": "x",
        "name": "Uniform"
    },
    "SimGuidedPartitioner": {
        "marker": "o",
        "name": "SimGuided"
    }
}


if __name__ == '__main__':
    df = experiment()
    add_approx_error_to_df(df)
    plot(df)
    plot2(df)

