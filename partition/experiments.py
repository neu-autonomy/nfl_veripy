from importlib import reload
import partition
import partition.Partitioner
import partition.Analyzer
import partition.Propagator
from partition.models import model_xiang_2020_robot_arm, model_gh1, model_gh2, random_model, lstm
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import os
import glob
import time
import math

save_dir = "{}/results/experiments".format(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(save_dir, exist_ok=True)
img_save_dir = save_dir+"/imgs/"
os.makedirs(img_save_dir, exist_ok=True)

experiments = [
  #  {
    #    'neurons': (2,100,2),
    #    'activation': 'relu',
     #   'seeds': range(10),
    #    'name': "Small NN",
  #  },
   # {
      #  'neurons': (2,10,20,50,20,10,2),
     #   'neurons': (2, 100, 100, 100, 100, 100, 100, 2),

    #    'activation': 'relu',
    #    'seeds': range(10),
    #    'name': "Deep NN",
   # },
    # {
    #     'model_fn': cnn_2layer,
    #     'model_args': {
    #         'in_ch': 1,
    #         'in_dim': 4,
    #         'width': 4,
    #         'linear_size': 16,
    #         'out_dim': 2,
    #     },
    #     'seeds': range(10),
    #     'name': "CNN",
    # },
    # {
    #     'model_fn': lstm,
    #     'model_args': {
    #         'hidden_size': 64, 
    #         'num_classes': 2, 
    #         'input_size': 64,
    #         'num_slices': 8,
    #     },
    #     'input_shape': (8,8),
    #     'lstm': True,
    #     'seeds': range(10),
    #     'name': "LSTM",
    # },
    {
        'model_fn': random_model,
        'model_args': {
            'neurons': (2,100,2),
            'activation': 'relu',
        },
        'seeds': range(10),
        'name': "Small NN",
    },
    # {
    #     'model_fn': random_model,
    #     'model_args': {
    #         'neurons': (2,100,100,100,100,100,100,2),
    #         'activation': 'relu',
    #     },
    #     'seeds': range(10),
    #     'name': "Deep NN",
    # },
    # {
    #     'model_fn': random_model,
    #     'model_args': {
    #         'neurons': (4,100,100,10),
    #         'activation': 'relu',
    #     },
    #     'seeds': range(10),
    #     'name': "Larger Input/Output Dims",
    # },
  #  {
     #   'neurons': (2,5,10),
   #     'activation': 'relu',
   #     'seeds': range(10),
   #     'name': "Larger Output Dimension",
  #  },
    {
        'neurons': (4,100,100,10),
        'activation': 'relu',
        'seeds': range(10),
       'name': "Different Activation",
    },
]

def experiment_input_range(lstm=False, neurons=None, input_shape=None):
    if lstm:
        # input_shape = (8,8)
        input_range = np.zeros(input_shape+(2,))
        input_range[-1,0:2,1] = 1.
    else:
        # For random models
        input_range = np.zeros((neurons[0],2))
        input_range[:,1] = 1.
    return input_range

# Select which algorithms and hyperparameters to evaluate
partitioners = ["None","SimGuided", "GreedySimGuided"]#, "AdaptiveSimGuided"]
# propagators = ["IBP_LIRPA"]
# propagators = ["CROWN_LIRPA", "FastLin_LIRPA"]
propagators = ["IBP_LIRPA", "CROWN_LIRPA", "FastLin_LIRPA"]
# propagators = ["SDP"]
partitioner_hyperparams_to_use = {
    "None":
        {
            "interior_condition": ["lower_bnds", "linf", "convex_hull"],
        },
    "UnGuided":
        {
            "termination_condition_type": ["num_propagator_calls"],
            "termination_condition_value": [200],
            "num_simulations": [1000],
            "interior_condition": ["lower_bnds", "linf", "convex_hull"],
        },
    "SimGuided":
        {
            "termination_condition_type": ["time_budget"],

          #  "termination_condition_type": ["num_propagator_calls"],
            "termination_condition_value": [2],
            "num_simulations": [1000],
            "interior_condition": ["lower_bnds", "linf"],
        },
    "GreedySimGuided":
        {
            "termination_condition_type": ["num_propagator_calls"],

            "termination_condition_type": ["time_budget"],
    
            "termination_condition_value": [2],
            "num_simulations": [1000],
            "interior_condition": ["lower_bnds", "linf"],
        },
    "AdaptiveSimGuided":
        {
            "termination_condition_type": ["num_propagator_calls"],
            "termination_condition_type": ["time_budget"],

            "termination_condition_value": [2],
            "num_simulations": [1000],
            "interior_condition": ["lower_bnds", "linf", "convex_hull"],
        },
}

def collect_data_for_table():

    df = pd.DataFrame()

    for experiment in experiments:
        for seed in experiment['seeds']:
            model_fn = experiment['model_fn']
            model, model_info = model_fn(seed=seed, **experiment['model_args'])
            input_range = experiment_input_range(lstm=('lstm' in experiment and experiment['lstm']),
                neurons=experiment['model_args']['neurons'], input_shape=experiment.get('input_shape', None))
            df = run_experiment(model=model, model_info=model_info, df=df, save_df=False,
                partitioners=partitioners, propagators=propagators, partitioner_hyperparams_to_use=partitioner_hyperparams_to_use,
                input_range=input_range)

    # Save the df in the "results" dir (so you don't have to re-run the expt)
    current_datetime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    df.to_pickle("{}/{}.pkl".format(save_dir, current_datetime))
    return df

def run_experiment(model=None, model_info=None, df=None, save_df=True, input_range=None, partitioners=None, propagators=None, partitioner_hyperparams_to_use=None):
    
    if model is None or model_info is None:
        neurons = [10,5,2]
        model, model_info = random_model(activation='relu', neurons=neurons, seed=0)

    if input_range is None:
        # # For CNN
        # input_range = np.zeros((1, 4, 4)+(2,))
        # input_range[0,0,0,1] = 1.

        # # For LSTM
        # input_shape = (8,8)
        # input_range = np.zeros(input_shape+(2,))
        # input_range[-1,0:2,1] = 1.

        # For random models
        input_range = np.zeros((model_info['model_neurons'][0],2))
        input_range[:,1] = 1.
        input_range[0,1] = 1.
        input_range[1,1] = 1.
        # uniform_partitions = np.ones((neurons[0]), dtype=int)
        # uniform_partitions[0:2] = 10

        # model, model_info = random_model(activation='relu', neurons=[2,5,20,40,40,20,2], seed=0)
        # input_range = np.array([ # (num_inputs, 2)
        #                   [0., 1.], # x0min, x0max
        #                   [0., 1.] # x1min, x1max
        # ])

        # neurons = [3,5,2]
        # model, model_info = random_model(activation='relu', neurons=neurons, seed=0)
        # input_range = np.array([ # (num_inputs, 2)
        #                   [0., 1.], # x0min, x0max
        #                   [0., 1.], # x1min, x1max
        #                   [0., 1.] # x2min, x2max
        # ])

        # model, model_info = model_xiang_2020_robot_arm(activation="relu")
        # input_range = np.array([ # (num_inputs, 2)
        #                   [np.pi/3, 2*np.pi/3], # x0min, x0max
        #                   [np.pi/3, 2*np.pi/3] # x1min, x1max
        # ])
    

    if partitioners is None or propagators is None or partitioner_hyperparams_to_use is None:
        # Select which algorithms and hyperparameters to evaluate
        # partitioners = ["SimGuided", "GreedySimGuided", "UnGuided"]
        # partitioners = ["AdaptiveSimGuided", "SimGuided", "GreedySimGuided"]
        partitioners = ["None", "SimGuided", "GreedySimGuided"]
        # partitioners = ["UnGuided"]
        # propagators = ["SDP"]
        propagators = ["IBP_LIRPA", "CROWN_LIRPA", "FastLin_LIRPA"]
        partitioner_hyperparams_to_use = {
            "None":
                {
                    "interior_condition": ["linf"],
                },
            # "Uniform":
            #     {
            #         # "num_partitions": [1,2,4,8,16,32]
            #         "num_partitions": uniform_partitions,
            #         "interior_condition": ["convex_hull"],
            #     },
            "UnGuided":
                {
                    "termination_condition_type": ["num_propagator_calls"],
                    # "termination_condition_value": [1,2,4,8,16,32,64,128, 256, 512, 1024],
                    "termination_condition_value": [100],
                    # "termination_condition_type": ["input_cell_size"],
                    # "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                    "num_simulations": [1000],
                    "interior_condition": ["linf"],
                },
            "SimGuided":
                {
                    "termination_condition_type": ["num_propagator_calls"],
                    "termination_condition_value": [100],
                    # "termination_condition_value": [1,2,4,8,16,32,64,128, 256, 512, 1024],
                    # "termination_condition_value": [1,2,4,16,32,64,128,],
                    # "termination_condition_type": ["input_cell_size"],
                    # "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                    "num_simulations": [1000],
                    "interior_condition": ["linf"],
                },
            "GreedySimGuided":
                {
                    "termination_condition_type": ["num_propagator_calls"],
                    "termination_condition_value": [100],
                    # "termination_condition_value": [1,2,4,8,16,32,64,128, 256, 512, 1024],
                    # "termination_condition_value": [1,2,4,16,32,64,128,],

                    # "termination_condition_type": ["input_cell_size"],
                    # # "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                    # "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001],
                    # "termination_condition_value": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5],
                    "num_simulations": [1000],
                    "interior_condition": ["linf"],
                },
            "AdaptiveSimGuided":
                {
                    "termination_condition_type": ["num_propagator_calls"],
                    "termination_condition_value": [100],
                    # "termination_condition_value": [1,2,4,8,16,32,64,128, 256, 512, 1024],
                    # "termination_condition_value": [1,2,4,16,32,64,128,],
                    # "tolerance_eps": [1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                    #"tolerance_expanding_step": [0.001],
                    #"k_NN": [1],
                    "num_simulations": [1000],
                    "interior_condition": ["linf"],
                    #"tolerance_range": [0.05]
                },
        }

    # Auto-run combinations of algorithms & hyperparams, log results to pandas dataframe
    if df is None:
        df = pd.DataFrame()
    analyzer = partition.Analyzer.Analyzer(model)
    for partitioner, propagator in itertools.product(partitioners, propagators):
        partitioner_keys = list(partitioner_hyperparams_to_use[partitioner].keys())
        partitioner_hyperparams = {"type": partitioner}
        for partitioner_vals in itertools.product(*list(partitioner_hyperparams_to_use[partitioner].values())):
            for partitioner_i in range(len(partitioner_keys)):
                partitioner_hyperparams[partitioner_keys[partitioner_i]] = partitioner_vals[partitioner_i]
            propagator_hyperparams = {"type": propagator, "input_shape": input_range.shape[:-1]}
            if model_info["model_neurons"][-1] == 2 or partitioner_hyperparams["interior_condition"] is not "convex_hull":
                data_row = run_and_add_row(analyzer, input_range, partitioner_hyperparams, propagator_hyperparams, model_info)
                df = df.append(data_row, ignore_index=True)
    
    # Also record the "exact" bounds (via sampling) in the same dataframe
    output_range_exact = analyzer.get_exact_output_range(input_range)
    df = df.append({
        "output_range_estimate": output_range_exact,
        "input_range": input_range,
        "propagator": "EXACT",
        "partitioner": "EXACT",
    }, ignore_index=True)

    if save_df:
        # Save the df in the "results" dir (so you don't have to re-run the expt)
        current_datetime = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        df.to_pickle("{}/{}.pkl".format(save_dir, current_datetime))

    return df

def run_and_add_row(analyzer, input_range, partitioner_hyperparams, propagator_hyperparams, model_info={}):
    print("Partitioner: {},\n Propagator: {}".format(partitioner_hyperparams, propagator_hyperparams))
    np.random.seed(0)
    analyzer.partitioner = partitioner_hyperparams
    analyzer.propagator = propagator_hyperparams
    t_start = time.time()
    output_range, analyzer_info = analyzer.get_output_range(input_range)
    t_end = time.time()
    
    np.random.seed(0)
    if partitioner_hyperparams["interior_condition"] == "convex_hull":
        exact_hull = analyzer.get_exact_hull(input_range, N=int(1e5))
        error = analyzer.partitioner.get_error(exact_hull, analyzer_info["estimated_hull"])
    else:
        exact_output_range, _,_ = analyzer.partitioner.sample(input_range, analyzer.propagator, N=int(1e5))
        error = analyzer.partitioner.get_error(exact_output_range, output_range)
    print(error)
    # print(t_end-t_start)
    # print(analyzer_info["propagator_computation_time"])

    pars = '_'.join([str(key)+"_"+str(value) for key, value in sorted(partitioner_hyperparams.items(), key=lambda kv: kv[0]) if key not in ["make_animation", "show_animation", "type"]])
    pars2 = '_'.join([str(key)+"_"+str(value) for key, value in sorted(propagator_hyperparams.items(), key=lambda kv: kv[0]) if key not in ["input_shape", "type"]])

    # analyzer_info["save_name"] = img_save_dir+partitioner_hyperparams['type']+"_"+propagator_hyperparams['type']+"_"+pars+"_"+pars2+".png"
    # analyzer.visualize(input_range, output_range, show=False, show_legend=False, **analyzer_info)

    stats = {
        "computation_time": t_end - t_start,
        "propagator_computation_time": t_end - t_start,
        "output_range_estimate": output_range,
        "input_range": input_range,
        "propagator": type(analyzer.propagator).__name__,
        "partitioner": type(analyzer.partitioner).__name__,
        "error": error,
        # "neurons": ,
        # "activation": ,
    }
    analyzer_info.pop("exact_hull", None)
    analyzer_info.pop("estimated_hull", None)
    data_row = {**stats, **analyzer_info, **partitioner_hyperparams, **propagator_hyperparams, **model_info}
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
    plt.rcParams['font.size'] = '20'
    output_range_exact = get_exact_output_range(df)
    for partitioner in df["partitioner"].unique():
        for propagator in df["propagator"].unique():
            if propagator == "EXACT" or partitioner == "EXACT":
                continue
            if partitioner == "UniformPartitioner":
                continue
            df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
            if propagator == "IBPAutoLIRPAPropagator" and partitioner == "SimGuidedPartitioner":
                linestyle = '--'
            else:
                linestyle = '-'

            plt.loglog(df_[stat].values, df_["error"],
                marker=algs[partitioner]["marker"],
                ms=8,
                color=cm.get_cmap("tab20c")(4*algs[propagator]["color_ind"]+algs[partitioner]["color_ind"]),
                label=algs[partitioner]["name"]+'-'+algs[propagator]["name"],
                linestyle=linestyle)

            # if propagator == "SDPPropagator" and partitioner == "SimGuidedPartitioner":
            #     pt = (df_[stat].values[0], df_["error"].values[0])
            #     text = (pt[0], pt[1]-0.1)
            #     plt.gca().annotate("Vanilla SDP\n(Fazlyab '19)", xy=pt,  xycoords='data',
            #                 xytext=text, textcoords='data',
            #                 arrowprops=dict(facecolor='black', shrink=0.05),
            #                 horizontalalignment='center', verticalalignment='top',
            #                 )

            # if propagator == "CROWNAutoLIRPAPropagator" and partitioner == "SimGuidedPartitioner":
            #     pt = (df_[stat].values[0], df_["error"].values[0])
            #     text = (pt[0], pt[1]-0.3)
            #     plt.gca().annotate("Vanilla CROWN\n(Zhang '18)", xy=pt,  xycoords='data',
            #                 xytext=text, textcoords='data',
            #                 arrowprops=dict(facecolor='black', shrink=0.05),
            #                 horizontalalignment='center', verticalalignment='top',
            #                 )

            # if propagator == "IBPAutoLIRPAPropagator" and partitioner == "SimGuidedPartitioner":
            #     pt = (df_[stat].values[2], df_["error"].values[2])
            #     text = (pt[0], pt[1]+1.)
            #     pt = (df_[stat].values[0], df_["error"].values[0])
            #     string = "All Blue Circles:\n(Xiang '20)"
            #     plt.gca().annotate(string, xy=pt,  xycoords='data',
            #                 xytext=text, textcoords='data',
            #                 arrowprops=dict(facecolor='black', shrink=0.05),
            #                 horizontalalignment='center', verticalalignment='top',
            #                 )
            #     pt = (df_[stat].values[1], df_["error"].values[1])
            #     plt.gca().annotate(string, xy=pt,  xycoords='data',
            #                 xytext=text, textcoords='data',
            #                 arrowprops=dict(facecolor='black', shrink=0.05),
            #                 horizontalalignment='center', verticalalignment='top',
            #                 )
            #     pt = (df_[stat].values[2], df_["error"].values[2])
            #     plt.gca().annotate(string, xy=pt,  xycoords='data',
            #                 xytext=text, textcoords='data',
            #                 arrowprops=dict(facecolor='black', shrink=0.05),
            #                 horizontalalignment='center', verticalalignment='top',
            #                 )
            #     plt.gca().text(text[0]+0.8, text[1]-0.8, "...", fontsize=30)

    plt.xlabel(stats[stat]["name"])
    plt.ylabel('Approximation Error')
    # plt.xlabel(stats[stat]["name"], fontsize=36)
    # plt.xticks(fontsize=36)
    # plt.ylabel('Approximation Error', fontsize=36)
    # plt.yticks(fontsize=36)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #    ncol=3, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.show()

algs ={
    "NoPartitioner": {
        "name": "None",
    },
    "UniformPartitioner": {
        "marker": "x",
        "color_ind": 0,
        "name": "Uniform",
    },
    "UnGuidedPartitioner": {
        "marker": "+",
        "color_ind": 0,
        "name": "Unguided",
    },
    "SimGuidedPartitioner": {
        "marker": "o",
        "color_ind": 0,
        "name": "SG",
    },
    "GreedySimGuidedPartitioner": {
        "marker": "^",
        "color_ind": 1,
        "name": "GSG",
    },
    "AdaptiveSimGuidedPartitioner": {
        "marker": "*",
        "color_ind": 2,
        "name": "AGSG",
    },
    "IBPPropagator": {
        "color_ind": 0,
        "name": "IBP",
    },
    "CROWNPropagator": {
        "color_ind": 1,
        "name": "CROWN",
    },
    "IBPAutoLIRPAPropagator": {
        "color_ind": 0,
        "name": "IBP",
    },
    "CROWNAutoLIRPAPropagator": {
        "color_ind": 1,
        "name": "CROWN",
    },
    "FastLinAutoLIRPAPropagator": {
        "color_ind": 3,
        "name": "Fast-Lin",
    },
    "SDPPropagator": {
        "color_ind": 2,
        "name": "SDP",
    },
    "relu": {
        "name": "ReLU",
    }
}

stats = {
    "propagator_computation_time": {
        "name": "Computation Time (Propagator Only) [s]"
    },
    "num_partitions": {
        "name": "Number of Partitions"
    },
    "num_propagator_calls": {
        "name": "Number of Propagator Calls"
    },
}

names = {
    "lower_bnds": "Lower Bounds",
    "linf": "$\ell_\infty$-ball",
    "convex_hull": "Convex Hull",
}

citations = {
    "IBP": {
        "None": "\\cite{gowal2018effectiveness}",
        "SG": "\\cite{xiang2020reachable}",
    },
    "Fast-Lin": {
        "None": "\\cite{Weng_2018}",
    },
    "CROWN": {
        "None": "\\cite{zhang2018efficient}",
    },
    "SDP": {
        "None": "\\cite{fazlyab2019safety}",
    },
}

# def make_table(df):
#     partitioners = ["NoPartitioner", "UniformPartitioner", "SimGuidedPartitioner", "GreedySimGuidedPartitioner"]
#     propagators = ["IBPAutoLIRPAPropagator", "FastLinAutoLIRPAPropagator", "CROWNAutoLIRPAPropagator"]#, "SDPPropagator"]

#     neurons = df.model_neurons.iloc[0]
#     activation = df.model_activation.iloc[0]

#     print("\\begin{tabular}{c c| "+"c "*len(partitioners)+"}") 
#     print("\\hline \\multicolumn{"+str(2+len(partitioners))+"}{c}{Neurons: "+str(neurons)+" -- Activation: "+str(algs[activation]["name"])+"}\\\\ \\hline")
#     print("&& \\multicolumn{"+str(len(partitioners))+"}{c}{Partitioner} \\\\")
#     row = "&"
#     for partitioner in partitioners:
#         row += " & " + algs[partitioner]["name"]
#     print(row + " \\\\ \\hline")
#     print("\\multirow{"+str(len(propagators))+"}{*}{\\STAB{\\rotatebox[origin=c]{90}{Propagator}}}")
#     for propagator in propagators:
#         row = "& " + algs[propagator]["name"]
#         for partitioner in partitioners:
#             df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
#             stat = round(df_.error.to_numpy()[0], 3)
#             row += " & "
#             if partitioner == "NoPartitioner" or (propagator == "IBPAutoLIRPAPropagator" and partitioner in ["UniformPartitioner", "SimGuidedPartitioner"]):
#                 row += "\\cellcolor{Gray} "
#             row += str(stat)
#         print(row + " \\\\")
#     print("\\end{tabular}")

def table_single_model(df, partitioners, propagators, boundaries, neurons, name, activation):
    # print("\\multirow{"+str(len(propagators)*len(partitioners))+"}{*}{\\shortstack{"+name+" \\\\ "+str(neurons)+" \\\\ "+activation+"}} &")
    # print("\\multirow{"+str(len(propagators))+"}{*}{\\STAB{\\rotatebox[origin=c]{90}{Propagator}}}")
    for propagator in propagators:
        first = True
        for partitioner in partitioners:
            row = ""
            if first:
                row += "\\multirow{"+str(len(partitioners))+"}{*}{"+algs[propagator]["name"]+"} & "
                first = False
            else: row += " & "
            row += algs[partitioner]['name']
            if algs[propagator]['name'] in citations and algs[partitioner]['name'] in citations[algs[propagator]['name']]:
                row += "~"+citations[algs[propagator]['name']][algs[partitioner]['name']]
            for boundary in boundaries:
                df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator) & (df["interior_condition"] == boundary)]
                stat = df_.error.mean()
                # stat = round(stat, 3)
                if math.isnan(stat):
                    stat = "-"
                else:
                    stat = '{:.2E}'.format(stat).lower()
                    stat = '\\num{'+stat+'}'
                row += " & "
                # if partitioner == "NoPartitioner" or (propagator == "IBPAutoLIRPAPropagator" and partitioner in ["UniformPartitioner", "SimGuidedPartitioner"]):
                #     row += "\\cellcolor{Gray} "
                row += str(stat)
        # if first: row = row[1:]; first = False
            print(row + " \\\\")
        print("\\hline")

def make_big_table(df):
    partitioners = ["NoPartitioner", "SimGuidedPartitioner", "GreedySimGuidedPartitioner", "AdaptiveSimGuidedPartitioner"]

    propagators = ["IBPAutoLIRPAPropagator", "FastLinAutoLIRPAPropagator", "CROWNAutoLIRPAPropagator", "SDPPropagator"]
    boundaries = ["lower_bnds", "linf", "convex_hull"]

    neurons = df.model_neurons.iloc[0]
    activation = df.model_activation.iloc[0]

    for experiment in experiments:
        print("\\begin{tabular}{|c|c||"+(("c|"*len(boundaries))+"|")+"}")
        print("\\hline")
        print("\\multicolumn{2}{|c||}{Algorithm} & \\multicolumn{"+str(len(boundaries))+"}{c||}{Boundary Type} \\\\")
        row = "Prop. & Part."
        for boundary in boundaries:
            row += " & "+names[boundary]
        print(row + " \\\\ \\hline")
        # print("&&"+(" & \\multicolumn{"+str(len(partitioners))+"}{c|}{Partitioner}")*len(boundaries)+" \\\\")
        # row = "&&"
        # for boundary in boundaries:
        #     for partitioner in partitioners:
        #         row += " & " + algs[partitioner]["name"]
        # print(row + " \\\\ \\hline")

        try:
            neurons = experiment['model_args']['neurons']
            activation = experiment['model_args']['activation']
        except:
            neurons = (0,)
            activation = 'relu'
            # neurons = experiment['neurons']
            # activation = experiment['activation']

        # import pdb; pdb.set_trace()

        table_single_model(df,
        # table_single_model(df[(df["model_neurons"] == neurons) & (df["model_activation"] == activation)], 
            partitioners, propagators, boundaries, neurons, experiment['name'], activation)
        print("\\end{tabular}")
        print("\n\n --- \n\n")

# def make_table(df):
#     partitioners = ["NoPartitioner", "UniformPartitioner", "SimGuidedPartitioner", "GreedySimGuidedPartitioner"]
#     propagators = ["IBPAutoLIRPAPropagator", "FastLinAutoLIRPAPropagator", "CROWNAutoLIRPAPropagator"]#, "SDPPropagator"]

#     neurons = df.model_neurons.iloc[0]
#     activation = df.model_activation.iloc[0]

#     print("\\begin{tabular}{c c| "+"c "*len(partitioners)+"}") 
#     print("\\hline \\multicolumn{"+str(2+len(partitioners))+"}{c}{Neurons: "+str(neurons)+" -- Activation: "+str(algs[activation]["name"])+"}\\\\ \\hline")
#     print("&& \\multicolumn{"+str(len(partitioners))+"}{c}{Partitioner} \\\\")
#     row = "&"
#     for partitioner in partitioners:
#         row += " & " + algs[partitioner]["name"]
#     print(row + " \\\\ \\hline")
#     print("\\multirow{"+str(len(propagators))+"}{*}{\\STAB{\\rotatebox[origin=c]{90}{Propagator}}}")
#     for propagator in propagators:
#         row = "& " + algs[propagator]["name"]
#         for partitioner in partitioners:
#             df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator)]
#             stat = round(df_.error.to_numpy()[0], 3)
#             row += " & "
#             if partitioner == "NoPartitioner" or (propagator == "IBPAutoLIRPAPropagator" and partitioner in ["UniformPartitioner", "SimGuidedPartitioner"]):
#                 row += "\\cellcolor{Gray} "
#             row += str(stat)
#         print(row + " \\\\")
#     print("\\end{tabular}")

# def table_single_model(df, partitioners, propagators, boundaries, neurons, name, activation):
#     print("\\multirow{"+str(len(propagators))+"}{*}{\\shortstack{"+name+" \\\\ "+str(neurons)+" \\\\ "+activation+"}} &")
#     print("\\multirow{"+str(len(propagators))+"}{*}{\\STAB{\\rotatebox[origin=c]{90}{Propagator}}}")
#     first = True
#     for propagator in propagators:
#         row = "&& " + algs[propagator]["name"]
#         for boundary in boundaries:
#             for partitioner in partitioners:
#                 df_ = df[(df["partitioner"] == partitioner) & (df["propagator"] == propagator) & (df["interior_condition"] == boundary)]
#                 stat = df_.error.mean()
#                 # stat = round(stat, 3)
#                 if math.isnan(stat):
#                     stat = "-"
#                 else:
#                     stat = '{:.3E}'.format(stat).lower()
#                     stat = '\\SI{'+stat+'}'
#                 row += " & "
#                 if partitioner == "NoPartitioner" or (propagator == "IBPAutoLIRPAPropagator" and partitioner in ["UniformPartitioner", "SimGuidedPartitioner"]):
#                     row += "\\cellcolor{Gray} "
#                 row += str(stat)
#         if first: row = row[1:]; first = False
#         print(row + " \\\\")
#     print("\\hline")

# def make_big_table(df):
#     partitioners = ["NoPartitioner", "SimGuidedPartitioner", "GreedySimGuidedPartitioner"]
#     propagators = ["IBPAutoLIRPAPropagator", "FastLinAutoLIRPAPropagator", "CROWNAutoLIRPAPropagator", "SDPPropagator"]
#     boundaries = ["lower_bnds", "linf", "convex_hull"]

#     neurons = df.model_neurons.iloc[0]
#     activation = df.model_activation.iloc[0]

#     print("\\begin{tabular}{c|c c| "+(("c "*len(partitioners))+"|")*len(boundaries)+"}") 
#     row = "&&"
#     for boundary in boundaries:
#         row += " & \\multicolumn{"+str(len(partitioners))+"}{c|}{Boundary: "+names[boundary]+"}"
#     print(row + " \\\\")
#     print("&&"+(" & \\multicolumn{"+str(len(partitioners))+"}{c|}{Partitioner}")*len(boundaries)+" \\\\")
#     row = "&&"
#     for boundary in boundaries:
#         for partitioner in partitioners:
#             row += " & " + algs[partitioner]["name"]
#     print(row + " \\\\ \\hline")

#     for experiment in experiments:
#         table_single_model(df[(df["model_neurons"] == experiment['neurons']) & (df["model_activation"] == experiment["activation"])], 
#             partitioners, propagators, boundaries, experiment['neurons'], experiment['name'], experiment['activation'])

#     print("\\end{tabular}")

if __name__ == '__main__':

    # Run an experiment
    # df = run_experiment()

    # Make table
    df = collect_data_for_table()

    # If you want to plot w/o re-running the experiments, comment out the experiment line.
    if 'df' not in locals():
        # If you know the path
        latest_file = save_dir+"14-07-2020_18-56-40.pkl"

        # If you want to look up most recently made df
        list_of_files = glob.glob(save_dir+"/*.pkl")
        latest_file = max(list_of_files, key=os.path.getctime)

        df = pd.read_pickle(latest_file)

        # # Plot from corl 2020 submission
        # latest_file = save_dir+"/07-28-2020_16-00-59.pkl"
        # df = pd.read_pickle(latest_file)
        # latest_file = save_dir+"/07-28-2020_15-48-05.pkl"
        # df2 = pd.read_pickle(latest_file)
        # df = pd.concat([df, df2])
        
        # # Plot for acc 2020 table
        # latest_file = save_dir+"/09-12-2020_13-36-03.pkl"
        # df = pd.read_pickle(latest_file)
        # latest_file = save_dir+"/09-12-2020_14-02-27.pkl"
        # df2 = pd.read_pickle(latest_file)
        # df = pd.concat([df, df2])

    # add_approx_error_to_df(df)
    # plot(df, stat="num_partitions")
    # plot(df, stat="num_propagator_calls")
    # plot(df, stat="computation_time")
    # plot(df, stat="propagator_computation_time")
    print("\n --- \n")

    # make_table(df)
    make_big_table(df)

    print("\n --- \n")
