import numpy as np
import tensorflow as tf
import torch
import os
from nn_closed_loop.utils.nn import create_model
from crown_ibp.conversions.keras2torch import keras2torch

model_name = 'full_mlp_supervised'
dir_path = os.path.dirname(os.path.realpath(__file__))
NNET_MODEL_PATH = dir_path + '/' + model_name + '.nnet'
KERAS_MODEL_PATH = dir_path + '/' + model_name + '.h5'

def main():
    f = open(NNET_MODEL_PATH, "r")
    f.readline()
    f.readline()
    input_string = f.readline()
    layer_sizes = np.array(input_string[0:-2].split(','), dtype=int)
    neurons_per_layer = np.array(layer_sizes[1:-1], dtype=int)
    input_shape = (int(layer_sizes[0]),)
    output_shape = (int(layer_sizes[-1]),)
    model = create_model(neurons_per_layer, input_shape=input_shape, output_shape=output_shape)

    for i in range(5):
        f.readline()
    

    layer_weights = []
    for i in range(len(layer_sizes)-1):
        raw_line = f.readline()
        layer_weight = np.array(raw_line[0:-2].split(','), dtype=float)
        for j in range(layer_sizes[i+1]-1):
            raw_line = f.readline()
            row = np.array(raw_line[0:-2].split(','), dtype=float)
            layer_weight = np.vstack((layer_weight, row))

        raw_line = f.readline()
        layer_bias = np.array(raw_line[0:-2].split(','), dtype=float)
        for j in range(layer_sizes[i+1]-1):
            raw_line = f.readline()
            row = np.array(raw_line[0:-2].split(','), dtype=float)
            layer_bias = np.hstack((layer_bias, row))

        # import pdb; pdb.set_trace()

        # print("weights: {}".format(model.layers[i].get_weights()[0].shape))
        # print("bias: {}".format(model.layers[i].get_weights()[1].shape))
        model.layers[i].set_weights([layer_weight.T, layer_bias])


    model.compile(optimizer="rmsprop", loss="mse")
    model.save(KERAS_MODEL_PATH)
    # import pdb; pdb.set_trace()

        # layer_weights.append((layer_weight, layer_bias))

        
    

    
    print('haha')

def build_model_structure(layer_sizes):
    weight_matrices_shapes = []

    for i in range(len(layer_sizes)-1):
        weight_shape = (layer_sizes[i+1], layer_sizes[i])
        weight_matrices_shapes.append(weight_shape)

    
    
    return weight_matrices_shapes


if __name__ == "__main__":
    main()