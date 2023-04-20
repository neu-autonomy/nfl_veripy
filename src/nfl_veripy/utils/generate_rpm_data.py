import os

import nfl_veripy.dynamics as dynamics
import numpy as np
import torch
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model, model_from_json

from crown_ibp.conversions.keras2torch import keras2torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def rpm_converter(
    system="DoubleIntegrator", model_name="default", model_type="torch"
):
    system = system.replace(
        "OutputFeedback", ""
    )  # remove OutputFeedback suffix if applicable
    path = "{}/../../models/{}/{}".format(dir_path, system, model_name)
    if system != "Taxinet":
        with open(path + "/model.json", "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(path + "/model.h5")
    else:
        model = load_model(path + "/model.h5")

    from nfl_veripy.utils.nn_bounds import BoundClosedLoopController

    from crown_ibp.bound_layers import BoundSequential

    torch_model = keras2torch(model, "torch_model")
    bounded_model = BoundClosedLoopController.convert(
        torch_model,
        bound_opts={"zero-lb": True},
        dynamics=dynamics.DoubleIntegrator(),
    )

    # torch_model = keras2torch(model, "torch_model")
    weights, biases = [], []
    for name, param in bounded_model.named_parameters():
        if "weight" in name:
            weights.append(param.detach().numpy())
        elif "bias" in name:
            biases.append(param.detach().numpy())

    # import pdb; pdb.set_trace()

    # weights, biases = [], []
    # for layer in model.layers:
    #     weights.append(layer.weights[0].numpy().T)
    #     biases.append(layer.weights[1].numpy())

    # import pdb; pdb.set_trace()

    state_dim = 2

    new_weights, new_biases = [], []
    new_input_layer_weight = np.vstack((weights[0], np.eye(state_dim)))
    new_input_layer_bias = np.hstack((biases[0], 100 * np.ones(state_dim)))
    new_weights.append(new_input_layer_weight)
    new_biases.append(new_input_layer_bias)

    for i in range(1, len(weights)):
        # import pdb; pdb.set_trace()
        top_augmented_row = np.hstack(
            (weights[i], np.zeros((weights[i].shape[0], state_dim)))
        )
        bottom_augmented_row = np.hstack(
            (np.zeros((state_dim, weights[i].shape[1])), np.eye(state_dim))
        )
        new_weight = np.vstack((top_augmented_row, bottom_augmented_row))
        new_weights.append(new_weight)

        if i < len(weights) - 1:
            new_bias = np.hstack((biases[i], np.zeros(state_dim)))
        else:
            new_bias = np.hstack((biases[i], -100 * np.ones(state_dim)))
        new_biases.append(new_bias)

    augmented_model = build_sequential_model(new_weights, new_biases)
    aug_torch_model = keras2torch(augmented_model, "converted_model")

    augmented_model_2 = build_sequential_model_2(new_weights, new_biases)
    aug_torch_model_2 = keras2torch(augmented_model_2, "converted_model")

    # for i in np.arange(-5, 5, 0.1):
    #     for j in np.arange(-5, 5, 0.1):
    #         test_input = torch.from_numpy(np.array([i, j], dtype='float32'))
    #         aug_torch_model = keras2torch(augmented_model, "converted_model")
    #         pred = torch_model(test_input)
    #         aug_pred = aug_torch_model(test_input)
    #         # import pdb; pdb.set_trace()

    #         if pred[0].detach().numpy()-aug_pred[0].detach().numpy() != 0 or (np.abs(aug_pred[1:].detach().numpy() - test_input.detach().numpy()) > 4.53676e-06).any():
    #             print('ERROR!!!')
    #             print("Problem input: {}".format(test_input))
    #             raise ValueError

    for i in np.arange(-5, 5, 0.1):
        for j in np.arange(-5, 5, 0.1):
            test_input = torch.from_numpy(np.array([i, j], dtype="float32"))
            pred = aug_torch_model(test_input)
            aug_pred = aug_torch_model_2(test_input)
            # import pdb; pdb.set_trace()

            if (
                np.abs(aug_pred.detach().numpy() - pred.detach().numpy())
                > 5.53676e-04
            ).any():
                print("ERROR!!!")
                print("Problem input: {}".format(test_input))
                raise ValueError

    convert_weights = []
    for name, param in aug_torch_model_2.named_parameters():
        convert_weights.append(param.detach().numpy())

    X_mean, X_std = np.array([0, 0]), np.array(
        [np.sqrt((6 + 6) ** 2 / 12), np.sqrt((6 + 6) ** 2 / 12)]
    )
    Y_mean, Y_std = np.array([4.75, 0]), np.array(
        [np.sqrt((5 - 4.5) ** 2 / 12), np.sqrt((0.25 + 0.25) ** 2 / 12)]
    )

    layer_sizes = [
        layer.weights[1].shape[0] for layer in augmented_model.layers
    ]

    # import pdb; pdb.set_trace()

    np.savez("weights.npz", *convert_weights)
    np.savez(
        "params.npz",
        X_mean=X_mean,
        X_std=X_std,
        Y_mean=Y_mean,
        Y_std=Y_std,
        layer_sizes=layer_sizes,
    )

    import shutil

    shutil.move(
        "weights.npz",
        "../../rpm2/Neural-Network-Reach/models/DoubleIntegrator/weights.npz",
    )
    shutil.move(
        "params.npz",
        "../../rpm2/Neural-Network-Reach/models/DoubleIntegrator/params.npz",
    )

    # num_layers = len(model["weights"])


# layer_sizes = vcat(size(model["weights"][1], 2), [length(vec(model["biases"][i])) for i in 1:num_layers])

# np.savez("weights.npz", *weights)
# num_layers = len(model["weights"])
# layer_sizes = np.vstack(size(model["weights"][1], 2), [length(vec(model["biases"][i])) for i in 1:num_layers])
# np.savez("params.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std, layer_sizes=layer_sizes)


def build_sequential_model(weights, biases, activation="relu"):
    input_shape = (2,)
    model = Sequential()
    model.add(
        Dense(
            weights[0].shape[0],
            input_shape=input_shape,
            activation=activation,
        )
    )
    # import pdb; pdb.set_trace()
    model.layers[0].set_weights([weights[0].T, biases[0]])

    for i in range(1, len(weights) - 1):
        model.add(Dense(weights[i].shape[0], activation=activation))
        model.layers[i].set_weights([weights[i].T, biases[i]])

    model.add(Dense(3))
    model.layers[-1].set_weights([weights[-1].T, biases[-1]])
    dt = 0.0625
    # dt = 1

    At = np.array([[1, dt], [0, 1]])
    bt = np.array([[0.5 * dt * dt], [dt]])
    model.add(Dense(2))
    system_dynamics_weight = np.hstack((bt, At))
    # A = np.array([[1, 1], [0, 1]])
    # B = np.array([[0.5], [1]])
    # model.add(Dense(2))
    # system_dynamics_weight = np.hstack((B, A))
    model.layers[-1].set_weights([system_dynamics_weight.T, np.zeros(2)])

    return model


def build_sequential_model_2(weights, biases, activation="relu"):
    input_shape = (2,)
    model = Sequential()
    model.add(
        Dense(
            weights[0].shape[0],
            input_shape=input_shape,
            activation=activation,
        )
    )
    # import pdb; pdb.set_trace()
    model.layers[0].set_weights([weights[0].T, biases[0]])

    for i in range(1, len(weights) - 1):
        model.add(Dense(weights[i].shape[0], activation=activation))
        model.layers[i].set_weights([weights[i].T, biases[i]])

    dt = 0.0625
    # dt = 1

    At = np.array([[1, dt], [0, 1]])
    bt = np.array([[0.5 * dt * dt], [dt]])
    model.add(Dense(2))
    # import pdb; pdb.set_trace()
    output_weights = np.hstack((np.outer(bt, weights[-1][0, 0]), At))
    output_bias = (bt * biases[-1][0]).flatten() + At @ biases[-1][-2:]
    model.layers[-1].set_weights([output_weights.T, output_bias])
    # A = np.array([[1, 1], [0, 1]])
    # B = np.array([[0.5], [1]])
    # model.add(Dense(2))
    # system_dynamics_weight = np.hstack((B, A))
    # model.layers[-1].set_weights([system_dynamics_weight.T, np.zeros(2)])

    return model


if __name__ == "__main__":
    rpm_converter()
