import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from crown_ibp.conversions.keras2torch import keras2torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def create_model(
    neurons_per_layer, input_shape, output_shape, activation="relu"
):
    model = Sequential()
    model.add(
        Dense(
            neurons_per_layer[0],
            input_shape=input_shape,
            activation=activation,
        )
    )
    for neurons in neurons_per_layer[1:]:
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(output_shape[0]))
    model.compile(optimizer="rmsprop", loss="mse")
    return model


def create_and_train_model(
    neurons_per_layer, xs, us, epochs=20, batch_size=32, verbose=0
):
    model = create_model(
        neurons_per_layer, input_shape=xs.shape[1:], output_shape=us.shape[1:]
    )
    model.fit(xs, us, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def save_model(model, name="model", dir=dir_path+"/../../models/double_integrator_debug/"):
    os.makedirs(dir, exist_ok=True)
    # serialize model to JSON
    model_json = model.to_json()
    with open(dir + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dir + name + ".h5")
    print("Saved model to disk")


def load_controller(name="double_integrator_mpc"):
    path = "{}/../../models/{}".format(dir_path, name)
    with open(path + "/model.json", "r") as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(path + "/model.h5")
    torch_model = keras2torch(model, "torch_model")
    return torch_model


def load_controller_unity(nx, nu):
    name = "unity"
    path = "{}/../../models/{}".format(dir_path, name)
    model_name = "/model_nx_{}_nu_{}".format(nx, nu)
    filename = path + model_name + ".json"
    try:
        with open(filename, "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(path + model_name + ".h5")
    except FileNotFoundError:
        model = create_model([5, 5], (nx,), (nu,))
        save_model(model, name=model_name, dir=path)
    torch_model = keras2torch(model, "torch_model")

    return torch_model


def plot_data(xs, us, system):

    import matplotlib.pyplot as plt

    if system == "double_integrator":
        xs = np.reshape(xs, (-1, 11, 2))
        us = np.reshape(us, (-1, 11, 1))
    elif system == "quadrotor":
        xs = np.reshape(xs, (-1, 11, 6))
        us = np.reshape(us, (-1, 11, 3))

    for i in range(100):
        plt.plot(xs[i, 0:, 0], xs[i, 0:, 1])
    plt.show()


def load_data(system="double_integrator"):

    if system == "double_integrator":
        import pickle

        path = dir_path+"/../../datasets/double_integrator/"

        with open(path+"xs.pkl", 'rb') as f:
            xs = pickle.load(f)            
        with open(path+"us.pkl", 'rb') as f:
            us = pickle.load(f)

    elif system == "quadrotor":

        import pandas as pd

        xs = (
            pd.read_csv("~/Downloads/quadrotor_nlmpc_x.csv", sep=",", header=None)
            .to_numpy()
            .T
        )
        us = (
            pd.read_csv("~/Downloads/quadrotor_nlmpc_u.csv", sep=",", header=None)
            .to_numpy()
            .T
        )

    else:
        raise NotImplementedError

    return xs, us


def create_and_save_deep_models():
    neurons_per_layers = []
    neurons_per_layers.append([5])
    neurons_per_layers.append([5, 5])
    neurons_per_layers.append([5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5, 5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5, 5, 5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5, 5, 5, 5, 5, 5])
    neurons_per_layers.append([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    for neurons_per_layer in neurons_per_layers:
        model = create_and_train_model(neurons_per_layer, xs, us, verbose=True)
        save_model(
            model,
            name="model",
            dir=dir_path
            + "/models/double_integrator_test_{}/".format(
                "_".join(map(str, neurons_per_layer))
            ),
        )


if __name__ == "__main__":

    system = "double_integrator"

    # View some of the trajectory data
    xs, us = load_data(system)

    plot_data(xs, us, system)

    neurons_per_layer = [10, 5]
    # model = create_model(neurons_per_layer)
    model = create_and_train_model(
        neurons_per_layer,
        xs,
        us,
        verbose=True
        )

    save_model(model, dir=dir_path+"/../../models/double_integrator_debug/")

    # Generate the NNs of various numbers of layers...
    # create_and_save_deep_models()
