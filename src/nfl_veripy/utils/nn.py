"""Create and train simple neural networks."""

import os
from typing import Optional

import numpy as np
from crown_ibp.conversions.keras2torch import keras2torch
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model, model_from_json

dir_path = os.path.dirname(os.path.realpath(__file__))


def create_model(
    neurons_per_layer: list[int],
    input_shape: tuple,
    output_shape: tuple,
    activation: str = "relu",
) -> Sequential:
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
    neurons_per_layer: list[int],
    xs: np.ndarray,
    us: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: bool = False,
    validation_split: float = 0.0,
) -> Sequential:
    model = create_model(
        neurons_per_layer, input_shape=xs.shape[1:], output_shape=us.shape[1:]
    )
    model.fit(
        xs,
        us,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_split=validation_split,
    )
    return model


def save_model(
    model: Sequential,
    system: str = "DoubleIntegrator",
    model_name: str = "default",
    path: Optional[str] = None,
) -> None:
    if not path:
        path = "{}/../_static/models/{}/{}".format(
            dir_path, system, model_name
        )
    os.makedirs(path, exist_ok=True)
    # serialize model to JSON
    model_json = model.to_json()
    name = "model"
    with open(path + "/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + "/" + name + ".h5")
    print("Saved model to disk")


def load_controller(
    system: str = "DoubleIntegrator",
    model_name: str = "default",
    model_type: str = "torch",
) -> Sequential:
    if type(model_name) is not str:
        # TODO: update type in signature to properly reflect other options
        return model_name
    system = system.replace(
        "OutputFeedback", ""
    )  # remove OutputFeedback suffix if applicable
    path = "{}/../_static/models/{}/{}".format(dir_path, system, model_name)
    if system != "Taxinet":
        with open(path + "/model.json", "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(path + "/model.h5")
    else:
        model = load_model(path + "/model.h5")

    if model_type == "keras":
        return model
    torch_model = keras2torch(model, "torch_model")
    weights = []
    for name, param in torch_model.named_parameters():
        weights.append(param.detach().numpy())

    return torch_model


def load_controller_unity(nx: int, nu: int) -> Sequential:
    system = "unity"
    path = "{}/../_static/models/{}".format(dir_path, system)
    model_name = "/nx_{}_nu_{}/model".format(
        str(nx).zfill(3), str(nu).zfill(3)
    )
    filename = path + model_name + ".json"
    try:
        with open(filename, "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(path + model_name + ".h5")
    except FileNotFoundError:
        model = create_model([5, 5], (nx,), (nu,))
        save_model(
            model,
            system=system,
            model_name="nx_{}_nu_{}".format(
                str(nx).zfill(3), str(nu).zfill(3)
            ),
        )
    torch_model = keras2torch(model, "torch_model")

    return torch_model


def rpm_converter(
    system="DoubleIntegrator", model_name="default", model_type="torch"
):
    system = system.replace(
        "OutputFeedback", ""
    )  # remove OutputFeedback suffix if applicable
    path = "{}/../_static/models/{}/{}".format(dir_path, system, model_name)
    if system != "Taxinet":
        with open(path + "/model.json", "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(path + "/model.h5")
    else:
        model = load_model(path + "/model.h5")

    torch_model = keras2torch(model, "torch_model")
    weights = []
    for name, param in torch_model.named_parameters():
        weights.append(param.detach().numpy())


def plot_data(xs: np.ndarray, us: np.ndarray, system: str) -> None:
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


def load_data(
    system: str = "double_integrator",
) -> tuple[np.ndarray, np.ndarray]:
    if system == "double_integrator":
        import pickle

        path = dir_path + "/../_static/datasets/double_integrator/"

        with open(path + "xs.pkl", "rb") as f:
            xs = pickle.load(f)
        with open(path + "us.pkl", "rb") as f:
            us = pickle.load(f)

    elif system == "quadrotor":
        import pandas as pd

        xs = (
            pd.read_csv(
                "~/Downloads/quadrotor_nlmpc_x.csv", sep=",", header=None
            )
            .to_numpy()
            .T
        )
        us = (
            pd.read_csv(
                "~/Downloads/quadrotor_nlmpc_u.csv", sep=",", header=None
            )
            .to_numpy()
            .T
        )

    else:
        raise NotImplementedError

    return xs, us


def create_and_save_deep_models() -> None:
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
            system="model",
            path=dir_path
            + "../_static/models/double_integrator_test_{}/".format(
                "_".join(map(str, neurons_per_layer))
            ),
        )


def train_n_models(xs: np.ndarray, us: np.ndarray) -> None:
    num_models = 3

    import nfl_veripy.analyzers as analyzers
    import nfl_veripy.constraints as constraints
    import nfl_veripy.dynamics as dynamics

    for i in range(num_models):
        neurons_per_layer = [10, 5]
        # model = create_model(neurons_per_layer)
        model = create_and_train_model(
            neurons_per_layer, xs, us, epochs=1000, verbose=False
        )

        us_pred = model.predict(xs)
        # ax.scatter(xs[:,0], xs[:,1], us_pred, color=colors[i])
        print(us_pred)

        controller = keras2torch(model, "tmp_model")
        dyn = dynamics.DoubleIntegrator()
        init_state_range = np.array(
            [  # (num_inputs, 2)
                [2.5, 3.0],  # x0min, x0max
                [-0.25, 0.25],  # x1min, x1max
            ]
        )
        initial_set = constraints.LpConstraint(
            range=init_state_range, p=np.inf
        )
        partitioner_hyperparams = {
            "type": "None",
            "make_animation": False,
            "show_animation": False,
        }
        propagator_hyperparams = {
            "type": "CROWN",
            "input_shape": init_state_range.shape[:-1],
        }
        analyzer = analyzers.ClosedLoopAnalyzer(controller, dyn)
        analyzer.partitioner = partitioner_hyperparams
        analyzer.propagator = propagator_hyperparams
        t_max = 5
        reachable_sets, analyzer_info = analyzer.get_reachable_set(
            initial_set, t_max=t_max
        )
        analyzer.visualize(
            initial_set,
            reachable_sets,
            show_samples=True,
            show=True,
            aspect="auto",
            iteration=None,
            inputs_to_highlight=[
                {"dim": [0], "name": "$x_0$"},
                {"dim": [1], "name": "$x_1$"},
            ],
            **analyzer_info,
        )


if __name__ == "__main__":
    system = "double_integrator"

    # View some of the trajectory data
    xs, us = load_data(system)

    train_n_models(xs, us)

    # plot_data(xs, us, system)

    # neurons_per_layer = [10, 5]
    # # model = create_model(neurons_per_layer)
    # model = create_and_train_model(
    #     neurons_per_layer,
    #     xs,
    #     us,
    #     verbose=True
    #     )

    # save_model(model, dir=dir_path+"/../../models/double_integrator_debug/")

    # Generate the NNs of various numbers of layers...
    # create_and_save_deep_models()
