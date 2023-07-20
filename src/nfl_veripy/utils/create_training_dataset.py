import os

import matplotlib.pyplot as plt
import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
import numpy as np
from nfl_veripy.utils.nn import create_and_train_model, save_model
from nfl_veripy.utils.utils import load_dataset, save_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))


def generate_dataset(dyn, input_constraint, dataset_name="default", t_max=5):
    num_samples = 100

    xs, us = dyn.collect_data(
        t_max,
        input_constraint,
        num_samples=num_samples,
    )

    save_dataset(
        xs, us, system=dyn.__class__.__name__, dataset_name=dataset_name
    )


def plot_dataset(xs, us, t_max):
    xs = np.reshape(xs, (-1, t_max + 1, xs.shape[-1]))
    for t in range(xs.shape[1]):
        plt.plot(xs[:, t, 0], xs[:, t, 1], ".")
    plt.show()


def create_dataset_and_train_model(nx, nu, t_max=10):
    init_state_range = np.empty((nx, 2))
    init_state_range[0, 0] = 0.9
    init_state_range[0, 1] = 1.1
    init_state_range[1:, 0] = -0.1
    init_state_range[1:, 1] = 0.1

    # init_state_range = np.array(
    #     [  # (num_inputs, 2)
    #         [0.9, -0.1],
    #         [1.1, 0.1],
    #     ]
    # ).T

    dyn = dynamics.Unity(nx=nx, nu=nu)
    input_constraint = constraints.LpConstraint(
        range=init_state_range, p=np.inf
    )
    dataset_name = "nx_{}_nu_{}_scalability".format(
        str(nx).zfill(3), str(nu).zfill(3)
    )

    generate_dataset(
        dyn, input_constraint, dataset_name=dataset_name, t_max=t_max
    )
    xs, us = load_dataset(
        system=dyn.__class__.__name__, dataset_name=dataset_name
    )
    # plot_dataset(xs, us, t_max)

    neurons_per_layer = [10, 10]
    model = create_and_train_model(neurons_per_layer, xs, us, verbose=True)

    model_name = "nx_{}_nu_{}".format(str(nx).zfill(3), str(nu).zfill(3))
    save_model(model, system=dyn.__class__.__name__, model_name=model_name)

    # print(xs, us)


if __name__ == "__main__":
    nu = 2
    for nx in [1, 2, 3, 4, 5]:
        create_dataset_and_train_model(nx, nu)
