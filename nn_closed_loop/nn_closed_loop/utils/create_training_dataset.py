import numpy as np
from nn_closed_loop.utils.utils import save_dataset, load_dataset
import nn_closed_loop.constraints as constraints
import nn_closed_loop.dynamics as dynamics
import matplotlib.pyplot as plt


def generate_dataset(dyn, input_constraint, dataset_name="default", t_max=5):

    num_samples = 100

    xs, us = dyn.collect_data(
        t_max,
        input_constraint,
        num_samples=num_samples,
    )

    save_dataset(xs, us, system=dyn.__class__.__name__, dataset_name=dataset_name)


def plot_dataset(xs, us, t_max):
    xs = np.reshape(xs, (-1, t_max+1, xs.shape[-1]))
    for i in range(xs.shape[0]):
        plt.plot(xs[i, :, 0], xs[i, :, 1])
    plt.show()


if __name__ == '__main__':
    nx = 6
    nu = 3

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

    t_max = 10

    dyn = dynamics.Unity(nx=nx, nu=nu)
    input_constraint = constraints.LpConstraint(
        range=init_state_range, p=np.inf
    )
    dataset_name = "nx_{}_nu_{}_scalability".format(str(nx).zfill(3), str(nu).zfill(3))

    import time
    t_start = time.time()
    generate_dataset(dyn, input_constraint, dataset_name=dataset_name, t_max=t_max)
    t_end = time.time()
    print(t_end - t_start)
    # xs, us = load_dataset(system=dyn.__class__.__name__, dataset_name=dataset_name)
    # plot_dataset(xs, us, t_max)
    # print(xs, us)
