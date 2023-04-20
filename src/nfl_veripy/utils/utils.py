import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm

dir_path = os.path.dirname(os.path.realpath(__file__))


def bisect(input_range):
    return sect(input_range, num_sects=2)


def sect(input_range, num_sects=3, select="random"):
    input_shape = input_range.shape[:-1]
    if select == "random":
        # doesnt work
        input_dim_to_sect = np.random.randint(0, num_inputs)
    else:
        lengths = input_range[..., 1] - input_range[..., 0]
        input_dim_to_sect = np.unravel_index(lengths.argmax(), lengths.shape)
    input_ranges = np.tile(
        input_range,
        (num_sects,) + tuple([1 for _ in range(len(input_shape) + 1)]),
    )
    diff = (
        input_range[input_dim_to_sect + (1,)]
        - input_range[input_dim_to_sect + (0,)]
    ) / float(num_sects)
    for i in range(num_sects - 1):
        new_endpt = input_range[input_dim_to_sect + (0,)] + (i + 1) * diff
        input_ranges[(i,) + input_dim_to_sect + (1,)] = new_endpt
        input_ranges[(i + 1,) + input_dim_to_sect + (0,)] = new_endpt
    return input_ranges


def get_sampled_outputs(input_range, propagator, N=1000):
    input_shape = input_range.shape[:-1]
    sampled_inputs = np.random.uniform(
        input_range[..., 0], input_range[..., 1], (N,) + input_shape
    )

    sampled_outputs = propagator.forward_pass(sampled_inputs)

    return sampled_outputs


def samples_to_range(sampled_outputs):
    num_outputs = sampled_outputs.shape[-1]
    output_range = np.empty((num_outputs, 2))
    output_range[:, 1] = np.max(sampled_outputs, axis=0)
    output_range[:, 0] = np.min(sampled_outputs, axis=0)
    return output_range


def stablebaselines2torch(good_sess, network_params, activation="relu"):
    import torch

    obs_ph, numlayer, w_tsr, b_tsr = network_params

    modules = []
    # with tf.compat.v1.Session() as sess:
    # init = tf.global_variables_initializer()
    # sess.run(init)
    for i in range(len(w_tsr)):
        w = good_sess.run(w_tsr[i]).T
        b = good_sess.run(b_tsr[i])
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        # print("Layer {}: Input: {}, Output: {}".format(i, w.shape[1], w.shape[0]))
        # print("Bias {}: shape: {}".format(b, b.shape))
        # print(w)
        # import pdb; pdb.set_trace()
        linear.weight.data.copy_(torch.Tensor(w))
        linear.bias.data.copy_(torch.Tensor(b))
        modules.append(linear)
        if i < len(w_tsr) - 1:
            # print('adding relu...')
            modules.append(torch.nn.ReLU())
        # else:
        # print('not adding relu.')
    torch_model = torch.nn.Sequential(*modules)
    return torch_model


def colors(t, cmap="tab10"):
    return cm.get_cmap(cmap)(t % len(cm.get_cmap(cmap).colors))


def save_dataset(xs, us, system="DoubleIntegrator", dataset_name="default"):
    path = "{}/../../datasets/{}/{}".format(dir_path, system, dataset_name)
    os.makedirs(path, exist_ok=True)
    with open(path + "/dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)


def load_dataset(system="DoubleIntegrator", dataset_name="default"):
    path = "{}/../../datasets/{}/{}".format(dir_path, system, dataset_name)
    with open(path + "/dataset.pkl", "rb") as f:
        xs, us = pickle.load(f)
    return xs, us


def range_to_polytope(state_range):
    num_states = state_range.shape[0]
    A = np.vstack([np.eye(num_states), -np.eye(num_states)])
    b = np.hstack([state_range[:, 1], -state_range[:, 0]])
    return A, b


def get_polytope_A(num):
    theta = np.linspace(0, 2 * np.pi, num=num + 1)
    A_out = np.dstack([np.cos(theta), np.sin(theta)])[0][:-1]
    return A_out


def get_next_state(xt, ut, At, bt, ct):
    return np.dot(At, xt.T) + np.dot(bt, ut.T)


def plot_polytope_facets(A, b, ls="-", show=True):
    import matplotlib.pyplot as plt

    cs = [
        "r",
        "g",
        "b",
        "brown",
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:brown",
    ]
    ls = ["-", "-", "-", "-", "--", "--", "--", "--"]
    num_facets = b.shape[0]
    x = np.linspace(1, 5, 2000)
    for i in range(num_facets):
        alpha = 0.2
        if A[i, 1] == 0:
            offset = -0.1 * np.sign(A[i, 0])
            plt.axvline(x=b[i] / A[i, 0], ls=ls[i], c=cs[i])
            plt.fill_betweenx(
                y=np.linspace(-2, 2, 2000),
                x1=b[i] / A[i, 0],
                x2=offset + b[i] / A[i, 0],
                fc=cs[i],
                alpha=alpha,
            )
        else:
            offset = -0.1 * np.sign(A[i, 1])
            y = (b[i] - A[i, 0] * x) / A[i, 1]
            plt.plot(x, y, ls=ls[i], c=cs[i])
            plt.fill_between(x, y, y + offset, fc=cs[i], alpha=alpha)
    if show:
        plt.show()


def get_polytope_verts(A, b):
    import pypoman

    # vertices = pypoman.duality.compute_polytope_vertices(A, b)
    vertices = pypoman.polygon.compute_polygon_hull(A, b)
    print(vertices)


def get_crown_matrices(
    propagator, state_set, num_control_inputs, sensor_noise
):
    nn_input_max = torch.Tensor(np.expand_dims(state_set.range[:, 1], axis=0))
    nn_input_min = torch.Tensor(np.expand_dims(state_set.range[:, 0], axis=0))
    if sensor_noise is not None:
        # Because there might be sensor noise, the NN could see a different
        # set of states than the system is actually in
        raise NotImplementedError

    # Compute the NN output matrices (for this backreachable_set)
    C = torch.eye(num_control_inputs).unsqueeze(0)
    return CROWNMatrices(
        *propagator.network(
            method_opt=propagator.method_opt,
            norm=np.inf,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )
    )


class CROWNMatrices:
    def __init__(self, lower_A, upper_A, lower_sum_b, upper_sum_b):
        self.upper_A = upper_A.detach()
        self.lower_A = lower_A.detach()
        self.upper_sum_b = upper_sum_b.detach()
        self.lower_sum_b = lower_sum_b.detach()

    def to_numpy(self):
        return (
            self.upper_A_numpy,
            self.lower_A_numpy,
            self.upper_sum_b_numpy,
            self.lower_sum_b_numpy,
        )

    @property
    def lower_A_numpy(self):
        return self.lower_A.numpy()[0]

    @property
    def upper_A_numpy(self):
        return self.upper_A.numpy()[0]

    @property
    def lower_sum_b_numpy(self):
        return self.lower_sum_b.numpy()[0]

    @property
    def upper_sum_b_numpy(self):
        return self.upper_sum_b.numpy()[0]


def plot_time_data(info):
    labels = {
        "br_lp": "LPs (Backreach)",
        "bp_lp": "LPs (BReach)",
        "nstep_bp_lp": "LPs (ReBReach)",
        "crown": "CROWN (BReach)",
        "nstep_crown": "CROWN (ReBReach)",
        "other": "Other (BReach)",
        "nstep_other": "Other (ReBReach)",
    }
    num_entries = {}
    vals = []
    for dict in info["per_timestep"]:
        step_values = {}
        for key in labels:
            if key in dict:
                step_values[key] = sum(dict[key])
                if key in num_entries:
                    num_entries[key] += len(dict[key])
                else:
                    num_entries[key] = len(dict[key])
            else:
                step_values[key] = 0
                if key in num_entries:
                    num_entries[key] += 0
                else:
                    num_entries[key] = 0

        vals.append(step_values)

    summed_vals = {
        "br_lp": [0],
        "bp_lp": [0],
        "nstep_bp_lp": [0],
        "crown": [0],
        "nstep_crown": [0],
        "other": [0],
        "nstep_other": [0],
    }
    for i in range(len(vals)):
        for key in vals[i]:
            summed_vals[key].append(summed_vals[key][i] + vals[i][key])
    summed_value_list = list(summed_vals.values())

    if num_entries["bp_lp"] > 0:
        print("Number of LPs solved (BReach): {}".format(num_entries["bp_lp"]))
        print(
            "Time per LP solved (BReach): {0:.4f}".format(
                summed_vals["bp_lp"][-1] / num_entries["bp_lp"]
            )
        )
    if num_entries["nstep_bp_lp"] > 0:
        print(
            "Number of LPs solved (ReBReach): {}".format(
                num_entries["nstep_bp_lp"]
            )
        )
        print(
            "Time per LP solved (ReBReach): {0:.4f}".format(
                summed_vals["nstep_bp_lp"][-1] / num_entries["nstep_bp_lp"]
            )
        )
    print(
        "Number of CROWN calculations (BReach): {}".format(
            num_entries["crown"]
        )
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    cm = plt.cm.get_cmap("tab20")
    for i, dict in enumerate(vals):
        ax.bar(
            labels.values(),
            dict.values(),
            color=cm.colors[i],
            bottom=[x[i] for x in summed_value_list],
        )
    ax.set_ylabel("Time (s)", fontsize=12)

    textstr = "\n".join(
        (
            "Number of LPs (BReach): {}".format(num_entries["bp_lp"]),
            "Time per LP (BReach): {0:.4f}".format(
                summed_vals["bp_lp"][-1] / num_entries["bp_lp"]
            ),
            "Number of LPs (ReBReach): {}".format(num_entries["nstep_bp_lp"]),
            "Time per LP (ReBReach): {0:.4f}".format(
                summed_vals["nstep_bp_lp"][-1] / num_entries["nstep_bp_lp"]
            ),
        )
    )

    # these are matplotlib.patch.Patch properties
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0.5,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    textstr = "\n".join(
        (
            "Number of LPs (BReach): {}".format(num_entries["bp_lp"]),
            "Time per LP (BReach): {0:.4f}".format(
                summed_vals["bp_lp"][-1] / num_entries["bp_lp"]
            ),
        )
    )
    ax.text(
        0.5,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    plt.xticks(rotation=60, fontsize=12)
    plt.subplots_adjust(bottom=0.36)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    plt.show()


def create_cl_model(dynamics, num_steps):
    path = "{}/../../models/{}/{}".format(
        dir_path, "Pendulum", "default/single_pendulum_small_controller.torch"
    )
    controller = dynamics.controller_module
    # controller.load_state_dict(self.network.state_dict(), strict=False)
    controller.load_state_dict(torch.load(path), strict=False)
    dynamics = dynamics.dynamics_module
    model = ClosedLoopDynamics(controller, dynamics, num_steps=num_steps)
    return model


# Define computation as a nn.Module.
class ClosedLoopDynamics(nn.Module):
    def __init__(self, controller, dynamics, num_steps=1):
        super().__init__()
        self.controller = controller
        self.dynamics = dynamics
        self.num_steps = num_steps

    def forward(self, xt):
        xts = [xt]
        for i in range(self.num_steps):
            ut = self.controller(xts[-1])
            xt1 = self.dynamics(xts[-1], ut)

            xts.append(xt1)

        return xts[-1]


if __name__ == "__main__":
    # save_dataset(xs, us)
    # xs, us = load_dataset()

    import matplotlib.pyplot as plt

    A = np.array([[1, 1], [0, 1], [-1, -1], [0, -1]])
    b = np.array([2.8, 0.41, -2.7, -0.39])

    A2 = np.array(
        [[1, 1], [0, 1], [-0.97300157, -0.95230697], [0.05399687, -0.90461393]]
    )
    b2 = np.array([2.74723146, 0.30446292, -2.64723146, -0.28446292])

    # get_polytope_verts(A, b)
    plot_polytope_facets(A, b)
    # get_polytope_verts(A2, b2)
    plot_polytope_facets(A2, b2, ls="--")
    plt.show()
