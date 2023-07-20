import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.nn import (
    create_and_train_model,
    create_model,
    load_controller,
    save_model,
)

dir_path = os.path.dirname(os.path.realpath(__file__))


# Policy used to debug CROWN
def random_weight_controller():
    neurons_per_layer = [10, 10]
    xs = np.zeros((10, 1))
    us = np.zeros((10, 1))

    model = create_model(neurons_per_layer, xs.shape[1:], us.shape[1:])
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/random_weight_controller/",
    )


# Policy used to debug CROWN
def corner_policy():
    neurons_per_layer = [2]
    state_range = np.array([-10, 10])
    xs = np.random.uniform(
        low=state_range[0], high=state_range[1], size=(10000, 1)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        if x < 0:
            us[i] = x
        else:
            us[i] = 0

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=200)
    save_model(
        model, name="model", dir=dir_path + "/controllers/corner_policy3/"
    )


# Policy used to debug CROWN
def zero_input_controller():
    neurons_per_layer = [10, 10]
    state_range = np.array([[-20, 20], [-15, 15]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        vy = 0
        vx = 0

        us[i] = np.array([vx, vy])

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/zero_input_controller/",
    )


def potential_field_controller():
    neurons_per_layer = [10, 10]
    state_range = np.array([[-15, 15], [-15, 15]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
        vy = max(min(x[1] / (x[0] ** 2 + x[1] ** 2), 1), -1)
        us[i] = np.array([vx, vy])

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/small_potential_field/",
    )


# Control policy used for CDC 2022 paper
def complex_potential_field_controller():
    neurons_per_layer = [10, 10]
    state_range = np.array([[-10, 10], [-10, 10]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
        vy = max(
            min(
                x[1] / (x[0] ** 2 + x[1] ** 2)
                + np.sign(x[1])
                * 2
                * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
                * np.exp(-(0.5 * x[0] + 2)),
                1,
            ),
            -1,
        )
        us[i] = np.array([vx, vy])

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/complex_potential_field/",
    )


# Control policy used for CDC 2022 paper
def complex_potential_field_controller_go_straight():
    neurons_per_layer = [20, 20]
    state_range = np.array([[-10, 10], [-10, 10]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        if np.abs(x[1]) > 0.1:
            vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
            vy = max(
                min(
                    x[1] / (x[0] ** 2 + x[1] ** 2)
                    + np.sign(x[1])
                    * 2
                    * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
                    * np.exp(-(0.5 * x[0] + 2)),
                    1,
                ),
                -1,
            )
        else:
            vx = 0.8
            vy = 0
        us[i] = np.array([vx, vy])

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(
        model,
        system="GroundRobotSI",
        model_name="go_straight_complex_potential_field",
    )


# Control policy used for CDC 2022 paper (maybe)
def buggy_complex_potential_field_controller():
    neurons_per_layer = [10, 10]
    state_range = np.array([[-10, 10], [-10, 10]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        if np.abs(x[0] + x[1]) < 0.5:
            vx = 1
            vy = -1
        else:
            vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
            vy = max(
                min(
                    x[1] / (x[0] ** 2 + x[1] ** 2)
                    + np.sign(x[1])
                    * 2
                    * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
                    * np.exp(-(0.5 * x[0] + 2)),
                    1,
                ),
                -1,
            )
        us[i] = np.array([vx, vy])

    model = create_and_train_model(neurons_per_layer, xs, us, epochs=20)
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/buggy_complex_potential_field/",
    )


def buggy_complex_potential_field_controller2():
    import math

    neurons_per_layer = [30, 30, 30]
    state_range = np.array([[-10, 10], [-10, 10]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        # if (np.abs(x[0] + x[1]) < 0.1) and x[0] > -2.7 and x[0] < 0:
        #     vx = 1
        #     vy = -1
        # else:
        #     vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
        #     vy = max(
        #         min(
        #             x[1] / (x[0] ** 2 + x[1] ** 2)
        #             + np.sign(x[1])
        #             * 2
        #             * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
        #             * np.exp(-(0.5 * x[0] + 2)),
        #             1,
        #         ),
        #         -1,
        #     )
        center = 2.5
        if (
            (math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)) >= 2)
            and (math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)) <= 3)
            and x[0] < 0
            and x[1] > 0
        ):
            vx = np.clip(
                x[1] / math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)), -1, 1
            )
            vy = np.clip(
                -(x[0] + center)
                / math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)),
                -1,
                1,
            )
        else:
            vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
            vy = max(
                min(
                    x[1] / (x[0] ** 2 + x[1] ** 2)
                    + np.sign(x[1])
                    * 2
                    * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
                    * np.exp(-(0.5 * x[0] + 2)),
                    1,
                ),
                -1,
            )
        us[i] = np.array([vx, vy])

    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=10, verbose=True
    )
    save_model(
        model,
        system="GroundRobotSI",
        model_name="buggy_complex_potential_field5",
    )


def buggy_complex_potential_field_controller3():
    import math

    neurons_per_layer = [30, 30, 30]
    state_range = np.array([[-10, 10], [-10, 10]])
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 2)
    )
    us = np.zeros(xs.shape)
    for i, x in enumerate(xs):
        # if (np.abs(x[0] + x[1]) < 0.1) and x[0] > -2.7 and x[0] < 0:
        #     vx = 1
        #     vy = -1
        # else:
        #     vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
        #     vy = max(
        #         min(
        #             x[1] / (x[0] ** 2 + x[1] ** 2)
        #             + np.sign(x[1])
        #             * 2
        #             * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
        #             * np.exp(-(0.5 * x[0] + 2)),
        #             1,
        #         ),
        #         -1,
        #     )
        center = 2.5
        if (
            (math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)) <= 3)
            and x[0] < 0
            and x[1] > 0
        ):
            vx = np.clip(
                x[1] / math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)), -1, 1
            )
            vy = np.clip(
                -(x[0] + center)
                / math.sqrt(((x[0] + center) ** 2 + x[1] ** 2)),
                -1,
                1,
            )
        else:
            vx = max(min(1 + 2 * x[0] / (x[0] ** 2 + x[1] ** 2), 1), -1)
            vy = max(
                min(
                    x[1] / (x[0] ** 2 + x[1] ** 2)
                    + np.sign(x[1])
                    * 2
                    * (1 + np.exp(-(0.5 * x[0] + 2))) ** -2
                    * np.exp(-(0.5 * x[0] + 2)),
                    1,
                ),
                -1,
            )
        us[i] = np.array([vx, vy])

    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=10, verbose=True
    )
    save_model(
        model,
        system="GroundRobotSI",
        model_name="buggy_complex_potential_field4",
    )


def display_ground_robot_control_field(
    controller: torch.nn.Sequential, ax=None
) -> None:
    x, y = np.meshgrid(np.linspace(-7.5, 4, 20), np.linspace(-7.2, 7.2, 20))
    # import pdb; pdb.set_trace()
    inputs = np.hstack(
        (x.reshape(len(x) * len(x[0]), 1), y.reshape(len(y) * len(y[0]), 1))
    )
    us = controller.forward(torch.Tensor(inputs)).detach().numpy()

    if ax is None:
        # import pdb; pdb.set_trace()
        plt.quiver(
            x,
            y,
            us[:, 0].reshape(len(x), len(y)),
            us[:, 1].reshape(len(x), len(y)),
            color="k",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.rc("font", size=12)
        plt.show()
    else:
        ax.quiver(
            x,
            y,
            us[:, 0].reshape(len(x), len(y)),
            us[:, 1].reshape(len(x), len(y)),
        )


def display_ground_robot_DI_control_field(
    name="ground_robot_DI_avoid_origin_maneuver", ax=None
):
    controller = load_controller(
        system="GroundRobotDI", model_name=name, model_type="keras"
    )
    x, y = np.meshgrid(np.linspace(-7.5, 4, 20), np.linspace(-7.2, 7.2, 20))
    # import pdb; pdb.set_trace()
    inputs = np.hstack(
        (
            x.reshape(len(x) * len(x[0]), 1),
            y.reshape(len(y) * len(y[0]), 1),
            np.zeros((len(x) * len(x[0]), 1)),
            np.zeros((len(y) * len(y[0]), 1)),
        )
    )
    us = controller.predict(inputs)

    if ax is None:
        # import pdb; pdb.set_trace()
        plt.quiver(
            x,
            y,
            us[:, 0].reshape(len(x), len(y)),
            us[:, 1].reshape(len(x), len(y)),
            color="k",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.rc("font", size=12)
        plt.show()
    else:
        ax.quiver(
            x,
            y,
            us[:, 0].reshape(len(x), len(y)),
            us[:, 1].reshape(len(x), len(y)),
        )


def tree_trunks_vs_quad():
    neurons_per_layer = [50, 50]
    state_range = np.array(
        [[-8, 5], [-12, 12], [0, 5], [-4, 4], [-4, 4], [-3, 3]]
    )
    g = 9.8
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    # import pdb; pdb.set_trace()
    # us = np.zeros(xs.shape)
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        # vx_eval = vx.subs([(x, pos[0]), (y, pos[1])])
        # vy_eval = vy1.subs([(x, pos[0]), (y, pos[1])]) + np.sign(
        #     pos[1]
        # ) * vy2.subs([(x, pos[0]), (y, pos[1])])
        # ax_ = vx_eval * dvxdx.subs(
        #     [(x, pos[0]), (y, pos[1])]
        # ) + vy_eval * dvxdy.subs([(x, pos[0]), (y, pos[1])])
        # ay_ = vx_eval * (
        #     dvy1dx.subs([(x, pos[0]), (y, pos[1])])
        #     + np.sign(pos[1]) * dvy2dx.subs([(x, pos[0]), (y, pos[1])])
        # ) + vy_eval * (
        #     dvy1dy.subs([(x, pos[0]), (y, pos[1])])
        #     + np.sign(pos[1]) * dvy2dy.subs([(x, pos[0]), (y, pos[1])])
        # )
        # vx_eval = vx(pos[0], pos[1])
        # vy_eval = vy1(pos[0], pos[1]) + np.sign(pos[1]) * vy2(pos[0], pos[1])
        # ax_ = (
        #     vx_eval * dvxdx(pos[0], pos[1]) + vy_eval * dvxdy(pos[0], pos[1])
        # )
        # ay_ = (
        #     vx_eval * dvy1dx(pos[0], pos[1])
        #     + np.sign(pos[1]) * dvy2dx(pos[0], pos[1])
        #     + vy_eval
        #     * (
        #         dvy1dy(pos[0], pos[1])
        #         + np.sign(pos[1]) * dvy2dy(pos[0], pos[1])
        #     )
        # )

        # vx_eval = max(min(vx(pos[0], pos[2]), 1), -1)
        # vy_eval = max(min(
        #     vy1(pos[0], pos[2]) + np.sign(pos[2]) * vy2(pos[0], pos[2]), 1),
        #     -1,
        # )
        # ax_ = (
        #     vx_eval * dvxdx(pos[0], pos[2]) + vy_eval * dvxdy(pos[0], pos[2])
        # )
        # ay_ = vx_eval * (
        #     dvy1dx(pos[0], pos[2]) + np.sign(pos[2]) * dvy2dx(pos[0], pos[2])
        # ) + vy_eval * (
        #     dvy1dy(pos[0], pos[2]) + np.sign(pos[2]) * dvy2dy(pos[0], pos[2])
        # )

        # # tree_trunks_vs_quadrotor_9_
        # ax_ = 0.5 * (
        #     0.2
        #     + np.sign(pos[1]) * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 2 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * np.sign(pos[0]) * pos[3] / (pos[0] ** 2 + pos[1] ** 2)
        # )
        # ay_ = 0.5 * (
        #     -np.sign(pos[1]) * 2 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     - pos[1] / 15
        #     + 2 * 2 * np.sign(pos[1]) * pos[4] / (pos[0] ** 2 + pos[1] ** 2)
        # )

        # ax_ = 0.5 * (
        #     0.2
        #     + np.sign(pos[1]) * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * np.sign(pos[0]) * pos[3] / (pos[0] ** 2 + pos[1] ** 2)
        # )
        # ay_ = 0.5 * (
        #     -np.sign(pos[1]) * 2 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 0.5 * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     - pos[1] / 15
        #     + 1.5 * np.sign(pos[1]) * pos[4] / (pos[0] ** 2 + pos[1] ** 2)
        # )

        # ax_ = 0.5 * (
        #     0.2
        #     + np.sign(pos[1]) * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * np.sign(pos[0]) * pos[3] / (pos[0] ** 2 + pos[1] ** 2)
        # )
        # ay_ = 0.5 * (
        #     -np.sign(pos[1]) * 2 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     - pos[1] / 15
        #     + 2 * np.sign(pos[1]) * pos[4] / (pos[0] ** 2 + pos[1] ** 2)
        # )

        # tree_trunks_vs_quadrotor_12__
        # ax_ = 0.5 * (
        #     0.2
        #     + np.sign(pos[1]) * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1.5 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * np.sign(pos[0]) * pos[3] / (pos[0] ** 2 + pos[1] ** 2)
        # )
        # ay_ = 0.5 * (
        #     -np.sign(pos[1]) * 2 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
        #     + 1 * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
        #     - pos[1] / 15
        #     + 2 * np.sign(pos[1]) * pos[4] / (pos[0] ** 2 + pos[1] ** 2)
        # )

        # ax = (1 / g) * max(min(ax_, np.pi / 9), -np.pi / 9)
        # ay = -(1 / g) * max(min(ay_, np.pi / 9), -np.pi / 9)
        # az = g

        ax_ = (
            0.4
            + np.sign(pos[1]) * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
            + 1.5 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
            + 1 * np.sign(pos[0]) * pos[3] / (pos[0] ** 2 + pos[1] ** 2)
        )
        ay_ = (
            -np.sign(pos[1]) * 2 * pos[0] / (pos[0] ** 2 + pos[1] ** 2)
            + 1 * pos[1] / (pos[0] ** 2 + pos[1] ** 2)
            - pos[1] / 15
            + 2 * np.sign(pos[1]) * pos[4] / (pos[0] ** 2 + pos[1] ** 2)
        )
        az_ = g - 5 * pos[5]

        # ax_ = 0.5*(0.2)
        # ay_ = 0.5*(-pos[1]/15)

        # for tree in trees:
        #     rel_x, rel_y = pos[0] - tree[0], pos[1] - tree[1]
        #     ax_ += 0.5 * (
        #         np.sign(rel_y) * rel_y / (rel_x**2 + rel_y**2)
        #         + 1.5 * rel_x / (rel_x**2 + rel_y**2)
        #         + 1 * np.sign(rel_x) * pos[3] / (rel_x**2 + rel_y**2)
        #     )
        #     ay_ += 0.5 * (
        #         -np.sign(rel_y) * 2 * rel_x / (rel_x**2 + rel_y**2)
        #         + 1 * rel_y / (rel_x**2 + rel_y**2)
        #         + 2 * np.sign(rel_y) * pos[4] / (rel_x**2 + rel_y**2)
        #     )

        ax = max(min(ax_, np.pi / 9), -np.pi / 9)
        ay = -max(min(ay_, np.pi / 9), -np.pi / 9)
        az = max(min(az_, 2 * g), 0)
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=8, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/tree_trunks_vs_quadrotor_20/",
    )


def simple_quad():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-6, 6], [-6, 6], [-3, 3]]
    )
    g = 9.8
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    # import pdb; pdb.set_trace()
    # us = np.zeros(xs.shape)
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        ax_ = 1 * (pos[0] / (pos[0] ** 2 + pos[1] ** 2))
        ay_ = 1 * (pos[1] / (pos[0] ** 2 + pos[1] ** 2))
        az_ = g - 3 * pos[5]

        ax = max(min(ax_, np.pi / 9), -np.pi / 9)
        ay = -max(min(ay_, np.pi / 9), -np.pi / 9)
        az = max(min(az_, 2 * g), 0)
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=5, verbose=True
    )
    save_model(model, name="model", dir=dir_path + "/controllers/simple_quad/")


# Control policy used for OJCSYS 2022 paper
def discrete_quad_avoid_origin_maneuver():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-2, 2], [-2, 2], [-2, 2]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 16 * pos[0]
            ay = 16 * pos[1]
            az = 16 * (pos[2] - 2.5)
        elif np.abs(pos[0]) < 2.25 and np.abs(pos[1]) < 2.25:
            ax = 4 * np.sign(pos[0])
            ay = 4 * np.sign(pos[1])
            az = 4 * np.sign(pos[2] - 2.5)
        else:
            az = 0
            ax = 0
            ay_ = -0.25 * pos[1] + 4 / pos[1] - 3 * pos[4]
            ay = max(min(ay_, 2), -2)
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=6, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(
        model,
        system="DiscreteQuadrotor",
        model_name="quad_avoid_origin_maneuver_2",
    )


# Control policy used for OJCSYS 2022 paper revision
def sized_up_discrete_quad_avoid_origin_maneuver(network_size):
    neurons_per_layer = network_size
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-2, 2], [-2, 2], [-2, 2]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 16 * pos[0]
            ay = 16 * pos[1]
            az = 16 * (pos[2] - 2.5)
        elif np.abs(pos[0]) < 2.25 and np.abs(pos[1]) < 2.25:
            ax = 4 * np.sign(pos[0])
            ay = 4 * np.sign(pos[1])
            az = 4 * np.sign(pos[2] - 2.5)
        else:
            az = 0
            ax = 0
            ay_ = -0.25 * pos[1] + 4 / pos[1] - 3 * pos[4]
            ay = max(min(ay_, 2), -2)
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    for j in range(1):
        model = create_and_train_model(
            neurons_per_layer, xs, us, epochs=6, verbose=True
        )

        suffix = ""
        for i in neurons_per_layer:
            suffix = suffix + "_{}".format(i)
        # suffix += '_{}'.format(j)
        save_model(
            model,
            system="DiscreteQuadrotor",
            model_name="discrete_quad_avoid_origin_maneuver" + suffix,
        )


def fast_discrete_quad_avoid_origin_maneuver():
    neurons_per_layer = [30, 30, 30]
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-3, 3], [-3, 3], [-3, 3]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(5000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 32 * pos[0]
            ay = 32 * pos[1]
            az = 32 * (pos[2] - 2.5)
        elif np.abs(pos[0]) < 2.25 and np.abs(pos[1]) < 2.25 or True:
            ax = 8 * np.sign(pos[0])
            ay = 8 * np.sign(pos[1])
            az = 8 * np.sign(pos[2] - 2.5)
        else:
            az = 0
            ax = 0
            ay_ = 0  # -0.25*pos[1]+4/pos[1]-3*pos[4]
            ay = max(min(ay_, 3), -3)
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=6, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(
        model,
        system="DiscreteQuadrotor",
        model_name="discrete_quad_avoid_origin_maneuver_fast_test",
    )


# Test
def discrete_quad_test():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [[-8, 5], [-12, 12], [0, 5], [-3, 3], [-3, 3], [-3, 3]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 32 * pos[0]
            ay = 32 * pos[1]
            az = 32 * (pos[2] - 2.5)
        else:
            ax = 8 * np.sign(pos[0])
            ay = 8 * np.sign(pos[1])
            az = 8 * np.sign(pos[2] - 2.5)
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=6, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(model, system="DiscreteQuadrotor", model_name="test3")


def discrete_quad_test2():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-2, 2], [-2, 2], [-2, 2]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 16 * pos[0]
            ay = 0
            az = 0
        else:
            ax = 4 * np.sign(pos[0])
            ay = 0
            az = 0
        us[i] = np.array([ax, ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=6, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(model, system="DiscreteQuadrotor", model_name="test2")


def quad_avoid_origin_maneuver():
    neurons_per_layer = [30, 30]
    g = 9.8
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-2, 2], [-2, 2], [-2, 2]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 40 * pos[0]
            ay = 40 * pos[1]
            az = 39.2 * (pos[2] - 2.5)
        elif np.abs(pos[0]) < 2.25 and np.abs(pos[1]) < 2.25:
            ax = 10 * np.sign(pos[0])
            ay = 10 * np.sign(pos[1])
            az = g + g * np.sign(pos[2] - 2.5)
        else:
            az = g
            ax = 0
            ay_ = -0.25 * pos[1] + 4 / pos[1] - 6 * pos[4]
            ay = max(min(ay_, 3), -3)
        us[i] = np.array([ax, -ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=6, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(model, system="Quadrotor", model_name="avoid_origin_maneuver")


def quad_test():
    neurons_per_layer = [30, 30]
    g = 9.8
    state_range = np.array(
        [[-8, 5], [-12, 12], [1, 4], [-5, 5], [-5, 5], [-5, 5]]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(10000000, 6)
    )
    us = np.zeros((len(xs), 3))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 40 * pos[0]
            ay = 40 * pos[1]
            az = 39.2 * (pos[2] - 2.5)
        else:
            ax = 10 * np.sign(pos[0])
            ay = 10 * np.sign(pos[1])
            az = g + g * np.sign(pos[2] - 2.5)

        us[i] = np.array([ax, -ay, az])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=6, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(model, system="Quadrotor", model_name="test")


def ground_robotDI():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-10, 10],
            [-10, 10],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        ax_ = 0.5
        ay_ = 0

        ax = max(min(ax_, 1), -1)
        ay = max(min(ay_, 1), -1)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=5, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/ground_robot_DI_simple/",
    )


def ground_robotDI_obstacle():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        ax_ = 1 * (pos[0] / (pos[0] ** 2 + pos[1] ** 2))
        ay_ = 1 * (pos[1] / (pos[0] ** 2 + pos[1] ** 2))

        ax = max(min(ax_, 1), -1)
        ay = max(min(ay_, 1), -1)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=5, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/ground_robot_DI_obstacle_simple/",
    )


def ground_robotDI_obstacle_2D():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        if abs(pos[0]) < 0.33:
            ax_ = 12 * pos[0]
        elif abs(pos[0]) < 2:
            ax_ = 10 * pos[0] / (pos[0] ** 5)
        else:
            ax_ = 1
        if abs(pos[1]) < 0.33:
            ay_ = 12 * pos[1]
        elif abs(pos[1]) < 2:
            ay_ = 10 * pos[1] / (pos[1] ** 5)
        else:
            ay_ = -1 * np.sign(pos[1])

        ax = max(min(ax_, 4), -4)
        ay = max(min(ay_, 4), -4)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=15, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path
        + "/controllers/ground_robot_DI_obstacle_simple_2D_slant/",
    )


def ground_robotDI_avoid_origin_2D_limited():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 2 and np.abs(pos[0]) < 2:
            ax = 4 * np.sign(pos[0])
            ay = 4 * np.sign(pos[1])
        else:
            ax = 0
            ay = 0
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=15, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/ground_robot_DI_avoid_origin_limited/",
    )


def ground_robotDI_avoid_origin_2D_maneuver():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 2 and np.abs(pos[1]) < 2:
            ax = 4 * np.sign(pos[0])
            ay = 4 * np.sign(pos[1])
        else:
            ax = 0
            ay_ = -0.2 * pos[1] + 4 / pos[1]
            ay = max(min(ay_, 2), -2)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=15, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path + "/controllers/ground_robot_DI_avoid_origin_maneuver/",
    # )
    save_model(
        model,
        system="GroundRobotDI",
        model_name="ground_robot_DI_avoid_origin_maneuver",
    )


def ground_robotDI_avoid_origin_2D_maneuver_velocity():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 2 and np.abs(pos[1]) < 2:
            ax = 4 * np.sign(pos[0])
            ay = 4 * np.sign(pos[1])
        else:
            ax = 0
            ay_ = -0.2 * pos[1] + 4 / pos[1] - 3 * pos[3]
            ay = max(min(ay_, 2), -2)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=15, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path
        + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    )


def ground_robotDI_avoid_origin_2D_maneuver_velocity2():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        if np.abs(pos[0]) < 0.25 and np.abs(pos[1]) < 0.25:
            ax = 16 * pos[0]
            ay = 16 * pos[1]
        elif np.abs(pos[0]) < 2.25 and np.abs(pos[1]) < 2.25:
            ax = 4 * np.sign(pos[0])
            ay = 4 * np.sign(pos[1])
        else:
            ax = 0
            ay_ = -0.25 * pos[1] + 4 / pos[1] - 3 * pos[3]
            ay = max(min(ay_, 2), -2)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=10, verbose=True
    )
    # save_model(
    #     model,
    #     name="model",
    #     dir=dir_path
    #     + "/controllers/ground_robot_DI_avoid_origin_maneuver_velocity/",
    # )
    save_model(
        model,
        system="GroundRobotDI",
        model_name="ground_robot_DI_avoid_origin_maneuver_velocity",
    )


def ground_robotDI_obstacle_2D_circle():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-2, 2],
            [-2, 2],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        ax_ = (
            np.sign(pos[0]) * 10 * pos[0] / ((pos[0] ** 2 + pos[1] ** 2) ** 4)
        )
        ay_ = (
            np.sign(pos[1]) * 10 * pos[1] / ((pos[0] ** 2 + pos[1] ** 2) ** 4)
        )

        ax = max(min(ax_, 4), -4)
        ay = max(min(ay_, 4), -4)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=15, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/ground_robot_DI_obstacle_2D_circle/",
    )


def ground_robotDI_sine():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-30, 30],
            [-30, 30],
            [-3, 3],
            [-3, 3],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(2000000, 4)
    )
    us = np.zeros((len(xs), 2))
    for i, pos in enumerate(xs):
        ax_ = 0
        ay_ = -2 * (2 * np.pi / 10) ** 2 * np.sin(2 * np.pi / 10 * pos[0])

        ax = max(min(ax_, 1), -1)
        ay = max(min(ay_, 1), -1)
        us[i] = np.array([ax, ay])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=8, verbose=True
    )
    save_model(
        model,
        name="model",
        dir=dir_path + "/controllers/ground_robot_DI_sine/",
    )


def double_integratorx4():
    neurons_per_layer = [20, 20]
    state_range = np.array(
        [
            [-20, 20],
            [-20, 20],
            [-20, 20],
            [-20, 20],
            [-10, 10],
            [-10, 10],
            [-10, 10],
            [-10, 10],
        ]
    )
    xs = np.random.uniform(
        low=state_range[:, 0], high=state_range[:, 1], size=(1000000, 8)
    )
    us = np.zeros((len(xs), 4))
    for i, pos in enumerate(xs):
        ax_ = 0.5
        ay_ = 0
        az_ = 0
        ai_ = 0

        ax = max(min(ax_, 1), -1)
        ay = max(min(ay_, 1), -1)
        az = max(min(az_, 1), -1)
        ai = max(min(ai_, 1), -1)
        us[i] = np.array([ax, ay, az, ai])
        if np.mod(i, 1000000) == 0:
            print("yeayea")
    print("ok")
    model = create_and_train_model(
        neurons_per_layer, xs, us, epochs=5, verbose=True
    )
    save_model(model, name="model", dir=dir_path + "/controllers/simple_4_DI/")


def build_controller_from_matlab(filename="quad_mpc_data.mat"):
    neurons_per_layer = [25, 25, 25]

    file = dir_path + "/controllers/MATLAB_data/" + filename
    mat = scipy.io.loadmat(file)
    xs = mat["data"][0][0][0][:, 0:6]
    us = mat["data"][0][0][1]

    model = create_and_train_model(
        neurons_per_layer,
        xs,
        us,
        epochs=50,
        verbose=True,
        validation_split=0.1,
    )
    save_model(
        model, name="model", dir=dir_path + "/controllers/quadrotor_matlab_5/"
    )


def generate_mpc_data_quadrotor(num_samples=100):
    dyn = dynamics.Quadrotor()
    state_range = np.array(
        [  # (num_inputs, 3)
            [-2.8, -0.8, -0.8, -0.5, -0.5, -0.5],
            [0.2, 0.8, 0.8, 0.5, 0.5, 0.5],
        ]
    ).T
    input_constraint = constraints.LpConstraint(range=state_range)
    # import pdb; pdb.set_trace()
    xs, us = dyn.collect_data(
        t_max=1, input_constraint=input_constraint, num_samples=num_samples
    )
    with open("xs.pickle", "wb") as handle:
        pickle.dump(xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("us.pickle", "wb") as handle:
        pickle.dump(xs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    # random_weight_controller()
    # avoid_origin_controller_simple()
    # stop_at_origin_controller()
    # zero_input_controller()
    # complex_potential_field_controller()
    # buggy_complex_potential_field_controller2()
    # complex_potential_field_controller_go_straight()
    # display_ground_robot_control_field(name='complex_potential_field')
    # build_controller_from_matlab("quad_mpc_data_paths_not_as_small.mat")
    # generate_mpc_data_quadrotor()
    # tree_trunks_vs_quad()
    # simple_quad()
    # ground_robotDI_obstacle_2D()
    # ground_robotDI_avoid_origin_2D_maneuver_velocity2()
    # fast_discrete_quad_avoid_origin_maneuver()
    # quad_avoid_origin_maneuver()
    # discrete_quad_test()
    # double_integratorx4()
    # ground_robotDI_sine()
    # corner_policy()
    # quad_sizes = [
    #     [64, 64],
    #     [128, 128],
    #     [128, 128, 128],
    #     [256, 256],
    #     [256, 256, 256],
    # ]
    quad_sizes = [[100, 100]]
    for size in quad_sizes:
        sized_up_discrete_quad_avoid_origin_maneuver(network_size=size)


if __name__ == "__main__":
    main()
