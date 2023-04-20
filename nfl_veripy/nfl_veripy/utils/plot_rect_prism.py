import matplotlib.pyplot as plt
import numpy as np


def x_y_edge(
    x_range, y_range, z_range, ax, color, linewidth, fc_color, zorder=2
):
    lines = []
    xx, yy = np.meshgrid(x_range, y_range)

    for value in [0, 1]:
        zz = np.expand_dims(np.expand_dims(z_range[value], axis=0), axis=0)
        lines.append(
            ax.plot_wireframe(
                xx, yy, zz, color=color, linewidth=linewidth, zorder=zorder
            )
        )
        # lines.append(
        #     ax.plot_surface(xx, yy, zz, color=fc_color, zorder=zorder)
        # )
    return lines


def y_z_edge(
    x_range, y_range, z_range, ax, color, linewidth, fc_color, zorder=2
):
    lines = []
    yy, zz = np.meshgrid(y_range, z_range)

    for value in [0, 1]:
        xx = np.expand_dims(np.expand_dims(x_range[value], axis=0), axis=0)
        lines.append(
            ax.plot_wireframe(
                xx, yy, zz, color=color, linewidth=linewidth, zorder=zorder
            )
        )
        # lines.append(ax.plot_surface(xx, yy, zz, color="r", alpha=0.2))
    return lines


def x_z_edge(
    x_range, y_range, z_range, ax, color, linewidth, fc_color, zorder=2
):
    lines = []
    xx, zz = np.meshgrid(x_range, z_range)

    for value in [0, 1]:
        yy = np.expand_dims(np.expand_dims(y_range[value], axis=0), axis=0)
        lines.append(
            ax.plot_wireframe(
                xx, yy, zz, color=color, linewidth=linewidth, zorder=zorder
            )
        )
        # lines.append(ax.plot_surface(xx, yy, zz, color="r", alpha=0.2))
    return lines


def rect_prism(
    x_range, y_range, z_range, axes, color, linewidth, fc_color, zorder=None
):
    lines = []
    lines += x_y_edge(
        x_range[0],
        y_range[0],
        z_range[0],
        axes,
        color,
        linewidth,
        fc_color,
        zorder=zorder,
    )
    lines += y_z_edge(
        x_range[0],
        y_range[0],
        z_range[0],
        axes,
        color,
        linewidth,
        fc_color,
        zorder=zorder,
    )
    lines += x_z_edge(
        x_range[0],
        y_range[0],
        z_range[0],
        axes,
        color,
        linewidth,
        fc_color,
        zorder=zorder,
    )
    return lines


def main():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    rect_prism(
        np.array([-1, 1]),
        np.array([-1, 1]),
        np.array([-0.5, 0.5]),
        axes=ax,
        color="r",
        linewidth=1,
        fc_color="None",
        zorder=1,
    )
    plt.show()


if __name__ == "__main__":
    main()
