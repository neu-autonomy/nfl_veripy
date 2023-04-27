"""Tools for representing sets during reachability analysis."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pypoman
from matplotlib.patches import Rectangle


def make_rect_from_arr(
    arr: np.ndarray,
    dims: np.ndarray,
    color: str,
    linewidth: float,
    fc_color: str,
    ls: int,
    zorder: Optional[int] = None,
    angle: float = 0.0,
) -> Rectangle:
    rect = Rectangle(
        arr[dims, 0],
        arr[dims[0], 1] - arr[dims[0], 0],
        arr[dims[1], 1] - arr[dims[1], 0],
        angle=angle,
        fc=fc_color,
        linewidth=linewidth,
        edgecolor=color,
        zorder=zorder,
        linestyle=ls,
    )
    return rect


class RotatedLpConstraint:
    def __init__(self, pose=None, W=None, theta=0, vertices=None):
        self.width = W
        self.theta = theta
        self.pose = pose
        self.bounding_box = np.vstack(
            (np.min(vertices, axis=0), np.max(vertices, axis=0))
        ).T
        self.vertices = vertices

    def plot(
        self,
        ax,
        plot_2d=True,
    ):
        if not plot_2d:
            raise NotImplementedError
        from scipy.spatial import ConvexHull, convex_hull_plot_2d

        hull = ConvexHull(self.vertices)
        convex_hull_plot_2d(hull, ax)

    def plot3d(
        self,
        ax,
        dims,
        color,
        fc_color="None",
        linewidth=1,
        zorder=2,
        plot_2d=True,
    ):
        raise NotImplementedError

    def get_t_max(self):
        return len(self.bounding_box)


def make_polytope_from_arrs(
    ax,
    A: np.ndarray,
    b: np.ndarray,
    color: str,
    label: str,
    zorder: int,
    ls: str,
    linewidth: float = 1.5,
) -> list:
    vertices = np.stack(pypoman.polygon.compute_polygon_hull(A, b + 1e-10))
    lines = ax.plot(
        [v[0] for v in vertices] + [vertices[0][0]],
        [v[1] for v in vertices] + [vertices[0][1]],
        color=color,
        label=label,
        zorder=zorder,
        ls=ls,
        linewidth=linewidth,
    )
    return lines
