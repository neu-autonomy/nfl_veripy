import numpy as np
import pypoman
from matplotlib.patches import Rectangle
from nn_closed_loop.utils.plot_rect_prism import rect_prism

class Constraint:
    def __init__(self):
        pass

class PolytopeConstraint(Constraint):
    def __init__(self, A=None, b=None):
        Constraint.__init__(self)
        self.A = A
        self.b = b

    def to_linf(self):
        if isinstance(self.A, list):
            # Mainly for backreachability, return a list of ranges if
            # the constraint contains a list of polytopes
            ranges = []
            for i in range(len(self.A)):
                vertices = np.stack(
                    pypoman.duality.compute_polytope_vertices(self.A[i], self.b[i])
                )
                ranges.append(np.dstack(
                    [np.min(vertices, axis=0), np.max(vertices, axis=0)]
                )[0])
        else:
            vertices = np.stack(
                pypoman.duality.compute_polytope_vertices(self.A, self.b)
            )
            ranges = np.dstack(
                [np.min(vertices, axis=0), np.max(vertices, axis=0)]
            )[0]
        return ranges

    def plot(self, ax, dims, color, fc_color="None", linewidth=3, label=None, zorder=2, plot_2d=True):
        if not plot_2d:
            raise NotImplementedError
            return self.plot3d(ax, dims, color, fc_color=fc_color, linewidth=linewidth, zorder=zorder)

        # TODO: this doesn't use the computed input_dims...

        lines = []

        if isinstance(self.A, list):
            # Backward reachability
            # input_constraint.A will be a list
            # of polytope facets, whose union is the estimated
            # backprojection set

            for i in range(len(self.A)):
                line = make_polytope_from_arrs(ax, self.A[i], self.b[i], color, label, zorder)
                lines += line

        else:
            # Forward reachability
            for i in range(len(self.b)):
                line = make_polytope_from_arrs(ax, self.A, self.b[i], color, label, zorder)
                lines += line

        return lines

    def get_t_max(self):
        return len(self.b)


class LpConstraint(Constraint):
    def __init__(self, range=None, p=np.inf):
        Constraint.__init__(self)
        self.range = range
        self.p = p

    def plot(self, ax, dims, color, fc_color="None", linewidth=3, zorder=2, plot_2d=True):
        if not plot_2d:
            return self.plot3d(ax, dims, color, fc_color=fc_color, linewidth=linewidth, zorder=zorder)
        if isinstance(self.range, list) or (isinstance(self.range, np.ndarray) and self.range.ndim == 3):
            for i in range(len(self.range)):
                rect = make_rect_from_arr(self.range[i], dims, color, linewidth, fc_color, zorder=zorder)
                ax.add_patch(rect)
        else:
            rect = make_rect_from_arr(self.range, dims, color, linewidth, fc_color, zorder=zorder)
            ax.add_patch(rect)
        return [rect]
    
    def plot3d(self, ax, dims, color, fc_color="None", linewidth=1, zorder=2, plot_2d=True):
        if isinstance(self.range, list) or (isinstance(self.range, np.ndarray) and self.range.ndim == 3):
            for i in range(len(self.range)):
                rect = rect_prism(*self.range[i][dims, :], ax, color, linewidth, fc_color, zorder=zorder)
        else:
            rect = rect_prism(*self.range[dims, :], ax, color, linewidth, fc_color, zorder=zorder)
        return rect

    def get_t_max(self):
        return len(self.range)


def make_rect_from_arr(arr, dims, color, linewidth, fc_color, zorder=None):
    rect = Rectangle(
        arr[dims, 0],
        arr[dims[0], 1]
        - arr[dims[0], 0],
        arr[dims[1], 1]
        - arr[dims[1], 0],
        fc=fc_color,
        linewidth=linewidth,
        edgecolor=color,
        zorder=zorder,
    )
    return rect


def make_polytope_from_arrs(ax, A, b, color, label, zorder):
    vertices = np.stack(
        pypoman.polygon.compute_polygon_hull(
            A, b + 1e-10
        )
    )
    lines = ax.plot(
        [v[0] for v in vertices] + [vertices[0][0]],
        [v[1] for v in vertices] + [vertices[0][1]],
        color=color,
        label=label,
        zorder=zorder,
    )
    return lines
