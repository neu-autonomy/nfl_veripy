import numpy as np
import pypoman
from matplotlib.patches import Rectangle

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

    def plot(self, ax, dims, color, fc_color="None", linewidth=3, label=None):
        # TODO: this doesn't use the computed input_dims...

        if isinstance(self.A, list):
            # Backward reachability
            # input_constraint.A will be a list
            # of polytope facets, whose union is the estimated
            # backprojection set

            for i in range(len(self.A)):
                make_polytope_from_arrs(ax, self.A[i], self.b[i], color, label)

        else:
            # Forward reachability
            for i in range(len(self.b)):
                make_polytope_from_arrs(ax, self.A, self.b[i], color, label)

    def get_t_max(self):
        return len(self.b)


class LpConstraint(Constraint):
    def __init__(self, range=None, p=np.inf):
        Constraint.__init__(self)
        self.range = range
        self.p = p

    def plot(self, ax, dims, color, fc_color="None", linewidth=3):
        if isinstance(self.range, list) or (isinstance(self.range, np.ndarray) and self.range.ndim == 3):
            for i in range(len(self.range)):
                rect = make_rect_from_arr(self.range[i], dims, color, linewidth, fc_color)
                ax.add_patch(rect)
        else:
            rect = make_rect_from_arr(self.range, dims, color, linewidth, fc_color)
            ax.add_patch(rect)
        return rect

    def get_t_max(self):
        return len(self.range)


def make_rect_from_arr(arr, dims, color, linewidth, fc_color):
    rect = Rectangle(
        arr[dims, 0],
        arr[dims[0], 1]
        - arr[dims[0], 0],
        arr[dims[1], 1]
        - arr[dims[1], 0],
        fc=fc_color,
        linewidth=linewidth,
        edgecolor=color,
    )
    return rect


def make_polytope_from_arrs(ax, A, b, color, label):
    vertices = np.stack(
        pypoman.polygon.compute_polygon_hull(
            A, b + 1e-10
        )
    )
    ax.plot(
        [v[0] for v in vertices] + [vertices[0][0]],
        [v[1] for v in vertices] + [vertices[0][1]],
        color=color,
        label=label,
    )
