import numpy as np
import pypoman
from matplotlib.patches import Rectangle
from nn_closed_loop.utils.plot_rect_prism import rect_prism
from nn_closed_loop.utils.utils import range_to_polytope


class Constraint:
    def __init__(self):
        pass

    def to_fwd_reachable_input_objects(self):
        raise NotImplementedError

    def to_fwd_reachable_output_objects(self, num_states):
        raise NotImplementedError


class PolytopeConstraint(Constraint):
    def __init__(self, A=None, b=None):
        Constraint.__init__(self)
        self.A = A
        self.b = b

    def __add__(self, x):
        if x is None:
            return
        self.A.append(x.A)
        self.b.append(x.b)

    def add_cell(self, output_constraint):
        reachable_set_this_cell = [o.b for o in output_constraint]
        if self.b is None:
            self.b = np.stack(reachable_set_this_cell)

        tmp = np.dstack(
            [self.b, np.stack(reachable_set_this_cell)]
        )
        self.b = np.max(tmp, axis=-1)
        return reachable_set_this_cell

    def get_cell(self, input_range):
        # This is a disaster hack to partition polytopes
        A_rect, b_rect = range_to_polytope(input_range)
        rectangle_verts = pypoman.polygon.compute_polygon_hull(
            A_rect, b_rect
        )
        input_polytope_verts = pypoman.polygon.compute_polygon_hull(
            self.A, self.b
        )
        partition_verts = pypoman.intersection.intersect_polygons(
            input_polytope_verts, rectangle_verts
        )
        (
            A_inputs_,
            b_inputs_,
        ) = pypoman.duality.compute_polytope_halfspaces(
            partition_verts
        )
        constraint = self.__class__(
            A_inputs_, b_inputs_
        )
        return constraint

    def to_range(self):

        # only used to compute slope in non-closedloop manner...
        input_polytope_verts = pypoman.duality.compute_polytope_vertices(
            self.A, self.b
        )
        input_range = np.empty((self.A.shape[1], 2))
        input_range[:, 0] = np.min(np.stack(input_polytope_verts), axis=0)
        input_range[:, 1] = np.max(np.stack(input_polytope_verts), axis=0)
        return input_range

    def set_bound(self, i, max_value, min_value):
        self.b[i] = max_value

    def to_fwd_reachable_input_objects(self):
        A_inputs = self.A
        b_inputs = self.b

        # Get bounds on each state from A_inputs, b_inputs
        try:
            vertices = np.stack(
                pypoman.compute_polytope_vertices(A_inputs, b_inputs)
            )
        except:
            # Sometimes get arithmetic error... this may fix it
            vertices = np.stack(
                pypoman.compute_polytope_vertices(
                    A_inputs, b_inputs + 1e-6
                )
            )
        x_max = np.max(vertices, 0)
        x_min = np.min(vertices, 0)
        norm = np.inf
        return A_inputs, b_inputs, x_max, x_min, norm

    def to_fwd_reachable_output_objects(self, num_states):
        A_out = self.A
        num_facets = A_out.shape[0]
        self.b = np.zeros((num_facets))
        return A_out, num_facets

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

    def plot(self, ax, dims, color, fc_color="None", linewidth=1.5, label=None, zorder=2, plot_2d=True, ls='-'):
        if not plot_2d:
            raise NotImplementedError
            return self.plot3d(ax, dims, color, fc_color=fc_color, linewidth=linewidth, zorder=zorder)

        # TODO: this doesn't use the computed input_dims...

        if linewidth != 2.5:
            linewidth = 1.5

        lines = []

        if isinstance(self.A, list):
            # Backward reachability
            # input_constraint.A will be a list
            # of polytope facets, whose union is the estimated
            # backprojection set

            for i in range(len(self.A)):
                line = make_polytope_from_arrs(ax, self.A[i], self.b[i], color, label, zorder, ls, linewidth)
                lines += line

        else:
            # Forward reachability
            if isinstance(self.b, np.ndarray) and self.b.ndim == 1:
                line = make_polytope_from_arrs(ax, self.A, self.b, color, label, zorder, ls, linewidth)
                lines += line
            else:
                for i in range(len(self.b)):
                    line = make_polytope_from_arrs(ax, self.A, self.b[i], color, label, zorder, ls, linewidth)
                    lines += line

        return lines

    def get_t_max(self):
        return len(self.b)


class LpConstraint(Constraint):
    def __init__(self, range=None, p=np.inf, crown_matrices=None):
        Constraint.__init__(self)
        self.range = range
        self.p = p
        self.crown_matrices = crown_matrices

    def __add__(self, other):
        if other is None:
            return self
        self.range[:, 0] = np.minimum(other.range[:, 0], self.range[:, 0])
        self.range[:, 1] = np.maximum(other.range[:, 1], self.range[:, 1])
        return self

    def set_bound(self, i, max_value, min_value):
        self.range[i, 0] = min_value
        self.range[i, 1] = max_value

    def to_range(self):
        input_range = self.range
        return input_range

    def get_cell(self, input_range):
        return self.__class__(range=input_range, p=self.p)

    def add_cell(self, output_constraint):
        reachable_set_this_cell = [o.range for o in output_constraint]
        if self.range is None:
            self.range = np.stack(reachable_set_this_cell)

        tmp = np.stack(
            [self.range, np.stack(reachable_set_this_cell)],
            axis=-1,
        )

        self.range[..., 0] = np.min(tmp[..., 0, :], axis=-1)
        self.range[..., 1] = np.max(tmp[..., 1, :], axis=-1)

        return np.stack(reachable_set_this_cell)

    def to_fwd_reachable_input_objects(self):
        x_min = self.range[..., 0]
        x_max = self.range[..., 1]
        norm = self.p
        A_inputs = None
        b_inputs = None
        return A_inputs, b_inputs, x_max, x_min, norm

    def to_fwd_reachable_output_objects(self, num_states):
        A_out = np.eye(num_states)
        num_facets = A_out.shape[0]
        self.range = np.zeros((num_facets, 2))
        return A_out, num_facets

    def plot(self, ax, dims, color, fc_color="None", linewidth=3, zorder=2, plot_2d=True, ls='-'):
        if not plot_2d:
            return self.plot3d(ax, dims, color, fc_color=fc_color, linewidth=linewidth, zorder=zorder)
        if isinstance(self.range, list) or (isinstance(self.range, np.ndarray) and self.range.ndim == 3):
            for i in range(len(self.range)):
                rect = make_rect_from_arr(self.range[i], dims, color, linewidth, fc_color, ls, zorder=zorder)
                ax.add_patch(rect)
        else:
            rect = make_rect_from_arr(self.range, dims, color, linewidth, fc_color, ls, zorder=zorder)
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


def make_rect_from_arr(arr, dims, color, linewidth, fc_color, ls, zorder=None):
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
        linestyle=ls,
    )
    return rect


def make_polytope_from_arrs(ax, A, b, color, label, zorder, ls, linewidth=1.5):
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
        ls=ls,
        linewidth=linewidth
    )
    return lines
