"""Tools for representing sets during reachability analysis."""
from __future__ import annotations

import collections
from typing import Optional, Union

import jax
import jax_verify
import numpy as np
import pypoman
from scipy.spatial import ConvexHull

from nfl_veripy.utils.utils import CROWNMatrices, range_to_polytope

from .constraint_utils import make_polytope_from_arrs


class PolytopeConstraint:
    """Represents single timestep's set of states with a H-rep polytope."""

    def __init__(
        self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.A = A
        self.b = b
        self.cells: list[PolytopeConstraint] = []
        self.crown_matrices: Optional[CROWNMatrices] = None
        self.main_constraint_stale = False

    def update_main_constraint_with_cells(self, overapprox: bool) -> None:
        if len(self.cells) == 0:
            raise ValueError("Can't update because self.cells is empty.")
        elif len(self.cells) == 1:
            self.A = self.cells[0].A
            self.b = self.cells[0].b
            self.main_constraint_stale = False
        else:
            if overapprox:
                # TODO: compute all vertices, then get conv hull using pypoman
                raise NotImplementedError
            else:
                # Simplest under-approximation of union of polytopes is one of
                # those polytopes :/
                # TODO: compute a better under-approximation :)
                self.A = self.cells[0].A
                self.b = self.cells[0].b
                self.main_constraint_stale = False

    def add_cell(self, other: Optional[PolytopeConstraint]) -> None:
        if other is None:
            return

        self.cells.append(other)
        self.main_constraint_stale = True

    def get_cell(self, input_range: np.ndarray) -> PolytopeConstraint:
        # This is a disaster hack to partition polytopes
        A_rect, b_rect = range_to_polytope(input_range)
        rectangle_verts = pypoman.polygon.compute_polygon_hull(A_rect, b_rect)
        input_polytope_verts = pypoman.polygon.compute_polygon_hull(
            self.A, self.b
        )
        partition_verts = pypoman.intersection.intersect_polygons(
            input_polytope_verts, rectangle_verts
        )
        (
            A_inputs_,
            b_inputs_,
        ) = pypoman.duality.compute_polytope_halfspaces(partition_verts)
        constraint = self.__class__(A_inputs_, b_inputs_)
        return constraint

    def to_range(self) -> np.ndarray:
        if self.A is None or self.b is None:
            raise ValueError(
                "Can't convert PolytopeConstraint to range, since self.A or"
                " self.b are None."
            )

        # only used to compute slope in non-closedloop manner...
        input_polytope_verts = pypoman.duality.compute_polytope_vertices(
            self.A, self.b
        )
        input_range = np.empty((self.A.shape[1], 2))
        input_range[:, 0] = np.min(np.stack(input_polytope_verts), axis=0)
        input_range[:, 1] = np.max(np.stack(input_polytope_verts), axis=0)
        return input_range

    def set_bound(self, i: int, max_value: float, min_value: float) -> None:
        if self.b is None:
            raise ValueError(
                "Can't set bound on PolytopeConstraint, since self.b is None."
            )
        self.b[i] = max_value

    def to_reachable_input_objects(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if self.A is None or self.b is None:
            raise ValueError(
                "Can't convert PolytopeConstraint to"
                " to_reachable_input_objects, since self.A or self.b are None."
            )

        A_inputs = self.A
        b_inputs = self.b

        # Get bounds on each state from A_inputs, b_inputs
        try:
            vertices_list = pypoman.compute_polytope_vertices(
                A_inputs, b_inputs
            )  # type: list[np.ndarray]
        except Exception:
            # Sometimes get arithmetic error... this may fix it
            vertices_list = pypoman.compute_polytope_vertices(
                A_inputs, b_inputs + 1e-6
            )
        vertices = np.stack(vertices_list)
        x_max = np.max(vertices, 0)  # type: np.ndarray
        x_min = np.min(vertices, 0)  # type: np.ndarray
        norm = np.inf
        return A_inputs, b_inputs, x_max, x_min, norm

    def to_fwd_reachable_output_objects(
        self, num_states: int
    ) -> tuple[np.ndarray, int]:
        if self.A is None:
            raise ValueError(
                "Can't convert PolytopeConstraint to"
                " to_fwd_reachable_output_objects, since self.A is None."
            )
        A_out = self.A
        num_facets = A_out.shape[0]
        self.b = np.zeros((num_facets))
        return A_out, num_facets

    def to_linf(self) -> np.ndarray:
        if isinstance(self.A, list):
            # Mainly for backreachability, return a list of ranges if
            # the constraint contains a list of polytopes
            ranges = []
            for A, b in zip(self.A, self.b):
                vertices = np.stack(
                    pypoman.duality.compute_polytope_vertices(A, b)
                )
                ranges.append(
                    np.dstack(
                        [np.min(vertices, axis=0), np.max(vertices, axis=0)]
                    )[0]
                )
        else:
            vertices = np.stack(
                pypoman.duality.compute_polytope_vertices(self.A, self.b)
            )
            ranges = np.dstack(
                [np.min(vertices, axis=0), np.max(vertices, axis=0)]
            )[0]
        return ranges

    def plot(
        self,
        ax,
        dims,
        color,
        fc_color="None",
        linewidth=1.5,
        label=None,
        zorder=2,
        plot_2d=True,
        ls="-",
    ):
        if not plot_2d:
            raise NotImplementedError

        # TODO: this doesn't use the computed input_dims...

        if linewidth != 2.5:
            linewidth = 1.5

        lines = []

        if isinstance(self.A, list):
            # Backward reachability
            # input_constraint.A will be a list
            # of polytope facets, whose union is the estimated
            # backprojection set

            for A, b in zip(self.A, self.b):
                line = make_polytope_from_arrs(
                    ax,
                    A,
                    b,
                    color,
                    label,
                    zorder,
                    ls,
                    linewidth,
                )
                lines += line

        else:
            # Forward reachability
            if isinstance(self.b, np.ndarray) and self.b.ndim == 1:
                line = make_polytope_from_arrs(
                    ax, self.A, self.b, color, label, zorder, ls, linewidth
                )
                lines += line
            else:
                for A, b in zip(self.A, self.b):
                    line = make_polytope_from_arrs(
                        ax,
                        A,
                        b,
                        color,
                        label,
                        zorder,
                        ls,
                        linewidth,
                    )
                    lines += line

        return lines

    def add_timestep_constraint(
        self,
        other: Union[PolytopeConstraint, MultiTimestepPolytopeConstraint],
    ) -> MultiTimestepPolytopeConstraint:
        if self.A is None or self.b is None:
            raise ValueError(
                "Trying to add_timestep_constraint but self.A or self.b are"
                " None."
            )
        if other.A is None or other.b is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.A or other.b are"
                " None."
            )
        return MultiTimestepPolytopeConstraint(
            constraints=[self] + other.to_multistep_constraint().constraints
        )

    def to_multistep_constraint(self) -> MultiTimestepPolytopeConstraint:
        if self.A is None or self.b is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.A or"
                " self.b are None."
            )
        constraint = MultiTimestepPolytopeConstraint(constraints=[self])
        constraint.cells = [
            cell.to_multistep_constraint() for cell in self.cells
        ]
        return constraint

    def get_area(self) -> float:
        estimated_verts = pypoman.polygon.compute_polygon_hull(self.A, self.b)
        estimated_hull = ConvexHull(estimated_verts)
        estimated_area = estimated_hull.volume
        return estimated_area

    def get_polytope(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.A is not None
        assert self.b is not None
        return self.A, self.b

    def get_constraint_at_time_index(self, i: int) -> PolytopeConstraint:
        return self

    def to_jittable(self):
        return JittablePolytopeConstraint(
            self.A, self.b, {jax_verify.IntervalBound: None}, {}
        )

    @classmethod
    def from_jittable(cls, jittable_constraint):
        return cls(
            range=np.array(
                [jittable_constraint.lower, jittable_constraint.upper]
            )
        )

    def _tree_flatten(self):
        children = (self.A, self.b)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


JittablePolytopeConstraint = collections.namedtuple(
    "JittablePolytopeConstraint", ["A", "b", "bound_type", "kwargs"]
)


def unjit_polytope_constraints(*inputs):
    """Replace all the jittable bounds by standard bound objects."""

    def is_jittable_constraint(b):
        return isinstance(b, JittablePolytopeConstraint)

    def unjit_bound(b):
        return next(iter(b.bound_type)).from_jittable(b)

    return jax.tree_util.tree_map(
        lambda b: unjit_bound(b) if is_jittable_constraint(b) else b,
        inputs,
        is_leaf=is_jittable_constraint,
    )


jax.tree_util.register_pytree_node(
    PolytopeConstraint,
    PolytopeConstraint._tree_flatten,
    PolytopeConstraint._tree_unflatten,
)


class MultiTimestepPolytopeConstraint:
    # A: [(num_facets_0, num_states), ..., (num_facets_t, num_states)] <- num_timesteps # noqa
    # b: [(num_facets_0,), ..., (num_facets_t,)] <- num_timesteps
    def __init__(
        self,
        constraints: list[PolytopeConstraint] = [],
        # crown_matrices: Optional[CROWNMatrices] = None,
    ):
        self.constraints = constraints
        self.cells: list[MultiTimestepPolytopeConstraint] = []

    @property
    def A(self) -> list[np.ndarray]:
        return [
            constraint.A
            for constraint in self.constraints
            if constraint.A is not None
        ]

    @property
    def b(self) -> list[np.ndarray]:
        return [
            constraint.b
            for constraint in self.constraints
            if constraint.b is not None
        ]

    def plot(
        self,
        ax,
        dims,
        color,
        fc_color="None",
        linewidth=1.5,
        label=None,
        zorder=2,
        plot_2d=True,
        ls="-",
    ):
        if not plot_2d:
            raise NotImplementedError

        lines = []

        for i in range(len(self.A)):
            line = make_polytope_from_arrs(
                ax, self.A[i], self.b[i], color, label, zorder, ls, linewidth
            )
            lines += line

        return lines

    def get_t_max(self) -> int:
        if self.A is None:
            raise ValueError(
                "Can't get t_max from MultiTimestepPolytopeConstraint, since"
                " self.A is None."
            )
        return len(self.A)

    def to_reachable_input_objects(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if self.A is None:
            raise ValueError(
                "Can't convert PolytopeConstraint to"
                " to_reachable_input_objects, since self.A is None."
            )
        if self.b is None:
            raise ValueError(
                "Can't convert PolytopeConstraint to"
                " to_reachable_input_objects, since self.b is None."
            )

        A_inputs = self.A[0]
        b_inputs = self.b[0]

        # Get bounds on each state from A_inputs, b_inputs
        try:
            vertices_list = pypoman.compute_polytope_vertices(
                A_inputs, b_inputs
            )  # type: list[np.ndarray]
        except Exception:
            # Sometimes get arithmetic error... this may fix it
            vertices_list = pypoman.compute_polytope_vertices(
                A_inputs, b_inputs + 1e-6
            )
        vertices = np.stack(vertices_list)
        x_max = np.max(vertices, 0)  # type: np.ndarray
        x_min = np.min(vertices, 0)  # type: np.ndarray
        norm = np.inf
        return A_inputs, b_inputs, x_max, x_min, norm

    def add_timestep_constraint(
        self, other: Union[PolytopeConstraint, MultiTimestepPolytopeConstraint]
    ) -> MultiTimestepPolytopeConstraint:
        if other.A is None or other.b is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.A or other.b are"
                " None."
            )
        if self.A is None or self.b is None:
            return other.to_multistep_constraint()
        constraint = MultiTimestepPolytopeConstraint(
            constraints=self.constraints
            + other.to_multistep_constraint().constraints
        )
        if len(self.cells) == 0:
            # We're adding cells to a constraint with no cells, so all those
            # new cells should use the constraint's existing value for its
            # first N timesteps
            for cell in other.cells:
                constraint.add_cell(constraint.add_timestep_constraint(cell))
        else:
            # TODO: Not clear how one should combine self.cells and other.cells
            raise NotImplementedError
        return constraint

    def get_constraint_at_time_index(self, i: int) -> PolytopeConstraint:
        if self.A is None or self.b is None:
            raise ValueError(
                "Trying to get_constraint_at_time_index, but self.A or self.b"
                " are None"
            )
        constraint = PolytopeConstraint(A=self.A[i], b=self.b[i])
        for cell in self.cells:
            constraint.add_cell(cell.get_constraint_at_time_index(i))
        return constraint

    def add_cell(
        self, other: Optional[MultiTimestepPolytopeConstraint]
    ) -> None:
        if other is None:
            return

        self.cells.append(other)
        self.main_constraint_stale = True

    def to_multistep_constraint(self) -> MultiTimestepPolytopeConstraint:
        return self

    def to_jittable(self):
        return JittableMultiTimestepPolytopeConstraint(
            self.range[..., 0],
            self.range[..., 1],
            {jax_verify.IntervalBound: None},
            {},
        )

    @classmethod
    def from_jittable(cls, jittable_constraint):
        return cls(
            range=np.array(
                [jittable_constraint.lower, jittable_constraint.upper]
            )
        )

    def _tree_flatten(self):
        children = (self.range,)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


JittableMultiTimestepPolytopeConstraint = collections.namedtuple(
    "JittableMultiTimestepPolytopeConstraint",
    ["A", "b", "bound_type", "kwargs"],
)


def unjit_multi_timestep_polytope_constraints(*inputs):
    """Replace all the jittable bounds by standard bound objects."""

    def is_jittable_constraint(b):
        return isinstance(b, JittableMultiTimestepPolytopeConstraint)

    def unjit_bound(b):
        return next(iter(b.bound_type)).from_jittable(b)

    return jax.tree_util.tree_map(
        lambda b: unjit_bound(b) if is_jittable_constraint(b) else b,
        inputs,
        is_leaf=is_jittable_constraint,
    )


jax.tree_util.register_pytree_node(
    MultiTimestepPolytopeConstraint,
    MultiTimestepPolytopeConstraint._tree_flatten,
    MultiTimestepPolytopeConstraint._tree_unflatten,
)
