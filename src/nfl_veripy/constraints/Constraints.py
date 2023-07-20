"""Tools for representing sets during reachability analysis."""
from __future__ import annotations

import collections
from typing import Optional, Union

import jax
import jax_verify
import numpy as np
import pypoman
from scipy.spatial import ConvexHull

from nfl_veripy.constraints.constraint_utils import (
    make_polytope_from_arrs,
    make_rect_from_arr,
)
from nfl_veripy.utils.plot_rect_prism import rect_prism
from nfl_veripy.utils.utils import CROWNMatrices, range_to_polytope


class LpConstraint:
    """Represents single timestep's set of states with an lp-ball."""

    def __init__(
        self,
        range: Optional[np.ndarray] = None,
        p: float = np.inf,
        crown_matrices: Optional[CROWNMatrices] = None,
    ):
        super().__init__()
        self.range = range
        self.p = p
        self.crown_matrices = crown_matrices
        self.cells = []  # type: list[LpConstraint]
        self.main_constraint_stale: bool = False
        self.is_infeasible: bool = False

    def set_bound(self, i: int, max_value: float, min_value: float) -> None:
        if self.range is None:
            raise ValueError(
                "Can't set bound on LpConstraint, since self.range is None."
            )
        self.range[i, 0] = min_value
        self.range[i, 1] = max_value

    def to_range(self) -> np.ndarray:
        """Get axis-aligned hyperrectangle around set."""
        if self.range is None:
            raise ValueError("Won't return range, since self.range is None.")
        return self.range

    def get_cell(self, input_range: np.ndarray) -> LpConstraint:
        return self.__class__(range=input_range, p=self.p)

    def add_cell(self, other: Optional[LpConstraint]) -> None:
        if other is None:
            return

        self.cells.append(other)
        self.main_constraint_stale = True

    def add_cell_and_update_main_constraint(
        self, other: Optional[LpConstraint]
    ) -> None:
        """Insert cell into set and also update bounds of full set."""
        assert self.main_constraint_stale is False
        if other is None:
            return
        self.add_cell(other)
        if other.range is None:
            # No need to update main constraint
            self.main_constraint_stale = False
            return

        if len(self.cells) == 1 or self.range is None:
            self.range = other.range
        else:
            self.range[:, 0] = np.minimum(self.range[:, 0], other.range[:, 0])
            self.range[:, 1] = np.maximum(self.range[:, 1], other.range[:, 1])
        self.main_constraint_stale = False

    def update_main_constraint_with_cells(self, overapprox: bool) -> None:
        if overapprox:
            # get min of all mins, get max of all maxes
            tmp = np.stack(
                [c.range for c in self.cells if c.range is not None],
                axis=-1,
            )
            self.range = np.empty_like(self.cells[0].range)
            self.range[..., 0] = np.min(tmp[..., 0, :], axis=-1)
            self.range[..., 1] = np.max(tmp[..., 1, :], axis=-1)
        else:
            raise NotImplementedError

        self.main_constraint_stale = False

    def get_vertices(self) -> np.ndarray:
        Abp, bbp = range_to_polytope(self.range)
        return np.stack(pypoman.polygon.compute_polygon_hull(Abp, bbp))

    def to_reachable_input_objects(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        float,
    ]:
        if self.range is None:
            raise ValueError(
                "Can't convert LpConstraint to reachable_input_objects, since"
                " self.range is None."
            )
        x_min = self.range[..., 0]
        x_max = self.range[..., 1]
        norm = self.p
        A_inputs = None
        b_inputs = None
        return A_inputs, b_inputs, x_max, x_min, norm

    def to_fwd_reachable_output_objects(
        self, num_states: int
    ) -> tuple[np.ndarray, int]:
        A_out = np.eye(num_states)
        num_facets = A_out.shape[0]
        self.range = np.zeros((num_facets, 2))
        return A_out, num_facets

    def plot(
        self,
        ax,
        dims,
        color,
        fc_color="None",
        linewidth=3,
        zorder=2,
        plot_2d=True,
        ls="-",
    ):
        if not plot_2d:
            return self.plot3d(
                ax,
                dims,
                color,
                fc_color=fc_color,
                linewidth=linewidth,
                zorder=zorder,
            )
        rect = make_rect_from_arr(
            self.range, dims, color, linewidth, fc_color, ls, zorder=zorder
        )
        ax.add_patch(rect)
        return [rect]

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
        rect = rect_prism(
            *self.range[dims, :], ax, color, linewidth, fc_color, zorder=zorder
        )
        return rect

    def add_timestep_constraint(
        self, other: Union[SingleTimestepConstraint, MultiTimestepConstraint]
    ) -> MultiTimestepConstraint:
        constraints = [self] + other.to_multistep_constraint().constraints
        return MultiTimestepConstraint(constraints=constraints)

    def to_multistep_constraint(self) -> MultiTimestepConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.range is"
                " None."
            )
        return MultiTimestepConstraint(constraints=[self])

    def get_area(self) -> float:
        Abp, bbp = range_to_polytope(self.range)
        estimated_verts = pypoman.polygon.compute_polygon_hull(Abp, bbp)
        estimated_hull = ConvexHull(estimated_verts)
        estimated_area = estimated_hull.volume
        return estimated_area

    def get_polytope(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.range is not None
        num_states = self.range.shape[0]
        A = np.vstack([np.eye(num_states), -np.eye(num_states)])
        b = np.hstack([self.range[:, 1], -self.range[:, 0]])
        return A, b

    def get_constraint_at_time_index(self, i: int) -> LpConstraint:
        return self

    def to_jittable(self):
        return JittableLpConstraint(
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


JittableLpConstraint = collections.namedtuple(
    "JittableLpConstraint", ["lower", "upper", "bound_type", "kwargs"]
)


def unjit_lp_constraints(*inputs):
    """Replace all the jittable bounds by standard bound objects."""

    def is_jittable_constraint(b):
        return isinstance(b, JittableLpConstraint)

    def unjit_bound(b):
        return next(iter(b.bound_type)).from_jittable(b)

    return jax.tree_util.tree_map(
        lambda b: unjit_bound(b) if is_jittable_constraint(b) else b,
        inputs,
        is_leaf=is_jittable_constraint,
    )


jax.tree_util.register_pytree_node(
    LpConstraint, LpConstraint._tree_flatten, LpConstraint._tree_unflatten
)


class MultiTimestepConstraint:
    # range: (num_timesteps, num_states, 2)
    def __init__(
        self,
        constraints: list[SingleTimestepConstraint] = [],
        # crown_matrices: Optional[CROWNMatrices] = None,
    ):
        self.constraints = constraints
        self.cells: list[MultiTimestepConstraint] = []

    @property
    def range(self) -> np.ndarray:
        return np.array([constraint.range for constraint in self.constraints])

    def plot(
        self,
        ax,
        dims,
        color,
        fc_color="None",
        linewidth=3,
        zorder=2,
        plot_2d=True,
        ls="-",
    ):
        if not plot_2d:
            return self.plot3d(
                ax,
                dims,
                color,
                fc_color=fc_color,
                linewidth=linewidth,
                zorder=zorder,
            )
        for constraint in self.constraints:
            rect = constraint.plot(
                ax, dims, color, fc_color, linewidth, zorder=zorder
            )
        return [rect]

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
        for i in range(len(self.range)):
            rect = rect_prism(
                *self.constraints[i].range[dims, :],
                ax,
                color,
                linewidth,
                fc_color,
                zorder=zorder,
            )
        return rect

    def get_t_max(self) -> int:
        return len(self.constraints)

    def to_reachable_input_objects(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        float,
    ]:
        assert len(self.constraints) > 0
        last_constraint = self.constraints[-1]
        if last_constraint.range is None:
            raise ValueError(
                "Can't convert LpConstraint to reachable_input_objects, since"
                " self.range is None."
            )
        x_min = last_constraint.range[:, 0]
        x_max = last_constraint.range[:, 1]
        norm = last_constraint.p
        A_inputs = None
        b_inputs = None
        return A_inputs, b_inputs, x_max, x_min, norm

    def add_timestep_constraint(
        self, other: Union[SingleTimestepConstraint, MultiTimestepConstraint]
    ) -> MultiTimestepConstraint:
        if other.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.range is None."
            )
        if self.range is None:
            return other.to_multistep_constraint()
        return MultiTimestepConstraint(
            constraints=self.constraints
            + [
                constraint
                for constraint in other.to_multistep_constraint().constraints
            ]
        )

    def get_constraint_at_time_index(self, i: int) -> SingleTimestepConstraint:
        return self.constraints[i]

    def add_cell(self, other: Optional[MultiTimestepConstraint]) -> None:
        if other is None:
            return

        self.cells.append(other)
        self.main_constraint_stale = True

    def update_main_constraint_with_cells(self, overapprox: bool) -> None:
        if overapprox:
            # get min of all mins, get max of all maxes
            tmp = np.stack(
                [c.range for c in self.cells if c.range is not None],
                axis=-1,
            )
            ranges = np.empty_like(self.cells[0].range)
            ranges[..., 0] = np.min(tmp[..., 0, :], axis=-1)
            ranges[..., 1] = np.max(tmp[..., 1, :], axis=-1)
            self.constraints = []
            for t, range in enumerate(ranges):
                self.constraints.append(LpConstraint(range=range))

        else:
            raise NotImplementedError

        self.main_constraint_stale = False

    def to_multistep_constraint(self) -> MultiTimestepConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.range is"
                " None."
            )
        return self

    def to_jittable(self):
        return JittableMultiTimestepConstraint(
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


JittableMultiTimestepConstraint = collections.namedtuple(
    "JittableMultiTimestepConstraint",
    ["lower", "upper", "bound_type", "kwargs"],
)


def unjit_multi_timestep_constraints(*inputs):
    """Replace all the jittable bounds by standard bound objects."""

    def is_jittable_constraint(b):
        return isinstance(b, JittableMultiTimestepConstraint)

    def unjit_bound(b):
        return next(iter(b.bound_type)).from_jittable(b)

    return jax.tree_util.tree_map(
        lambda b: unjit_bound(b) if is_jittable_constraint(b) else b,
        inputs,
        is_leaf=is_jittable_constraint,
    )


jax.tree_util.register_pytree_node(
    MultiTimestepConstraint,
    MultiTimestepConstraint._tree_flatten,
    MultiTimestepConstraint._tree_unflatten,
)


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
        self.main_constraint_stale: bool = False
        self.is_infeasible: bool = False

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

    @property
    def range(self) -> np.ndarray:
        return self.to_range()

    @property
    def p(self) -> float:
        return np.inf

    def get_vertices(self) -> np.ndarray:
        return np.stack(
            pypoman.duality.compute_polytope_vertices(self.A, self.b)
        )

    def to_range(self) -> np.ndarray:
        if self.A is None or self.b is None:
            raise ValueError(
                "Can't convert PolytopeConstraint to range, since self.A or"
                " self.b are None."
            )

        # only used to compute slope in non-closedloop manner...
        input_polytope_verts = self.get_vertices()
        input_range = np.empty((self.A.shape[1], 2))
        input_range[:, 0] = np.min(input_polytope_verts, axis=0)
        input_range[:, 1] = np.max(input_polytope_verts, axis=0)
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
        other: Union[SingleTimestepConstraint, MultiTimestepConstraint],
    ) -> MultiTimestepConstraint:
        constraints = [self] + other.to_multistep_constraint().constraints
        return MultiTimestepConstraint(constraints=constraints)

    def to_multistep_constraint(self) -> MultiTimestepConstraint:
        if self.A is None or self.b is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.A or"
                " self.b are None."
            )
        constraint = MultiTimestepConstraint(constraints=[self])
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


SingleTimestepConstraint = Union[LpConstraint, PolytopeConstraint]
JittableSingleTimestepConstraint = Union[
    JittableLpConstraint, JittablePolytopeConstraint
]
