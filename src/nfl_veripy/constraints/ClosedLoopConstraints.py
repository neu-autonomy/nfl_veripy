"""Tools for representing sets during reachability analysis."""
from __future__ import annotations

import collections
from abc import ABC
from typing import Any, Optional, TypeGuard, Union

import jax
import numpy as np
import pypoman
from matplotlib.patches import Rectangle
from nfl_veripy.utils.plot_rect_prism import rect_prism
from nfl_veripy.utils.utils import (
    CROWNMatrices,
    get_polytope_A,
    range_to_polytope,
)
from scipy.spatial import ConvexHull

import jax_verify


class Constraint(ABC):
    """Abstract class describing a set of states."""

    def to_reachable_input_objects(self):
        """Get the objects needed for forward reachability."""

    def to_fwd_reachable_output_objects(self, num_states):
        """Get the objects needed for forward reachability."""

    def get_t_max(self) -> int:
        """Get the time duration this constraint represents"""

    def add_cell(self, other: Optional[Constraint]) -> None:
        """Expand this constraint's set to include a new set."""

    def update_main_constraint_with_cells(self, overapprox: bool) -> None:
        """Set bounds of full set based on combination of all cells."""

    def add_cell_and_update_main_constraint(
        self, other: Optional[LpConstraint]
    ) -> None:
        """Insert cell into set and also update bounds of full set."""

    def to_linf(self) -> np.ndarray:
        """Get hyperrectangle bounds around set."""

    def get_area(self) -> float:
        """Get area contained by set."""

    def get_polytope(self) -> tuple[np.ndarray, np.ndarray]:
        """Get polytope bounds around set."""

    def get_constraint_at_time_index(self, i: int) -> PolytopeConstraint:
        """Get set corresponding to a single timestep."""


class PolytopeConstraint(Constraint):
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

    # other should really be
    # Union[PolytopeConstraint, MultiTimestepPolytopeConstraint]
    def add_timestep_constraint(
        self,
        other: Union[
            SingleTimestepConstraint, MultiTimestepPolytopeConstraint
        ],
    ) -> MultiTimestepPolytopeConstraint:
        if not isinstance(
            other, (PolytopeConstraint, MultiTimestepPolytopeConstraint)
        ):
            raise TypeError(
                "in add_timestep_constraint, other should be"
                " Union[PolytopeConstraint, MultiTimestepPolytopeConstraint]."
            )
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
            A=[self.A] + other.A, b=[self.b] + other.b
        )

    def to_multistep_constraint(self) -> MultiTimestepPolytopeConstraint:
        if self.A is None or self.b is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.A or"
                " self.b are None."
            )
        constraint = MultiTimestepPolytopeConstraint(A=[self.A], b=[self.b])
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


class LpConstraint(Constraint):
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
        self.main_constraint_stale = False

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
        self.add_cell(other)

        if len(self.cells) == 1:
            self.range = other.range
        else:
            self.range[:, 0] = np.minimum(self.range[:, 0], other.range[:, 0])
            self.range[:, 1] = np.maximum(self.range[:, 1], other.range[:, 1])
        self.main_constraint_stale = False

    def update_main_constraint_with_cells(self, overapprox: bool) -> None:
        if overapprox:
            # get min of all mins, get max of all maxes
            tmp = np.stack(
                [c.range for c in self.cells],
                axis=-1,
            )
            self.range = np.empty_like(self.cells[0].range)
            self.range[..., 0] = np.min(tmp[..., 0, :], axis=-1)
            self.range[..., 1] = np.max(tmp[..., 1, :], axis=-1)
        else:
            raise NotImplementedError

        self.main_constraint_stale = False

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

    # other should really be Union[LpConstraint, MultiTimestepLpConstraint]
    def add_timestep_constraint(
        self, other: Union[SingleTimestepConstraint, MultiTimestepConstraint]
    ) -> MultiTimestepLpConstraint:
        if not isinstance(other, (LpConstraint, MultiTimestepLpConstraint)):
            raise TypeError(
                "in add_timestep_constraint, other should be"
                " Union[LpConstraint, MultiTimestepLpConstraint]."
            )
        if self.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but self.range is None."
            )
        if other.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.range is None."
            )
        return MultiTimestepLpConstraint(
            range=np.vstack([np.expand_dims(self.range, 0), other.range])
        )

    def to_multistep_constraint(self) -> MultiTimestepLpConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.range is"
                " None."
            )
        return MultiTimestepLpConstraint(range=np.expand_dims(self.range, 0))

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


class MultiTimestepLpConstraint(LpConstraint):
    # range: (num_timesteps, num_states, 2)
    def __init__(
        self,
        range: Optional[np.ndarray] = None,
        p: float = np.inf,
        crown_matrices: Optional[CROWNMatrices] = None,
    ):
        super().__init__(range=range, p=p, crown_matrices=crown_matrices)

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
        for i in range(len(self.range)):
            rect = make_rect_from_arr(
                self.range[i],
                dims,
                color,
                linewidth,
                fc_color,
                ls,
                zorder=zorder,
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
        for i in range(len(self.range)):
            rect = rect_prism(
                *self.range[i][dims, :],
                ax,
                color,
                linewidth,
                fc_color,
                zorder=zorder,
            )
        return rect

    def get_t_max(self) -> int:
        if self.range is None:
            raise ValueError(
                "Can't get t_max from MultiTimestepLpConstraint, since"
                " self.range is None."
            )
        return self.range.shape[0]

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
        x_min = self.range[-1, :, 0]
        x_max = self.range[-1, :, 1]
        norm = self.p
        A_inputs = None
        b_inputs = None
        return A_inputs, b_inputs, x_max, x_min, norm

    # other should really be Union[LpConstraint, MultiTimestepLpConstraint]
    def add_timestep_constraint(
        self, other: Union[SingleTimestepConstraint, MultiTimestepConstraint]
    ) -> MultiTimestepLpConstraint:
        if not isinstance(other, (LpConstraint, MultiTimestepLpConstraint)):
            raise TypeError(
                "in add_timestep_constraint, other should be"
                " Union[LpConstraint, MultiTimestepLpConstraint]."
            )
        if other.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.range is None."
            )
        if self.range is None:
            return other.to_multistep_constraint()
        return MultiTimestepLpConstraint(
            range=np.vstack(
                [self.range, other.to_multistep_constraint().range]
            )
        )

    def get_constraint_at_time_index(self, i: int) -> LpConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to get_constraint_at_time_index, but self.range is"
                " None"
            )
        return LpConstraint(range=self.range[i, ...])

    def add_cell(self, other: Optional[MultiTimestepLpConstraint]) -> None:
        if other is None:
            return

        self.cells.append(other)
        self.main_constraint_stale = True

    def update_main_constraint_with_cells(self, overapprox: bool) -> None:
        if overapprox:
            # get min of all mins, get max of all maxes
            tmp = np.stack(
                [c.range for c in self.cells],
                axis=-1,
            )
            self.range = np.empty_like(self.cells[0].range)
            self.range[..., 0] = np.min(tmp[..., 0, :], axis=-1)
            self.range[..., 1] = np.max(tmp[..., 1, :], axis=-1)
        else:
            raise NotImplementedError

        self.main_constraint_stale = False

    def to_multistep_constraint(self) -> MultiTimestepLpConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.range is"
                " None."
            )
        return self

    def to_jittable(self):
        return JittableMultiTimestepLpConstraint(
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


class MultiTimestepPolytopeConstraint(PolytopeConstraint):
    # A: [(num_facets_0, num_states), ..., (num_facets_t, num_states)] <- num_timesteps # noqa
    # b: [(num_facets_0,), ..., (num_facets_t,)] <- num_timesteps
    def __init__(
        self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None
    ):
        super().__init__(A=A, b=b)

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

    # other should really be
    # Union[PolytopeConstraint, MultiTimestepPolytopeConstraint]
    def add_timestep_constraint(
        self, other: Union[SingleTimestepConstraint, MultiTimestepConstraint]
    ) -> MultiTimestepPolytopeConstraint:
        if not isinstance(
            other, (PolytopeConstraint, MultiTimestepPolytopeConstraint)
        ):
            raise TypeError(
                "in add_timestep_constraint, other should be"
                " Union[PolytopeConstraint, MultiTimestepPolytopeConstraint]."
            )
        if other.A is None or other.b is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.A or other.b are"
                " None."
            )
        if self.A is None or self.b is None:
            return other.to_multistep_constraint()
        constraint = MultiTimestepPolytopeConstraint(
            A=self.A + [other.A], b=self.b + [other.b]
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
        if self.range is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.range is"
                " None."
            )
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


MultiTimestepConstraint = Union[
    MultiTimestepLpConstraint, MultiTimestepPolytopeConstraint
]
SingleTimestepConstraint = Union[LpConstraint, PolytopeConstraint]

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

JittableMultiTimestepLpConstraint = collections.namedtuple(
    "JittableMultiTimestepLpConstraint",
    ["lower", "upper", "bound_type", "kwargs"],
)


def unjit_multi_timestep_lp_constraints(*inputs):
    """Replace all the jittable bounds by standard bound objects."""

    def is_jittable_constraint(b):
        return isinstance(b, JittableMultiTimestepLpConstraint)

    def unjit_bound(b):
        return next(iter(b.bound_type)).from_jittable(b)

    return jax.tree_util.tree_map(
        lambda b: unjit_bound(b) if is_jittable_constraint(b) else b,
        inputs,
        is_leaf=is_jittable_constraint,
    )


jax.tree_util.register_pytree_node(
    MultiTimestepLpConstraint,
    MultiTimestepLpConstraint._tree_flatten,
    MultiTimestepLpConstraint._tree_unflatten,
)

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


class RotatedLpConstraint(Constraint):
    def __init__(self, pose=None, W=None, theta=0, vertices=None):
        Constraint.__init__(self)
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


def create_empty_constraint(
    boundary_type: str, num_facets: Optional[int] = None
) -> SingleTimestepConstraint:
    if boundary_type == "polytope":
        if num_facets:
            return PolytopeConstraint(A=get_polytope_A(num_facets))
        return PolytopeConstraint()
    elif boundary_type == "rectangle":
        return LpConstraint()
    else:
        raise NotImplementedError


def create_empty_multi_timestep_constraint(
    boundary_type: str, num_facets: Optional[int] = None
) -> MultiTimestepConstraint:
    if boundary_type == "polytope":
        if num_facets:
            return MultiTimestepPolytopeConstraint(
                A=get_polytope_A(num_facets)
            )
        return MultiTimestepPolytopeConstraint()
    elif boundary_type == "rectangle":
        return MultiTimestepLpConstraint()
    else:
        raise NotImplementedError


def state_range_to_constraint(
    state_range: np.ndarray, boundary_type: str
) -> Constraint:
    if boundary_type == "polytope":
        A, b = range_to_polytope(state_range)
        return PolytopeConstraint(A, b)
    elif boundary_type == "rectangle":
        return LpConstraint(range=state_range, p=np.inf)
    else:
        raise NotImplementedError


def is_lp_constraint_list(xs: list[Any]) -> TypeGuard[list[LpConstraint]]:
    return all(
        isinstance(x, LpConstraint) and isinstance(x.range, np.ndarray)
        for x in xs
    )


def is_polytope_constraint_list(
    xs: list[Any],
) -> TypeGuard[list[PolytopeConstraint]]:
    return all(isinstance(x, PolytopeConstraint) for x in xs)


def is_npndarray_list(xs: list[Any]) -> TypeGuard[list[np.ndarray]]:
    return all(isinstance(x, np.ndarray) for x in xs)


def list_to_constraint(
    reachable_sets: Union[list[LpConstraint], list[PolytopeConstraint]]
) -> MultiTimestepConstraint:
    if is_lp_constraint_list(reachable_sets):
        range = [r.range for r in reachable_sets]
        assert is_npndarray_list(range)  # ensure all ranges are not None
        return MultiTimestepLpConstraint(range=np.stack(range))
    elif is_polytope_constraint_list(reachable_sets):
        As = [r.A for r in reachable_sets]
        bs = [r.b for r in reachable_sets]
        assert is_npndarray_list(As)  # ensure all As are not None
        assert is_npndarray_list(bs)  # ensure all bs are not None
        return MultiTimestepPolytopeConstraint(A=np.stack(As), b=np.stack(bs))
    else:
        raise ValueError(
            "reachable_sets list contains constraints with None range or A, b."
        )
