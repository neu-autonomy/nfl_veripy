"""Tools for representing sets during reachability analysis."""
from __future__ import annotations

import collections
from typing import Optional, Union

import jax
import jax_verify
import numpy as np
import pypoman
from scipy.spatial import ConvexHull

from nfl_veripy.constraints.constraint_utils import make_rect_from_arr
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
        self, other: Union[LpConstraint, MultiTimestepLpConstraint]
    ) -> MultiTimestepLpConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but self.range is None."
            )
        if other.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.range is None."
            )
        return MultiTimestepLpConstraint(
            constraints=[self] + other.to_multistep_constraint().constraints
        )

    def to_multistep_constraint(self) -> MultiTimestepLpConstraint:
        if self.range is None:
            raise ValueError(
                "Trying to convert to multistep constraint but self.range is"
                " None."
            )
        return MultiTimestepLpConstraint(constraints=[self])

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


class MultiTimestepLpConstraint:
    # range: (num_timesteps, num_states, 2)
    def __init__(
        self,
        constraints: list[LpConstraint] = [],
        # crown_matrices: Optional[CROWNMatrices] = None,
    ):
        self.constraints = constraints
        self.cells: list[MultiTimestepLpConstraint] = []

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
        for i in range(len(self.range)):
            rect = make_rect_from_arr(
                self.constraints[i].range,
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
        self, other: Union[LpConstraint, MultiTimestepLpConstraint]
    ) -> MultiTimestepLpConstraint:
        if other.range is None:
            raise ValueError(
                "Trying to add_timestep_constraint but other.range is None."
            )
        if self.range is None:
            return other.to_multistep_constraint()
        return MultiTimestepLpConstraint(
            constraints=self.constraints
            + [
                constraint
                for constraint in other.to_multistep_constraint().constraints
            ]
        )

    def get_constraint_at_time_index(self, i: int) -> LpConstraint:
        return LpConstraint(range=self.constraints[i].range)

    def add_cell(self, other: Optional[MultiTimestepLpConstraint]) -> None:
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


# class MultiTimestepLpConstraint:
#     # range: (num_timesteps, num_states, 2)
#     def __init__(
#         self,
#         constraints: list[LpConstraint] = [],
#         # crown_matrices: Optional[CROWNMatrices] = None,
#     ):
#         self.constraints = constraints
#         self.cells: list[MultiTimestepLpConstraint] = []

#     @property
#     def range(self) -> np.ndarray:
#         return np.array([constraint.range for constraint in self.constraints])

#     def plot(
#         self,
#         ax,
#         dims,
#         color,
#         fc_color="None",
#         linewidth=3,
#         zorder=2,
#         plot_2d=True,
#         ls="-",
#     ):
#         if not plot_2d:
#             return self.plot3d(
#                 ax,
#                 dims,
#                 color,
#                 fc_color=fc_color,
#                 linewidth=linewidth,
#                 zorder=zorder,
#             )
#         for i in range(len(self.range)):
#             rect = make_rect_from_arr(
#                 self.constraints[i].range,
#                 dims,
#                 color,
#                 linewidth,
#                 fc_color,
#                 ls,
#                 zorder=zorder,
#             )
#             ax.add_patch(rect)
#         return [rect]

#     def plot3d(
#         self,
#         ax,
#         dims,
#         color,
#         fc_color="None",
#         linewidth=1,
#         zorder=2,
#         plot_2d=True,
#     ):
#         for i in range(len(self.range)):
#             rect = rect_prism(
#                 *self.constraints[i].range[dims, :],
#                 ax,
#                 color,
#                 linewidth,
#                 fc_color,
#                 zorder=zorder,
#             )
#         return rect

#     def get_t_max(self) -> int:
#         return len(self.constraints)

#     def to_reachable_input_objects(
#         self,
#     ) -> tuple[
#         Optional[np.ndarray],
#         Optional[np.ndarray],
#         np.ndarray,
#         np.ndarray,
#         float,
#     ]:
#         assert len(self.constraints) > 0
#         last_constraint = self.constraints[-1]
#         if last_constraint.range is None:
#             raise ValueError(
#                 "Can't convert LpConstraint to reachable_input_objects, since"
#                 " self.range is None."
#             )
#         x_min = last_constraint.range[:, 0]
#         x_max = last_constraint.range[:, 1]
#         norm = last_constraint.p
#         A_inputs = None
#         b_inputs = None
#         return A_inputs, b_inputs, x_max, x_min, norm

#     def add_timestep_constraint(
#         self, other: Union[LpConstraint, MultiTimestepLpConstraint]
#     ) -> MultiTimestepLpConstraint:
#         if other.range is None:
#             raise ValueError(
#                 "Trying to add_timestep_constraint but other.range is None."
#             )
#         if self.range is None:
#             return other.to_multistep_constraint()
#         return MultiTimestepLpConstraint(
#             constraints=self.constraints
#             + [
#                 constraint
#                 for constraint in other.to_multistep_constraint().constraints
#             ]
#         )

#     def get_constraint_at_time_index(self, i: int) -> LpConstraint:
#         return LpConstraint(range=self.constraints[i].range)

#     def add_cell(self, other: Optional[MultiTimestepLpConstraint]) -> None:
#         if other is None:
#             return

#         self.cells.append(other)
#         self.main_constraint_stale = True

#     def update_main_constraint_with_cells(self, overapprox: bool) -> None:
#         if overapprox:
#             # get min of all mins, get max of all maxes
#             tmp = np.stack(
#                 [c.range for c in self.cells if c.range is not None],
#                 axis=-1,
#             )
#             ranges = np.empty_like(self.cells[0].range)
#             ranges[..., 0] = np.min(tmp[..., 0, :], axis=-1)
#             ranges[..., 1] = np.max(tmp[..., 1, :], axis=-1)
#             self.constraints = []
#             for t, range in enumerate(ranges):
#                 self.constraints.append(LpConstraint(range=range))

#         else:
#             raise NotImplementedError

#         self.main_constraint_stale = False

#     def to_multistep_constraint(self) -> MultiTimestepLpConstraint:
#         if self.range is None:
#             raise ValueError(
#                 "Trying to convert to multistep constraint but self.range is"
#                 " None."
#             )
#         return self

#     def to_jittable(self):
#         return JittableMultiTimestepLpConstraint(
#             self.range[..., 0],
#             self.range[..., 1],
#             {jax_verify.IntervalBound: None},
#             {},
#         )

#     @classmethod
#     def from_jittable(cls, jittable_constraint):
#         return cls(
#             range=np.array(
#                 [jittable_constraint.lower, jittable_constraint.upper]
#             )
#         )

#     def _tree_flatten(self):
#         children = (self.range,)  # arrays / dynamic values
#         aux_data = {}  # static values
#         return (children, aux_data)

#     @classmethod
#     def _tree_unflatten(cls, aux_data, children):
#         return cls(*children, **aux_data)


# JittableMultiTimestepLpConstraint = collections.namedtuple(
#     "JittableMultiTimestepLpConstraint",
#     ["lower", "upper", "bound_type", "kwargs"],
# )


# def unjit_multi_timestep_lp_constraints(*inputs):
#     """Replace all the jittable bounds by standard bound objects."""

#     def is_jittable_constraint(b):
#         return isinstance(b, JittableMultiTimestepLpConstraint)

#     def unjit_bound(b):
#         return next(iter(b.bound_type)).from_jittable(b)

#     return jax.tree_util.tree_map(
#         lambda b: unjit_bound(b) if is_jittable_constraint(b) else b,
#         inputs,
#         is_leaf=is_jittable_constraint,
#     )


# jax.tree_util.register_pytree_node(
#     MultiTimestepLpConstraint,
#     MultiTimestepLpConstraint._tree_flatten,
#     MultiTimestepLpConstraint._tree_unflatten,
# )
