"""Tools for representing sets."""
from typing import Any, Optional, TypeGuard, Union

import numpy as np

from nfl_veripy.constraints.constraint_utils import (  # noqa
    RotatedLpConstraint,
    make_rect_from_arr,
)
from nfl_veripy.constraints.LpConstraint import (  # noqa
    LpConstraint,
    MultiTimestepLpConstraint,
    unjit_lp_constraints,
    unjit_multi_timestep_lp_constraints,
)
from nfl_veripy.constraints.PolytopeConstraint import (  # noqa
    MultiTimestepPolytopeConstraint,
    PolytopeConstraint,
    unjit_multi_timestep_polytope_constraints,
    unjit_polytope_constraints,
)
from nfl_veripy.utils.utils import get_polytope_A, range_to_polytope

MultiTimestepConstraint = Union[
    MultiTimestepLpConstraint,
    MultiTimestepPolytopeConstraint,
]
SingleTimestepConstraint = Union[LpConstraint, PolytopeConstraint]


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
                constraints=[PolytopeConstraint(A=get_polytope_A(num_facets))]
            )
        return MultiTimestepPolytopeConstraint()
    elif boundary_type == "rectangle":
        return MultiTimestepLpConstraint()
    else:
        raise NotImplementedError


def state_range_to_constraint(
    state_range: np.ndarray, boundary_type: str
) -> Union[LpConstraint, PolytopeConstraint]:
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
        return MultiTimestepLpConstraint(constraints=reachable_sets)
    elif is_polytope_constraint_list(reachable_sets):
        As = [r.A for r in reachable_sets]
        bs = [r.b for r in reachable_sets]
        assert is_npndarray_list(As)  # ensure all As are not None
        assert is_npndarray_list(bs)  # ensure all bs are not None
        return MultiTimestepPolytopeConstraint(
            constraints=[PolytopeConstraint(A=A, b=b) for A, b in zip(As, bs)]
        )
    else:
        raise ValueError(
            "reachable_sets list contains constraints with None range or A, b."
        )
