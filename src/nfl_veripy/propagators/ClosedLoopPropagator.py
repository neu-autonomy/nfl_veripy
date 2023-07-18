from copy import deepcopy
from itertools import product
from typing import Any, Optional

import numpy as np
from torch.multiprocessing import Pool

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics


class ClosedLoopPropagator:
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        self.dynamics = dynamics

        self.boundary_type: str = "rectangle"
        self.num_polytope_facets: Optional[int] = None
        self.num_iterations: int = 1

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = self.torch2network(network)

    def get_one_step_reachable_set(
        self, initial_set: constraints.SingleTimestepConstraint
    ) -> tuple[constraints.SingleTimestepConstraint, dict]:
        raise NotImplementedError

    def get_reachable_set(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        t_max: int,
    ) -> tuple[constraints.MultiTimestepConstraint, dict]:
        reachable_sets_list = []
        info = {"per_timestep": []}  # type: dict[str, Any]
        reachable_set, this_info = self.get_one_step_reachable_set(initial_set)
        reachable_sets_list.append(deepcopy(reachable_set))
        info["per_timestep"].append(this_info)
        for i in np.arange(
            0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
        ):
            next_initial_set = deepcopy(reachable_set)
            reachable_set, this_info = self.get_one_step_reachable_set(
                next_initial_set
            )
            reachable_sets_list.append(deepcopy(reachable_set))
            info["per_timestep"].append(this_info)

        reachable_sets = constraints.list_to_constraint(reachable_sets_list)

        return reachable_sets, info

    def get_one_step_backprojection_set(
        self,
        backreachable_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
        overapprox: bool = False,
        info: dict = {},
        facet_inds_to_optimize: Optional[np.ndarray] = None,
    ) -> tuple[Optional[constraints.SingleTimestepConstraint], dict]:
        raise NotImplementedError

    def get_single_element_backprojection_set(
        self,
        oc,
        input_constraint,
        t_max,
        num_partitions=None,
        overapprox=False,
        refined=False,
        heuristic="guided",
        all_lps=False,
        slow_cvxpy=False,
    ):
        part_input_constraints = []
        part_info = []

        input_constraint, this_info = self.get_one_step_backprojection_set(
            oc,
            input_constraint,
            num_partitions=num_partitions,
            overapprox=overapprox,
            collected_input_constraints=[oc]
            + deepcopy(part_input_constraints),
            refined=refined,
            heuristic=heuristic,
            all_lps=all_lps,
            slow_cvxpy=slow_cvxpy,
        )
        part_input_constraints.append(deepcopy(input_constraint))
        this_info["bp_set"] = input_constraint
        part_info.append(deepcopy(this_info))
        # info_['per_timestep'].append(this_info)

        if overapprox:
            for i in np.arange(
                0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
            ):
                next_output_constraint = over_approximate_constraint(
                    deepcopy(input_constraint)
                )
                next_input_constraint = deepcopy(next_output_constraint)
                input_constraint, this_info = (
                    self.get_one_step_backprojection_set(
                        next_output_constraint,
                        next_input_constraint,
                        num_partitions=num_partitions,
                        overapprox=overapprox,
                        collected_input_constraints=[oc]
                        + deepcopy(part_input_constraints),
                        infos=part_info,
                        refined=refined,
                        heuristic=heuristic,
                        all_lps=all_lps,
                        slow_cvxpy=slow_cvxpy,
                    )
                )
                part_input_constraints.append(deepcopy(input_constraint))
                this_info["bp_set"] = input_constraint
                part_info.append(deepcopy(this_info))
                # import pdb; pdb.set_trace()
                # info_['per_timestep'].append(this_info)
                # print(oc.range)
                # print(input_constraint.range)
        else:
            for i in np.arange(
                0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
            ):
                # TODO: Support N-step backprojection in the
                # under-approximation case
                raise NotImplementedError

        return deepcopy(part_input_constraints), deepcopy(part_info)

    def get_single_target_backprojection_set_old(
        self,
        output_constraint,
        input_constraint,
        t_max,
        num_partitions=None,
        overapprox=False,
        refined=False,
        heuristic="guided",
        all_lps=False,
        slow_cvxpy=False,
    ):
        element_list = []
        dim = output_constraint.range.shape[0]
        if not type(num_partitions).__module__ == np.__name__:
            num_partitions = np.array([num_partitions for i in range(dim)])
            # import pdb; pdb.set_trace()
        else:
            # num_partitions_ = num_partitions
            num_partitions_ = np.array([1, 1])
            # num_partitions_ = np.array([1, 1, 1, 1, 1, 1])
            # num_partitions_ = np.array([1,1,2,30])
        input_shape = output_constraint.range.shape[:-1]
        slope = np.divide(
            (
                output_constraint.range[..., 1]
                - output_constraint.range[..., 0]
            ),
            num_partitions_,
        )

        for el in product(
            *[range(int(num)) for num in num_partitions_.flatten()]
        ):
            oc_ = np.array(el).reshape(input_shape)
            output_range_ = np.empty_like(output_constraint.range, dtype=float)
            output_range_[..., 0] = output_constraint.range[
                ..., 0
            ] + np.multiply(oc_, slope)
            output_range_[..., 1] = output_constraint.range[
                ..., 0
            ] + np.multiply(oc_ + 1, slope)
            oc = constraints.LpConstraint(range=output_range_)

            element_list.append(oc)

        input_constraints_ = []
        info_ = {"per_timestep": []}

        arg_list = []

        for oc in element_list:
            arg_list.append(
                (
                    oc,
                    input_constraint,
                    t_max,
                    num_partitions,
                    overapprox,
                    refined,
                    heuristic,
                    all_lps,
                    slow_cvxpy,
                )
            )

        # import pdb; pdb.set_trace()
        parallel = False
        if parallel:
            with Pool(6) as p:
                partitioned_list = p.starmap(
                    self.get_single_element_backprojection_set, arg_list
                )

                input_constraints_ = [
                    partitioned_list[i][0]
                    for i in range(len(partitioned_list))
                ]
                info_["per_timestep"] = [
                    partitioned_list[i][1]
                    for i in range(len(partitioned_list))
                ]

                num_steps = len(partitioned_list[0][0])
        else:
            for oc in element_list:
                part_input_constraints = []
                part_info = []

                input_constraint, this_info = (
                    self.get_one_step_backprojection_set(
                        oc,
                        input_constraint,
                        num_partitions=num_partitions,
                        overapprox=overapprox,
                        collected_input_constraints=[oc]
                        + deepcopy(part_input_constraints),
                        refined=refined,
                        heuristic=heuristic,
                        all_lps=all_lps,
                        slow_cvxpy=slow_cvxpy,
                    )
                )
                part_input_constraints.append(deepcopy(input_constraint))
                this_info["bp_set"] = input_constraint
                part_info.append(deepcopy(this_info))
                # info_['per_timestep'].append(this_info)

                if overapprox:
                    for i in np.arange(
                        0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
                    ):
                        next_output_constraint = over_approximate_constraint(
                            deepcopy(input_constraint)
                        )
                        next_input_constraint = deepcopy(
                            next_output_constraint
                        )
                        input_constraint, this_info = (
                            self.get_one_step_backprojection_set(
                                next_output_constraint,
                                next_input_constraint,
                                num_partitions=num_partitions,
                                overapprox=overapprox,
                                collected_input_constraints=[oc]
                                + deepcopy(part_input_constraints),
                                infos=part_info,
                                refined=refined,
                                heuristic=heuristic,
                                all_lps=all_lps,
                                slow_cvxpy=slow_cvxpy,
                            )
                        )
                        part_input_constraints.append(
                            deepcopy(input_constraint)
                        )
                        this_info["bp_set"] = input_constraint
                        part_info.append(deepcopy(this_info))
                        # import pdb; pdb.set_trace()
                        # info_['per_timestep'].append(this_info)
                        # print(oc.range)
                        # print(input_constraint.range)
                else:
                    for i in np.arange(
                        0 + self.dynamics.dt + 1e-10, t_max, self.dynamics.dt
                    ):
                        # TODO: Support N-step backprojection in the
                        # under-approximation case
                        raise NotImplementedError

                input_constraints_.append(deepcopy(part_input_constraints))
                info_["per_timestep"].append(deepcopy(part_info))

                num_steps = len(part_input_constraints)

            # output_constraint: describes target set at t=t_max
            # input_constraints: [BP_{-1}, ..., BP_{-t_max}]
            #       i.e., [ states that will get to target set in 1 step,
            #               ...,
            #               states that will get to target set in t_max steps
            #             ]

        # import pdb; pdb.set_trace()

        input_constraints = [
            constraints.LpConstraint() for i in range(num_steps)
        ]
        info = {"per_timestep": []}

        for i, partition in enumerate(input_constraints_):
            for j, timestep in enumerate(partition):
                if input_constraints[j].range is None:
                    input_constraints[j].range = timestep.range
                else:
                    min_range = np.array(
                        [
                            np.minimum(
                                input_constraints[j].range[:, 0],
                                timestep.range[:, 0],
                            )
                        ]
                    ).T
                    max_range = np.array(
                        [
                            np.maximum(
                                input_constraints[j].range[:, 1],
                                timestep.range[:, 1],
                            )
                        ]
                    ).T
                    input_constraints[j].range = np.hstack(
                        (min_range, max_range)
                    )

        for i in range(num_steps):
            info["per_timestep"].append([])

        for i, partition in enumerate(info_["per_timestep"]):
            partition_info = info_["per_timestep"][i]
            for j, timestep in enumerate(partition_info):
                info["per_timestep"][j].append(timestep)

        return input_constraints, info


def over_approximate_constraint(constraint):
    print(
        "[warning] there is a sketchy over-approximation of a constraint"
        " occurring."
    )
    # Note: this is a super sketchy implementation that only works under
    # certain cases, specifically when all the contraints have the same A
    # matrix

    # TODO: Add an assert
    # TODO: implement a more general version

    # constraint.A = constraint.A[0]
    # constraint.b = [np.max(np.array(constraint.b), axis=0)]

    return constraint
