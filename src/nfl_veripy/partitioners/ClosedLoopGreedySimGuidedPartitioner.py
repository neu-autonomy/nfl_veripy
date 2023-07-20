from typing import Optional

import numpy as np

import nfl_veripy.constraints as constraints
import nfl_veripy.dynamics as dynamics

from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner


class ClosedLoopGreedySimGuidedPartitioner(ClosedLoopSimGuidedPartitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        super().__init__(dynamics=dynamics)

    def grab_from_M(
        self,
        M: list[
            tuple[
                constraints.SingleTimestepConstraint,
                constraints.MultiTimestepConstraint,
            ]
        ],
        output_range_sim: Optional[np.ndarray],
    ) -> tuple[
        constraints.SingleTimestepConstraint,
        constraints.MultiTimestepConstraint,
    ]:
        if len(M) == 1:
            input_range, output_range = M.pop(0)
        else:
            if self.interior_condition == "linf":
                # TEMP:
                # choose solely based on first timestep!
                # timestep_of_interest = 0
                # choose solely based on last timestep!
                timestep_of_interest = -1

                output_range_last_timestep = np.array(
                    [
                        out.get_constraint_at_time_index(
                            timestep_of_interest
                        ).range
                        for (_, out) in M
                    ]
                ).transpose(1, 2, 0)
                assert output_range_sim is not None
                output_range_sim_last_timestep = output_range_sim[
                    timestep_of_interest
                ]

                # look thru all output_range_s and see which are furthest from
                # sim output range
                z = np.empty_like(output_range_last_timestep)
                z[:, 0, :] = (
                    output_range_sim_last_timestep[:, 0]
                    - output_range_last_timestep[:, 0, :].T
                ).T
                z[:, 1, :] = (
                    output_range_last_timestep[:, 1, :].T
                    - output_range_sim_last_timestep[:, 1]
                ).T

                # This selects whatver output range is furthest from
                # a boundary --> however, it can get too fixated on a single
                # bndry, esp when there's a sharp corner, suggesting
                # we might need to sample more, because our idea of where the
                # sim bndry is might be too far inward
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range, output_range = M.pop(worst_M_index)
            else:
                raise NotImplementedError

        return input_range, output_range
