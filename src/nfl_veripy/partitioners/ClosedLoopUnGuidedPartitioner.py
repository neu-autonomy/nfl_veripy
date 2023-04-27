import numpy as np

import nfl_veripy.dynamics as dynamics

from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner


class ClosedLoopUnGuidedPartitioner(ClosedLoopSimGuidedPartitioner):
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        super().__init__(dynamics=dynamics)

    def check_if_partition_within_sim_bnds(
        self, output_range, output_range_sim
    ) -> bool:
        return False

    def get_sampled_out_range_guidance(
        self, input_constraint, propagator, t_max=5, num_samples=1000
    ):
        return None

    def squash_down_to_one_range(self, output_range_sim, M):
        # Same as ClosedLoopSimGuided's method, but ignore output_range_sim

        # (len(M)+1, t_max, n_states, 2)
        tmp = np.array([m[-1] for m in M])
        mins = np.min(tmp[..., 0], axis=0)
        maxs = np.max(tmp[..., 1], axis=0)
        return np.stack([mins, maxs], axis=2)
