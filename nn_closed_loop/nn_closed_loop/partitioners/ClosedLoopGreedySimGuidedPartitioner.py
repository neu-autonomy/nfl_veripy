from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner
import nn_closed_loop.constraints as constraints
import numpy as np
import pypoman
from itertools import product
from copy import deepcopy
from nn_closed_loop.utils.utils import range_to_polytope
import time
from nn_partition.utils.utils import sect
from sklearn.metrics import pairwise_distances


class ClosedLoopGreedySimGuidedPartitioner(ClosedLoopSimGuidedPartitioner):
    def __init__(self, dynamics, num_partitions=16):
        ClosedLoopSimGuidedPartitioner.__init__(self, dynamics=dynamics)

    # def check_if_partition_within_sim_bnds(
    #     self, output_range, output_range_sim
    # ):
    #     if self.interior_condition == "linf":
    #         # Check if output_range's linf ball is within
    #         # output_range_sim's linf ball
    #         inside = np.all(
    #             (output_range_sim[..., 0] - output_range[..., 0]) <= 0
    #         ) and np.all(
    #             (output_range_sim[..., 1] - output_range[..., 1]) >= 0
    #         )
    #     elif self.interior_condition == "lower_bnds":
    #         # Check if output_range's lower bnds are above each of
    #         # output_range_sim's lower bnds
    #         inside = np.all(
    #             (output_range_sim[..., 0] - output_range[..., 0]) <= 0
    #         )
    #     elif self.interior_condition == "convex_hull":
    #         # Check if every vertex of the hyperrectangle of output_ranges
    #         # lies within the convex hull of the sim pts
    #         ndim = output_range.shape[0]
    #         pts = np.empty((2 ** ndim, ndim + 1))
    #         pts[:, -1] = 1.0
    #         for i, pt in enumerate(product(*output_range)):
    #             pts[i, :-1] = pt
    #         inside = np.all(
    #             np.matmul(self.sim_convex_hull.equations, pts.T) <= 0
    #         )
    #     else:
    #         raise NotImplementedError
    #     return inside

    def grab_from_M(self, M, output_range_sim):
        if len(M) == 1:
            input_range_, output_range_ = M.pop(0)
        else:
            if self.interior_condition == "linf":

                # TEMP: choose solely based on last timestep!
                timestep_of_interest = -1
                M_last_timestep = [(inp, out[timestep_of_interest]) for (inp, out) in M]
                output_range_sim_last_timestep = output_range_sim[timestep_of_interest]

                # look thru all output_range_s and see which are furthest from sim output range
                M_numpy = np.dstack(
                    [output_range_ for (_, output_range_) in M_last_timestep]
                )
                z = np.empty_like(M_numpy)
                z[:, 0, :] = (output_range_sim_last_timestep[:, 0] - M_numpy[:, 0, :].T).T
                z[:, 1, :] = (M_numpy[:, 1, :].T - output_range_sim_last_timestep[:, 1]).T
                # This selects whatver output range is furthest from
                # a boundary --> however, it can get too fixated on a single
                # bndry, esp when there's a sharp corner, suggesting
                # we might need to sample more, because our idea of where the
                # sim bndry is might be too far inward
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            else:
                raise NotImplementedError

        return input_range_, output_range_
