import numpy as np

from .ClosedLoopSimGuidedPartitioner import ClosedLoopSimGuidedPartitioner


class ClosedLoopGreedySimGuidedPartitioner(ClosedLoopSimGuidedPartitioner):
    def __init__(
        self,
        dynamics,
        num_partitions=16,
        make_animation=False,
        show_animation=False,
    ):
        ClosedLoopSimGuidedPartitioner.__init__(
            self,
            dynamics=dynamics,
            make_animation=make_animation,
            show_animation=show_animation,
        )

    def grab_from_M(self, M, output_range_sim):
        if len(M) == 1:
            input_range_, output_range_ = M.pop(0)
        else:
            if self.interior_condition == "linf":
                # TEMP:
                # choose solely based on first timestep!
                # timestep_of_interest = 0
                # choose solely based on last timestep!
                timestep_of_interest = -1

                M_last_timestep = [
                    (inp, out.range[timestep_of_interest]) for (inp, out) in M
                ]
                output_range_sim_last_timestep = output_range_sim[
                    timestep_of_interest
                ]

                # look thru all output_range_s and see which are furthest from
                # sim output range
                M_numpy = np.dstack(
                    [output_range_ for (_, output_range_) in M_last_timestep]
                )
                z = np.empty_like(M_numpy)
                z[:, 0, :] = (
                    output_range_sim_last_timestep[:, 0] - M_numpy[:, 0, :].T
                ).T
                z[:, 1, :] = (
                    M_numpy[:, 1, :].T - output_range_sim_last_timestep[:, 1]
                ).T
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
