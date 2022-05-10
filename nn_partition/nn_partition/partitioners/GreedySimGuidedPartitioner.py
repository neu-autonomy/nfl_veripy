from .SimGuidedPartitioner import SimGuidedPartitioner
import numpy as np
from itertools import product
from sklearn.metrics import pairwise_distances
from shapely.geometry import Point, Polygon


class GreedySimGuidedPartitioner(SimGuidedPartitioner):
    def __init__(
        self,
        num_simulations=1000,
        interior_condition="linf",
        make_animation=False,
        show_animation=False,
        termination_condition_type="interior_cell_size",
        termination_condition_value=0.02,
        show_input=True,
        show_output=True,
        adaptive_flag=False,
    ):
        SimGuidedPartitioner.__init__(
            self,
            num_simulations=num_simulations,
            interior_condition=interior_condition,
            make_animation=make_animation,
            show_animation=show_animation,
            termination_condition_type=termination_condition_type,
            termination_condition_value=termination_condition_value,
            show_input=show_input,
            show_output=show_output,
            adaptive_flag=adaptive_flag,
        )

    def check_if_partition_within_sim_bnds(
        self, output_range, output_range_sim
    ):
        if self.interior_condition == "linf":
            # Check if output_range's linf ball is within
            # output_range_sim's linf ball
            inside = np.all(
                (output_range_sim[..., 0] - output_range[..., 0]) <= 0
            ) and np.all(
                (output_range_sim[..., 1] - output_range[..., 1]) >= 0
            )
        elif self.interior_condition == "lower_bnds":
            # Check if output_range's lower bnds are above each of
            # output_range_sim's lower bnds
            inside = np.all(
                (output_range_sim[..., 0] - output_range[..., 0]) <= 0
            )
        elif self.interior_condition == "convex_hull":
            # Check if every vertex of the hyperrectangle of output_ranges
            # lies within the convex hull of the sim pts
            ndim = output_range.shape[0]
            pts = np.empty((2 ** ndim, ndim + 1))
            pts[:, -1] = 1.0
            for i, pt in enumerate(product(*output_range)):
                pts[i, :-1] = pt
            inside = np.all(
                np.matmul(self.sim_convex_hull.equations, pts.T) <= 0
            )
        else:
            raise NotImplementedError
        return inside

    def grab_from_M(self, M, output_range_sim):
        if len(M) == 1:
            input_range_, output_range_ = M.pop(0)
        else:
            version = "orig"

            if self.interior_condition == "linf":
                # look thru all output_range_s and see which are furthest from sim output range
                M_numpy = np.dstack(
                    [output_range_ for (_, output_range_) in M]
                )
                z = np.empty_like(M_numpy)
                z[:, 0, :] = (output_range_sim[:, 0] - M_numpy[:, 0, :].T).T
                z[:, 1, :] = (M_numpy[:, 1, :].T - output_range_sim[:, 1]).T
                version = "orig"
                # version = 'random'
                # version = 'improvement'
                if version == "improvement":
                    pass
                elif version == "random":
                    # This will randomly select one of the boundaries randomly
                    # and choose the element that is causing that boundary
                    worst_M_index = np.random.choice(
                        np.unique(z.argmax(axis=-1))
                    )
                elif version == "orig":
                    # This selects whatver output range is furthest from
                    # a boundary --> however, it can get too fixated on a single
                    # bndry, esp when there's a sharp corner, suggesting
                    # we might need to sample more, because our idea of where the
                    # sim bndry is might be too far inward
                    worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                    worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            elif self.interior_condition == "lower_bnds":
                # look thru all lower bnds and see which are furthest from sim lower bnds
                M_numpy = np.dstack(
                    [output_range_[:, 0] for (_, output_range_) in M]
                )
                z = (output_range_sim[:, 0] - M_numpy[0].T).T
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            elif self.interior_condition == "convex_hull":

                # Create Point objects

                # estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                estimated_hull = self.squash_down_to_convex_hull(M)
                outer_pts = estimated_hull.points[estimated_hull.vertices]
                inner_pts = self.sim_convex_hull.points[
                    self.sim_convex_hull.vertices
                ]

                if version == "inner_boundary_check":

                    inner_poly = Polygon(inner_pts)
                    outerpoints_pts_outside = []
                    for point in outer_pts:
                        if Point(point).within(inner_poly) == False:
                            outerpoints_pts_outside.append(point)
                    outerpoints_pts_outside = np.array(outerpoints_pts_outside)
                else:
                    outerpoints_pts_outside = outer_pts
                paired_distances = pairwise_distances(
                    outerpoints_pts_outside, inner_pts
                )

                min_distances = np.min(paired_distances, axis=1)
                worst_index = np.unravel_index(
                    np.argmax(min_distances), shape=min_distances.shape
                )

                worst_hull_pt_index = estimated_hull.vertices[worst_index[0]]

                # each outer_range in M adds 2**ndim pts to the convex hull,
                # ==> can id which rect it came from by integer dividing
                # since we add outer_range pts to convhull.pts in order
                num_pts_per_output_range = 2 ** np.product(M[0][1].shape[:-1])
                worst_M_index = worst_hull_pt_index // num_pts_per_output_range
                input_range_, output_range_ = M.pop(worst_M_index)
            else:
                raise NotImplementedError

        return input_range_, output_range_
