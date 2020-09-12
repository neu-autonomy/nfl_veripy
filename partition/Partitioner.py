import numpy as np
from itertools import product
from partition.xiang import sect, bisect
from partition.object_boundary import getboundary
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from partition.network_utils import get_sampled_outputs, samples_to_range
import imageio
import os
from sklearn.metrics import pairwise_distances
import time

label_dict = {
    "linf": "$\ell_\infty$-ball",
    "convex_hull": "Convex Hull",
    "lower_bnds": "Lower Bounds",
}




class Partitioner():
    def __init__(self):
        return

    def get_output_range(self):
        raise NotImplementedError

    def get_sampled_outputs(self, input_range, propagator, N=1000):
        return get_sampled_outputs(input_range, propagator, N=N)

    def samples_to_range(self, sampled_outputs):
        return samples_to_range(sampled_outputs)

    def squash_down_to_one_range_old(self, u_e, M):
        u_e_ = u_e.copy()
        if len(M) > 0:
            # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            M_range = np.empty_like(u_e_)
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
        
            # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([u_e_, M_range])
            u_e_[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e_[:,0] = np.min(tmp[:,0,:], axis=1)
        return u_e_

    def squash_down_to_one_range(self, output_range_sim, M):
        u_e = np.empty_like(output_range_sim)
        if len(M) > 0:
            # Squash all of M down to one range
            M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
            u_e[:,1] = np.max(M_numpy[:,1,:], axis=1)
            u_e[:,0] = np.min(M_numpy[:,0,:], axis=1)
        
            # Combine M (remaining ranges) with u_e (interior ranges)
            tmp = np.dstack([output_range_sim, u_e])
            u_e[:,1] = np.max(tmp[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp[:,0,:], axis=1)
        return u_e

    def squash_down_to_convex_hull(self, M, sim_hull_pts=None):
        from scipy.spatial import ConvexHull
        ndim = M[0][1].shape[0]
        pts = np.empty((len(M)*(2**(ndim)), ndim))
        i = 0
        for (input_range, output_range) in M:
            for pt in product(*output_range):
                pts[i,:] = pt
                i += 1
        hull = ConvexHull(pts, incremental=True)
        if sim_hull_pts is not None:
            hull.add_points(sim_hull_pts)
        return hull


    def setup_visualization(self, input_range, output_range, propagator, show_samples=True, outputs_to_highlight=None, inputs_to_highlight=None, show_input=True, show_output=True):
        num_subplots = int(show_input)+int(show_output)
        self.animate_fig, self.animate_axes = plt.subplots(1,num_subplots)

        if num_subplots == 1:
            if show_input:
                self.ax_input = self.animate_axes
            else:
                self.ax_output = self.animate_axes
        else:
            self.ax_input, self.ax_output = self.animate_axes

        self.input_axis = 0
        self.output_axis = int(show_input)


        if inputs_to_highlight is None:
            # Automatically detect which input dims to show based on input_range
            num_input_dimensions_to_plot = 2
            input_shape = input_range.shape[:-1]
            lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
            flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
            flat_dims.sort()
            input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
            input_names = ["NN Input Dimension: {}".format(input_dims[0][0]), "NN Input Dimension: {}".format(input_dims[1][0])]
        else:
            input_dims = [x['dim'] for x in inputs_to_highlight]
            input_names = [x['name'] for x in inputs_to_highlight]
        self.input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])

        if outputs_to_highlight is None:
            output_dims = [(0,),(1,)]
            output_names = ["NN Output Dimension 0", "NN Output Dimension 1"]
        else:
            output_dims = [x['dim'] for x in outputs_to_highlight]
            output_names = [x['name'] for x in outputs_to_highlight]
        self.output_dims_ = tuple([tuple([output_dims[j][i] for j in range(len(output_dims))]) for i in range(len(output_dims[0]))])

        if show_input:
            scale = 0.05
            x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
            y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
            self.ax_input.set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
            self.ax_input.set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
        if show_output:
            scale = 0.05
            x_off = max((output_range[output_dims[0]+(1,)] - output_range[output_dims[0]+(0,)])*(scale), 1e-5)
            y_off = max((output_range[output_dims[1]+(1,)] - output_range[output_dims[1]+(0,)])*(scale), 1e-5)
            self.ax_output.set_xlim(output_range[output_dims[0]+(0,)] - x_off, output_range[output_dims[0]+(1,)]+x_off)
            self.ax_output.set_ylim(output_range[output_dims[1]+(0,)] - y_off, output_range[output_dims[1]+(1,)]+y_off)
        # if show_input:
        #     self.ax_input.set_xlabel(input_names[0])
        #     self.ax_input.set_ylabel(input_names[1])
        # if show_output:
        #     self.ax_output.set_xlabel(output_names[0])
        #     self.ax_output.set_ylabel(output_names[1])

        # Make a rectangle for the Exact boundaries
        sampled_outputs = self.get_sampled_outputs(input_range, propagator)
        if show_output:
            if show_samples:
                self.ax_output.scatter(sampled_outputs[...,output_dims[0]], sampled_outputs[...,output_dims[1]], c='k', marker='.', zorder=2,
                    label="Sampled Outputs")

        self.default_patches = [[] for _ in range(num_subplots)]
        self.default_lines = [[] for _ in range(num_subplots)]

        # Full input range
        if show_input:
            input_range__ = input_range[self.input_dims_]
            input_rect = Rectangle(input_range__[:2,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
                            fc='none', linewidth=2,edgecolor='k', zorder=3,
                            label="Full Input Set")
            self.ax_input.add_patch(input_rect)
            self.default_patches[0] = [input_rect]

        if show_output:        
            # Exact output range
            color = 'red'
            linestyle = '--'
            linewidth = 2
            zorder=3
            if self.interior_condition == "linf":
                output_range_exact = self.samples_to_range(sampled_outputs)
                output_range_exact_ = output_range_exact[self.output_dims_]
                rect = Rectangle(output_range_exact_[:2,0], output_range_exact_[0,1]-output_range_exact_[0,0], output_range_exact_[1,1]-output_range_exact_[1,0],
                                fc='none', linewidth=linewidth, linestyle=linestyle, zorder=zorder, edgecolor=color,
                                label="True Bounds ({})".format(label_dict[self.interior_condition]))
                self.ax_output.add_patch(rect)
                self.default_patches[self.output_axis].append(rect)
            elif self.interior_condition == "lower_bnds":
                output_range_exact = self.samples_to_range(sampled_outputs)
                output_range_exact_ = output_range_exact[self.output_dims_]
                line1 = self.ax_output.axhline(output_range_exact_[1,0], linewidth=linewidth, linestyle=linestyle, zorder=zorder, color=color,
                    label="True Bounds ({})".format(label_dict[self.interior_condition]))
                line2 = self.ax_output.axvline(output_range_exact_[0,0], linewidth=linewidth, linestyle=linestyle, zorder=zorder, color=color)
                self.default_lines[self.output_axis].append(line1)
                self.default_lines[self.output_axis].append(line2)
            elif self.interior_condition == "convex_hull":
                from scipy.spatial import ConvexHull
                self.true_hull = ConvexHull(sampled_outputs)
                self.true_hull_ = ConvexHull(sampled_outputs[...,output_dims].squeeze())
                line = self.ax_output.plot(
                    np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[0]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[0]]),
                    np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[1]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[1]]),
                    color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder, 
                    label="True Bounds ({})".format(label_dict[self.interior_condition]))
                self.default_lines[self.output_axis].append(line[0])
            else:
                raise NotImplementedError

    def visualize(self, M, interior_M, u_e, iteration=0, show_input=True, show_output=True, title=None):
        if show_input:
            self.ax_input.patches = self.default_patches[self.input_axis].copy()
            self.ax_input.lines = self.default_lines[self.input_axis].copy()
        if show_output:
            self.ax_output.patches = self.default_patches[self.output_axis].copy()
            self.ax_output.lines = self.default_lines[self.output_axis].copy()
        input_dims_ = self.input_dims_

        # Rectangles that might still be outside the sim pts
        first = True
        for (input_range_, output_range_) in M:
            if first:
                input_label = 'Cell of Partition'
                output_label = "One Cell's Estimated Bounds"
                first = False
            else:
                input_label = None
                output_label = None

            if show_output:
                output_range__ = output_range_[self.output_dims_]
                rect = Rectangle(output_range__[:,0], output_range__[0,1]-output_range__[0,0], output_range__[1,1]-output_range__[1,0],
                        fc='none', linewidth=1,edgecolor='tab:blue',
                        label=output_label)
                self.ax_output.add_patch(rect)

            if show_input:
                input_range__ = input_range_[input_dims_]
                rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
                        fc='none', linewidth=1,edgecolor='tab:blue',
                        label=input_label)
                self.ax_input.add_patch(rect)

        # Rectangles that are within the sim pts
        for (input_range_, output_range_) in interior_M:

            if show_output:
                output_range__ = output_range_[self.output_dims_]
                rect = Rectangle(output_range__[:2,0], output_range__[0,1]-output_range__[0,0], output_range__[1,1]-output_range__[1,0],
                        fc='none', linewidth=1,edgecolor='tab:blue')
                self.ax_output.add_patch(rect)

            if show_input:
                input_range__ = input_range_[input_dims_]
                rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
                        fc='none', linewidth=1,edgecolor='tab:blue')
                self.ax_input.add_patch(rect)

        if show_output:
            linewidth = 3
            color = 'black'
            if self.interior_condition == "linf":
                # Make a rectangle for the estimated boundaries
                output_range_estimate = self.squash_down_to_one_range(u_e, M)
                output_range_estimate_ = output_range_estimate[self.output_dims_]
                rect = Rectangle(output_range_estimate_[:2,0], output_range_estimate_[0,1]-output_range_estimate_[0,0], output_range_estimate_[1,1]-output_range_estimate_[1,0],
                                fc='none', linewidth=linewidth,edgecolor=color,
                                label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
                self.ax_output.add_patch(rect)
            elif self.interior_condition == "lower_bnds":
                output_range_estimate = self.squash_down_to_one_range(u_e, M)
                output_range_estimate_ = output_range_estimate[self.output_dims_]
                self.ax_output.axhline(output_range_estimate_[1,0],
                    linewidth=linewidth,color=color,
                    label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
                self.ax_output.axvline(output_range_estimate_[0,0],
                    linewidth=linewidth,color=color)
            elif self.interior_condition == "convex_hull":
                from scipy.spatial import ConvexHull
                M_ = [(input_range_, output_range_[self.output_dims_]) for (input_range_, output_range_) in M]
                hull = self.squash_down_to_convex_hull(M_, self.true_hull_.points)
                self.ax_output.plot(
                    np.append(hull.points[hull.vertices,0], hull.points[hull.vertices[0],0]),
                    np.append(hull.points[hull.vertices,1], hull.points[hull.vertices[0],1]),
                    color=color, linewidth=linewidth,
                    label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
            else:
                raise NotImplementedError

        # plt.tight_layout()

        if title is not None:
            plt.suptitle(title)

        if self.show_animation:
            plt.pause(0.01)

        animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(animation_save_dir, exist_ok=True)
        plt.savefig(animation_save_dir+"tmp_{}.png".format(str(iteration).zfill(6)))
        plt.savefig(animation_save_dir+"tmp.png".format(str(iteration).zfill(6)))

    def get_error(self, output_range_exact, output_range_estimate):
        if self.interior_condition == "linf":
            true_area = np.product(output_range_exact[...,1] - output_range_exact[...,0])
            estimated_area = np.product(output_range_estimate[...,1] - output_range_estimate[...,0])
            error = (estimated_area - true_area) / true_area
        elif self.interior_condition == "lower_bnds":
            # Need to put lower bnd error into proper scale --> one idea is to use
            # length in each dimension of output (i.e., if you get 0.1 away
            # from lower bnd in a dimension that has range 100, that's more impressive
            # than in a dimension that has range 0.01)
            # lower_bnd_error_area = np.product(output_range_exact[...,0] - output_range_estimate[...,0])
            # true_area = np.product(output_range_exact[...,1] - output_range_exact[...,0])
            # error = lower_bnd_error_area / true_area

            # Just add up the distance btwn true and estimated lower bnds for each dimension
            error = np.sum(output_range_exact[...,0] - output_range_estimate[...,0])

        elif self.interior_condition == "convex_hull":
            true_area = output_range_exact.area
            estimated_area = output_range_estimate.area
            error = (estimated_area - true_area) / true_area
        else:
            raise NotImplementedError
        return error

    def check_termination(self, input_range_, num_propagator_calls, u_e, output_range_sim, M, iteration):
        if self.termination_condition_type == "input_cell_size":

          #  print(input_range_[...,1] - input_range_[...,0])
            M_numpy = np.dstack([input_range for (input_range,_) in M])

            terminate = np.min(M_numpy[:,1] - M_numpy[:,0]) <= self.termination_condition_value
        elif self.termination_condition_type == "num_propagator_calls":
            terminate = num_propagator_calls >= self.termination_condition_value
        elif self.termination_condition_type == "pct_improvement":
            # This doesnt work very well, because a lot of times
            # the one-step improvement is zero
            last_u_e = u_e.copy()
            if self.interior_condition in ["lower_bnds", "linf"]:
                u_e = self.squash_down_to_one_range(output_range_sim, M)
                improvement = self.get_error(last_u_e, u_e)
                if iteration == 0: improvement = np.inf
            elif self.interior_condition == "convex_hull":
               # raise NotImplementedError
                last_hull = estimated_hull.copy()

                estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                improvement = self.get_error(last_hull, estimated_hull)
            terminate = improvement <= self.termination_condition_value
        elif self.termination_condition_type == "pct_error":
            if self.interior_condition in ["lower_bnds", "linf"]:
                u_e = self.squash_down_to_one_range(output_range_sim, M)
                error = self.get_error(output_range_sim, u_e)
            elif self.interior_condition == "convex_hull":
                estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                error = self.get_error(self.sim_convex_hull, estimated_hull)
            terminate = error <= self.termination_condition_value
        elif self.termination_condition_type == "verify":
            M_ = M+[(input_range_, output_range_)]
            ndim = M_[0][1].shape[0]
            pts = np.empty((len(M_)*(2**(ndim)), ndim))
            i = 0
            for (input_range, output_range) in M:
                for pt in product(*output_range):
                    pts[i,:] = pt
                    i += 1
            # print(pts)
            # output_ranges_ = [x[1] for x in M+[(input_range_, output_range_)]]
            # ndim = output_ranges_[0].shape[0]
            # pts = np.empty((ndim**2, ndim+1))
            # pts[:,-1] = 1.
            # for i, pt in enumerate(product(*output_ranges_)):
            #     pts[i,:-1] = pt
            # inside = np.all(np.matmul(self.termination_condition_value[0], pts.T) <= self.termination_condition_value[1])
        else:
            raise NotImplementedError
        return terminate

    def compile_info(self, output_range_sim, M, interior_M, num_propagator_calls, t_end_overall, t_start_overall, propagator_computation_time):
        info = {}

        if self.interior_condition in ["lower_bnds", "linf"]:
            estimated_range = self.squash_down_to_one_range(output_range_sim, M)
            estimated_error = self.get_error(output_range_sim, estimated_range)
        elif self.interior_condition == "convex_hull":
            info["exact_hull"] = self.sim_convex_hull
            estimated_hull = self.squash_down_to_convex_hull(M+interior_M, self.sim_convex_hull.points)
            info["estimated_hull"] = estimated_hull
            estimated_error = self.get_error(self.sim_convex_hull, estimated_hull)

        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        info["estimation_error"] = estimated_error
        info["computation_time"] = t_end_overall - t_start_overall
        info["propagator_computation_time"] = propagator_computation_time

        return info

    def partition_loop(self, M, interior_M, output_range_sim, sect_method, num_propagator_calls, input_range, u_e, propagator, propagator_computation_time, t_start_overall):
        # Used by UnGuided, SimGuided, GreedySimGuided, etc.
        iteration = 0
        terminate = False
        while len(M) != 0 and not terminate:
            input_range_, output_range_ = self.grab_from_M(M, output_range_sim) # (Line 9)

            if self.check_if_partition_within_sim_bnds(output_range_, output_range_sim):
                # Line 11
                interior_M.append((input_range_, output_range_))
            else:
                # Line 14
                terminate = self.check_termination(input_range, num_propagator_calls, u_e, output_range_sim, M+[(input_range_, output_range_)], iteration)

                if not terminate:
                    # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                    # Lines 16-17
                    for input_range_ in input_ranges_:
                        t_start = time.time()
                        output_range_, _ = propagator.get_output_range(input_range_)
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += 1
                        M.append((input_range_, output_range_)) # Line 18

                       # if self.interior_condition in ["lower_bnds", "linf"]:
                        #    estimated_range = self.squash_down_to_one_range(output_range_sim, M)
                        #    estimated_error = self.get_error(output_range_sim, estimated_range)
                      #  elif self.interior_condition == "convex_hull":
                      #      estimated_hull = self.squash_down_to_convex_hull(M+interior_M, self.sim_convex_hull.points)
                      #      estimated_error = self.get_error(self.sim_convex_hull, estimated_hull)

                else: # Lines 19-20
                    M.append((input_range_, output_range_))

                if self.make_animation:
                    u_e = self.squash_down_to_one_range(output_range_sim, M)
                    # title = "# Partitions: {}, Error: {}".format(str(len(M)+len(interior_M)), str(round(error, 3)))
                    title = "# Propagator Calls: {}".format(str(num_propagator_calls))
                   # title = None
                    self.visualize(M, interior_M, u_e, iteration, show_input=self.show_input, show_output=self.show_output, title=title)
            iteration += 1

        # Line 24
        u_e = self.squash_down_to_one_range(output_range_sim, M)
        t_end_overall = time.time()

        # Stats & Visualization
        info = self.compile_info(output_range_sim, M, interior_M, num_propagator_calls, t_end_overall, t_start_overall, propagator_computation_time)
        if self.make_animation:
            self.compile_animation(iteration)

        return u_e, info

    def sample(self, input_range, propagator, N=None):
        ### This is only used for error estimation (evaluation!)
        if N is None:
            N = self.num_simulations
        # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
        # (Line 5)
        input_shape = input_range.shape[:-1]
        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (int(N),)+input_shape)
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:,0] = np.min(sampled_outputs, axis=0)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            self.sim_convex_hull = ConvexHull(sampled_outputs)

        return output_range_sim, sampled_outputs

class NoPartitioner(Partitioner):

    def __init__(self, num_simulations=1000, termination_condition_type="input_cell_size", termination_condition_value=0.1, interior_condition="linf", make_animation=False, show_animation=False):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation

    def get_output_range(self, input_range, propagator):
        input_shape = input_range.shape[:-1]

        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (self.num_simulations,)+input_shape)
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            self.sim_convex_hull = ConvexHull(sampled_outputs)

        output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:,0] = np.min(sampled_outputs, axis=0)

        t_start = time.time()
        output_range, info = propagator.get_output_range(input_range)
        t_end = time.time()

        propagator_computation_time = t_end - t_start
        num_propagator_calls = 1
        info = self.compile_info(output_range_sim, [(input_range, output_range)], [], num_propagator_calls, t_end, t_start, propagator_computation_time)

        return output_range, info

class UniformPartitioner(Partitioner):
    def __init__(self, num_simulations=1000, num_partitions=16,termination_condition_type="input_cell_size", termination_condition_value=0.1, interior_condition="linf", make_animation=False, show_animation=False, show_input=True, show_output=True):
        Partitioner.__init__(self)
        self.num_partitions = num_partitions
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation
        self.num_simulations = num_simulations

    def get_output_range(self, input_range, propagator, num_partitions=None):
        input_shape = input_range.shape[:-1]

        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (self.num_simulations,)+input_shape)
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            self.sim_convex_hull = ConvexHull(sampled_outputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:,0] = np.min(sampled_outputs, axis=0)

        propagator_computation_time = 0

        t_start= time.time()

        info = {}
        num_propagator_calls = 0

        input_shape = input_range.shape[:-1]
        if num_partitions is None:
            num_partitions = np.ones(input_shape, dtype=int)
            if isinstance(self.num_partitions, np.ndarray) and input_shape == self.num_partitions.shape:
                num_partitions = self.num_partitions
            elif len(input_shape) > 1:
                num_partitions[0,0] = self.num_partitions
            else:
                num_partitions *= self.num_partitions
        slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)

        ranges = []
        output_range = None
        
        for element in product(*[range(num) for num in num_partitions.flatten()]):
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[...,0] = input_range[...,0]+np.multiply(element_, slope)
            input_range_[...,1] = input_range[...,0]+np.multiply(element_+1, slope)
            output_range_, info_ = propagator.get_output_range(input_range_)
            num_propagator_calls += 1
            
            if output_range is None:
                output_range = np.empty(output_range_.shape)
                output_range[:,0] = np.inf
                output_range[:,1] = -np.inf

            tmp = np.dstack([output_range, output_range_])
            output_range[:,1] = np.max(tmp[:,1,:], axis=1)
            output_range[:,0] = np.min(tmp[:,0,:], axis=1)
            
            ranges.append((input_range_, output_range_))
            t_end = time.time()

        info = self.compile_info(output_range_sim, ranges, [], num_propagator_calls, t_end, t_start, propagator_computation_time)

        return output_range, info

class UnGuidedPartitioner(Partitioner):
    def __init__(self, num_simulations=1000, termination_condition_type="input_cell_size", termination_condition_value=0.1, interior_condition="linf", make_animation=False, show_animation=False, show_input=True, show_output=True):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation
        self.show_input = show_input
        self.show_output = show_output

    def grab_from_M(self, M, output_range_sim):
        input_range_, output_range_ = M.pop(0) 
        return input_range_, output_range_

    def compile_animation(self, iteration):
        animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        filenames = [animation_save_dir+"tmp_{}.png".format(str(i).zfill(6)) for i in range(iteration)]
        images = []
        for filename in filenames:
            try:
                image = imageio.imread(filename)
            except:
                continue
            images.append(imageio.imread(filename))
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            os.remove(filename)

        # Save the gif in a new animations sub-folder
        animation_filename = "tmp.gif"
        animation_save_dir = "{}/results/animations/".format(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(animation_save_dir, exist_ok=True)
        animation_filename = animation_save_dir+animation_filename
        imageio.mimsave(animation_filename, images)

    def get_output_range(self, input_range, propagator, verbose=False):
        t_start_overall = time.time()
        propagator_computation_time = 0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = 'max'
        input_shape = input_range.shape[:-1]
        info = {}

        num_propagator_calls = 0

        # Get initial output reachable set (Line 3)
        t_start = time.time()
        output_range, _ = propagator.get_output_range(input_range, verbose=verbose)
        t_end = time.time()
        propagator_computation_time += t_end - t_start
        num_propagator_calls += 1

        M = [(input_range, output_range)] # (Line 4)
        interior_M = []

        u_e = output_range.copy()

        output_range_sim_for_evaluation, sampled_outputs = self.sample(input_range, propagator)

        # output_range_sim is [-inf, inf] per dimension -- just a dummy
        output_range_sim = np.empty_like(output_range)
        output_range_sim[:,1] = -np.inf
        output_range_sim[:,0] = np.inf

        if self.make_animation:
            self.setup_visualization(input_range, output_range, propagator, show_input=self.show_input, show_output=self.show_output)

        print("Need to pass output_range_sim_for_evaluation to partition_loop somehow...")
        raise NotImplementedError
        u_e, info = self.partition_loop(M, interior_M, output_range_sim, sect_method, num_propagator_calls, input_range, u_e, propagator, propagator_computation_time, t_start_overall)
        
        return u_e, info

    def check_if_partition_within_sim_bnds(self, output_range, output_range_sim):
        return False

class SimGuidedPartitioner(Partitioner):
    def __init__(self, num_simulations=1000, termination_condition_type="input_cell_size", termination_condition_value=0.1, interior_condition="linf", make_animation=False, show_animation=False, show_input=True, show_output=True):
        Partitioner.__init__(self)
        self.num_simulations = num_simulations
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation
        self.show_input = show_input
        self.show_output = show_output

    def grab_from_M(self, M, output_range_sim):
        input_range_, output_range_ = M.pop(0) 
        return input_range_, output_range_

    def check_if_partition_within_sim_bnds(self, output_range, output_range_sim):
        # Check if output_range's linf ball is within
        # output_range_sim's linf ball
        inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0) and \
                    np.all((output_range_sim[...,1] - output_range[...,1]) >= 0)
        return inside

    def compile_animation(self, iteration):
        animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        filenames = [animation_save_dir+"tmp_{}.png".format(str(i).zfill(6)) for i in range(iteration)]
        images = []
        for filename in filenames:
            try:
                image = imageio.imread(filename)
            except:
                continue
            images.append(image)
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            os.remove(filename)

        # Save the gif in a new animations sub-folder
        animation_filename = "tmp.gif"
        animation_save_dir = "{}/results/animations/".format(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(animation_save_dir, exist_ok=True)
        animation_filename = animation_save_dir+animation_filename
        imageio.mimsave(animation_filename, images)

    def get_output_range(self, input_range, propagator):
        t_start_overall = time.time()

        propagator_computation_time = 0

        # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = 'max'
        input_shape = input_range.shape[:-1]

        num_propagator_calls = 0

        # Get initial output reachable set (Line 3)
        t_start = time.time()
        output_range, _ = propagator.get_output_range(input_range)
        t_end = time.time()
        propagator_computation_time += t_end - t_start
        num_propagator_calls += 1

        M = [(input_range, output_range)] # (Line 4)
        interior_M = []
        u_e = output_range.copy()
        
        # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        # (Line 5-6)
        output_range_sim, sampled_outputs = self.sample(input_range, propagator)

        if self.termination_condition_type == "verify":
            print(np.matmul(self.termination_condition_value[0], sampled_outputs.T))
            violated = np.all(np.matmul(self.termination_condition_value[0], sampled_outputs.T) > self.termination_condition_value[1])
            if violated:
                return "UNSAT"

        if self.make_animation:
            self.setup_visualization(input_range, output_range, propagator, show_input=self.show_input, show_output=self.show_output)
        
        u_e, info = self.partition_loop(M, interior_M, output_range_sim, sect_method, num_propagator_calls, input_range, u_e, propagator, propagator_computation_time, t_start_overall)

        return u_e, info

class GreedySimGuidedPartitioner(SimGuidedPartitioner):
    def __init__(self, num_simulations=1000, interior_condition="linf", make_animation=False, show_animation=False, termination_condition_type="interior_cell_size", termination_condition_value=0.02, show_input=True, show_output=True):
        SimGuidedPartitioner.__init__(self, num_simulations=num_simulations, interior_condition=interior_condition, make_animation=make_animation, show_animation=show_animation,
            termination_condition_type=termination_condition_type, termination_condition_value=termination_condition_value, show_input=show_input, show_output=show_output)

    def check_if_partition_within_sim_bnds(self, output_range, output_range_sim):
        if self.interior_condition == "linf":
            # Check if output_range's linf ball is within
            # output_range_sim's linf ball
            inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0) and \
                        np.all((output_range_sim[...,1] - output_range[...,1]) >= 0)
        elif self.interior_condition == "lower_bnds":
            # Check if output_range's lower bnds are above each of
            # output_range_sim's lower bnds
            inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0)
        elif self.interior_condition == "convex_hull":
            # Check if every vertex of the hyperrectangle of output_ranges
            # lies within the convex hull of the sim pts
            ndim = output_range.shape[0]
            pts = np.empty((2**ndim, ndim+1))
            pts[:,-1] = 1.
            for i, pt in enumerate(product(*output_range)):
                pts[i,:-1] = pt
            inside = np.all(np.matmul(self.sim_convex_hull.equations, pts.T) <= 0)
        else:
            raise NotImplementedError
        return inside

    def grab_from_M(self, M, output_range_sim):
        if len(M) == 1:
            input_range_, output_range_ = M.pop(0)
        else:
            version = 'orig'

            if self.interior_condition == "linf":
                # look thru all output_range_s and see which are furthest from sim output range
                M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
                z = np.empty_like(M_numpy)
                z[:,0,:] = (output_range_sim[:,0] - M_numpy[:,0,:].T).T
                z[:,1,:] = (M_numpy[:,1,:].T - output_range_sim[:,1]).T
                version = 'orig'
                # version = 'random'
                # version = 'improvement'
                if version == 'improvement':
                    pass
                elif version == 'random':
                    # This will randomly select one of the boundaries randomly
                    # and choose the element that is causing that boundary
                    worst_M_index = np.random.choice(np.unique(z.argmax(axis=-1)))
                elif version == 'orig':
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
                M_numpy = np.dstack([output_range_[:,0] for (_, output_range_) in M])
                z = (output_range_sim[:,0] - M_numpy[0].T).T
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            elif self.interior_condition == "convex_hull":
                from shapely.geometry import Point, Polygon

                    # Create Point objects

                # estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                estimated_hull = self.squash_down_to_convex_hull(M)
                outer_pts = estimated_hull.points[estimated_hull.vertices]
                inner_pts = self.sim_convex_hull.points[self.sim_convex_hull.vertices]

                if version == 'inner_boundary_check':

                    inner_poly = Polygon(inner_pts)
                    outerpoints_pts_outside =[]
                    for point in outer_pts:
                        if Point(point).within(inner_poly)==False:
                            outerpoints_pts_outside.append(point) 
                    outerpoints_pts_outside = np.array(outerpoints_pts_outside)
                else:  
                    outerpoints_pts_outside = outer_pts
                paired_distances = pairwise_distances(outerpoints_pts_outside, inner_pts)

                min_distances = np.min(paired_distances, axis=1)
                worst_index = np.unravel_index(np.argmax(min_distances), shape=min_distances.shape)

                worst_hull_pt_index = estimated_hull.vertices[worst_index[0]]

                # each outer_range in M adds 2**ndim pts to the convex hull,
                # ==> can id which rect it came from by integer dividing
                # since we add outer_range pts to convhull.pts in order
                num_pts_per_output_range = 2**np.product(M[0][1].shape[:-1])
                worst_M_index = worst_hull_pt_index // num_pts_per_output_range
                input_range_, output_range_ = M.pop(worst_M_index)
            else:
                raise NotImplementedError

        return input_range_, output_range_


class AdaptiveSimGuidedPartitioner(Partitioner):
    def __init__(self, num_simulations=1000, interior_condition="linf", make_animation=False, show_animation=False, termination_condition_type="interior_cell_size", termination_condition_value=0.02, show_input=True, show_output=True):


        self.num_simulations = num_simulations
        self.termination_condition_type = termination_condition_type
        self.termination_condition_value = termination_condition_value
        self.interior_condition = interior_condition
        self.make_animation = make_animation or show_animation
        self.show_animation = show_animation
        self.show_input = show_input
        self.show_output = show_output

    def compile_animation(self, iteration):
        animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        filenames = [animation_save_dir+"tmp_{}.png".format(str(i).zfill(6)) for i in range(iteration)]
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            os.remove(filename)

        # Save the gif in a new animations sub-folder
        animation_filename = "tmp.gif"
        animation_save_dir = "{}/results/animations/".format(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(animation_save_dir, exist_ok=True)
        animation_filename = animation_save_dir+animation_filename
        imageio.mimsave(animation_filename, images)
                    
    
    def get_error(self, output_range_exact, output_range_estimate):
        if self.interior_condition == "linf":
            true_area = np.product(output_range_exact[...,1] - output_range_exact[...,0])
            estimated_area = np.product(output_range_estimate[...,1] - output_range_estimate[...,0])
            error = (estimated_area - true_area) / true_area
        elif self.interior_condition == "lower_bnds":
            # Need to put lower bnd error into proper scale --> one idea is to use
            # length in each dimension of output (i.e., if you get 0.1 away
            # from lower bnd in a dimension that has range 100, that's more impressive
            # than in a dimension that has range 0.01)
            lower_bnd_error_area = np.product(output_range_exact[...,0] - output_range_estimate[...,0])
            true_area = np.product(output_range_exact[...,1] - output_range_exact[...,0])
            error = lower_bnd_error_area / true_area
        elif self.interior_condition == "convex_hull":

            true_area = output_range_exact.area
            estimated_area = output_range_estimate.area

           # true_area = self.boundary_area(output_range_exact)
           # estimated_area = self.boundary_area(output_range_estimate)
            error = (estimated_area - true_area) / true_area
        else:
            raise NotImplementedError
        return error

        #self.tolerance_step= tolerance_expanding_step
        #self.k_NN= k_NN
        #self.tolerance_range= tolerance_range

   # def grab_from_M(self, M, output_range_sim):
       # input_range_, output_range_ = M.pop(0) 
        #return input_range_, output_range_


    def grab_from_M(self, M, output_range_sim):
        if len(M) == 1:
            input_range_, output_range_ = M.pop(0)
        else:
            if self.interior_condition == "linf":
                # look thru all output_range_s and see which are furthest from sim output range
                M_numpy = np.dstack([output_range_ for (_, output_range_) in M])
                z = np.empty_like(M_numpy)
                z[:,0,:] = (output_range_sim[:,0] - M_numpy[:,0,:].T).T
                z[:,1,:] = (M_numpy[:,1,:].T - output_range_sim[:,1]).T

                version = 'orig'
                # version = 'random'
                # version = 'improvement'
                if version == 'improvement':
                    pass
                elif version == 'random':
                    # This will randomly select one of the boundaries randomly
                    # and choose the element that is causing that boundary
                    worst_M_index = np.random.choice(np.unique(z.argmax(axis=-1)))
                elif version == 'orig':
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
                M_numpy = np.dstack([output_range_[:,0] for (_, output_range_) in M])
                z = np.empty_like(M_numpy)
                z = (output_range_sim[:,0] - M_numpy.T).T
                worst_index = np.unravel_index(z.argmax(), shape=z.shape)
                worst_M_index = worst_index[-1]
                input_range_, output_range_ = M.pop(worst_M_index)
            elif self.interior_condition == "convex_hull":
                estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                outer_pts = estimated_hull.points[estimated_hull.vertices]
                inner_pts = self.sim_convex_hull.points[self.sim_convex_hull.vertices]
                paired_distances = pairwise_distances(outer_pts, inner_pts)
                min_distances = np.min(paired_distances, axis=1)
                worst_index = np.unravel_index(np.argmax(min_distances), shape=min_distances.shape)

                worst_hull_pt_index = estimated_hull.vertices[worst_index[0]]

                # each outer_range in M adds 2**ndim pts to the convex hull,
                # ==> can id which rect it came from by integer dividing
                # since we add outer_range pts to convhull.pts in order
                num_pts_per_output_range = 2**np.product(M[0][1].shape[:-1])
                worst_M_index = worst_hull_pt_index // num_pts_per_output_range
                input_range_, output_range_ = M.pop(worst_M_index)
            else:
                raise NotImplementedError

        return input_range_, output_range_


    def set_delta_step(self,range_,expanded_box, idx, stage=1):
        outer_points = np.zeros((2**range_.shape[0],2))
       # print("range of input:" , range_)
        outer_points = [[i,j] for i in range_[0,:] for j in range_[1,:]]
       # print("bounding box", outer_points)
       # inner_points[i] = expanded_box[i][0]
        range_area = np.product(range_[...,1] - range_[...,0])

        delta_step = np.zeros(range_.shape[0])
        if stage==1:
            c=0.5

            pairwise_distance= np.zeros(len(outer_points))
            for (i,points) in enumerate(outer_points):
                pairwise_distance[i] =((points[0]-expanded_box[0])**2+ (points[1]-expanded_box[1])**2)**0.5
            delta_step = c*np.max((pairwise_distance/range_area**0.5))*np.ones(range_.shape[0])
            print('delta step:',delta_step)
            return delta_step
        else:
            c=0.2

            distances = pairwise_distances(outer_points,expanded_box)
            min_distance = np.min(distances,axis=1)
            for (i,distance_pair) in enumerate(min_distance):
                delta_step[i] = c*distance_pair/range_area**0.5
            print('delta step:',delta_step)
            return delta_step

    def check_if_partition_within_sim_bnds(self, output_range, output_range_sim):
        if self.interior_condition == "linf":
            # Check if output_range's linf ball is within
            # output_range_sim's linf ball
            inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0) and \
                        np.all((output_range_sim[...,1] - output_range[...,1]) >= 0)
        elif self.interior_condition == "lower_bnds":
            # Check if output_range's lower bnds are above each of
            # output_range_sim's lower bnds
            inside = np.all((output_range_sim[...,0] - output_range[...,0]) <= 0)
        elif self.interior_condition == "convex_hull":
            # Check if the rectangle of output_range lies within the
            # convex hull of the sim pts
            ndim = output_range.shape[0]
            pts = np.empty((ndim**2, ndim+1))
            pts[:,-1] = 1.
            for i, pt in enumerate(product(*output_range)):
                pts[i,:-1] = pt
            inside = np.all(np.matmul(self.sim_convex_hull.equations, pts.T) <= 0)
        else:
            raise NotImplementedError
        return inside


    def get_output_range(self, input_range, propagator, verbose=False):

       # tolerance_eps = 0.05
        t_start_overall = time.time()

       ### old param tolerance_range = 0.01, c= 0.8, 2/3
        tolerance_step=0.0005
        tolerance_range=0.0001
        num_propagator_calls=0
        k_NN=1
        num_inputs = input_range.shape[0]


#####
       # Algorithm 1 of (Xiang, 2020): https://arxiv.org/pdf/2004.12273.pdf
        sect_method = 'max'
        input_shape = input_range.shape[:-1]
        info = {}

        num_propagator_calls = 0
        propagator_computation_time = 0


        M = [] # (Line 4)
        interior_M = []
        
        # Run N simulations (i.e., randomly sample N pts from input range --> query NN --> get N output pts)
        # (Line 5)
        sampled_inputs = np.random.uniform(input_range[...,0], input_range[...,1], (self.num_simulations,)+input_shape)
        sampled_outputs = propagator.forward_pass(sampled_inputs)

        if self.interior_condition == "convex_hull":
            from scipy.spatial import ConvexHull
            self.sim_convex_hull = ConvexHull(sampled_outputs)

        # Compute [u_sim], aka bounds on the sampled outputs (Line 6)
        output_range_sim = np.empty(sampled_outputs.shape[1:]+(2,))
        output_range_sim[:,1] = np.max(sampled_outputs, axis=0)
        output_range_sim[:,0] = np.min(sampled_outputs, axis=0)

        ##sampled_output_center=output_range_sim.mean(axis=1)
        if self.interior_condition == "convex_hull":
            sampled_output_center =(self.sim_convex_hull.points[self.sim_convex_hull.vertices]).mean(axis=0)
        else:
            sampled_output_center=output_range_sim.mean(axis=1)


####

        # = propagator.forward_pass(input_range[0,:])
        kdt = KDTree(sampled_outputs, leaf_size=30, metric='euclidean')
        center_NN=kdt.query(sampled_output_center.reshape(1,-1), k=k_NN, return_distance=False)
    

        input_range_new=np.empty_like(input_range)
        input_range_new[:,0] = np.min(sampled_inputs[center_NN], axis=0)
        input_range_new[:,1] = np.max(sampled_inputs[center_NN], axis=0)
 

        count =0
        M=[]
        delta_step = self.set_delta_step(output_range_sim,sampled_output_center, num_inputs, stage=1)
        prev_range =np.inf
        terminating_condition=False
        input_range_new= input_range_new+delta_step

        while terminating_condition==False:
 
            t_start = time.time()
            output_range_new, _= propagator.get_output_range(input_range_new, verbose=verbose)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1
            if self.termination_condition_type == "num_propagator_calls" and \
             (num_propagator_calls== (self.termination_condition_value)*0.8):
                break
            #if self.check_if_partition_within_sim_bnds(output_range_new, output_range_sim):

            if np.all((output_range_sim[:,0] - output_range_new[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_new[:,1]) >= 0):
                input_range_new= input_range_new+delta_step
              #  terminating_condition==False
            else:
                input_range_new= input_range_new-delta_step
                delta_step=1*delta_step/2

                if np.max(abs(delta_step))<tolerance_step:

                    diff=(input_range-input_range_new)

                    max_range= (np.max(abs(diff)))
                    if max_range<tolerance_range:# or abs(max_range-prev_range)<tol
                        break
                    else:
                        tol = 5
                        diff_rolled= np.concatenate(abs(diff))
                        if abs(max_range-prev_range)<tol:
                            count  = count+1
                        idx= num_inputs*2-1-count;
                        prev_range= np.inf;
                        if idx<0:
                            break
                        else:
                            prev_range = max_range
                            delta_step =self.set_delta_step(input_range, diff_rolled,idx, stage=2)  
                            if np.max(abs(delta_step))<tolerance_step:
                                break

    
                        
        if input_range[0,0] !=input_range_new[0,0]: 
   
            input_range_ = np.array([[input_range[0,0] ,input_range_new[0,0]],[input_range[1,0], input_range[1,1]]])
            t_start = time.time()
            output_range_,_ = propagator.get_output_range(input_range_, verbose=verbose)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1
    
            M.append((input_range_,output_range_)) 
      

        if[input_range_new[0,1]!=input_range[0,1]]:

                           #### approch2 only
            input_range_ = np.array([[input_range_new[0,1],input_range[0,1]],[input_range[1,0],input_range[1,1]]])
            t_start = time.time()
            output_range_,_ = propagator.get_output_range(input_range_, verbose=verbose)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1
   
            M.append((input_range_,output_range_))
                       
                        
        if[input_range_new[1,1]!=input_range[1,1]]:

            input_range_ = np.array([[input_range_new[0,0],input_range_new[0,1]],[input_range_new[1,1],input_range[1,1]]])
            t_start = time.time()
            output_range_,_ = propagator.get_output_range(input_range_, verbose=verbose)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1

            M.append((input_range_,output_range_))
                           

        if[input_range_new[1,0]!=input_range[1,0]]:
                        ### common partition between two approaches

            input_range_ = np.array([[input_range_new[0,0],input_range_new[0,1]],[input_range[1,0],input_range_new[1,0]]])
            t_start = time.time()
            output_range_,_ = propagator.get_output_range(input_range_, verbose=verbose)
            t_end = time.time()
            propagator_computation_time += t_end - t_start
            num_propagator_calls += 1

            M.append((input_range_,output_range_))      

        u_e = output_range_sim.copy()

        if self.make_animation:
            self.setup_visualization(input_range, output_range_sim, propagator, show_input=self.show_input, show_output=self.show_output)
        
        iteration = 0
        terminate = False
        while len(M) != 0 and not terminate:
            ##input_range_, output_range_ = self.grab_from_M(M, output_range_sim) # (Line 9)
            #input_range_, output_range_ = M.pop(0)
            input_range_, output_range_ =self.grab_from_M( M, output_range_sim)

            if self.check_if_partition_within_sim_bnds(output_range_, output_range_sim):
                # Line 11
                interior_M.append((input_range_, output_range_))
            else:
                # Line 14

                #################
                # Check if we should terminate the loop
                #
                if self.termination_condition_type == "input_cell_size":
                  #  print(np.min(input_range_[...,1] - input_range_[...,0]) )
                    terminate = np.min(input_range_[...,1] - input_range_[...,0]) <= self.termination_condition_value
                elif self.termination_condition_type == "num_propagator_calls":
                    terminate = num_propagator_calls >= self.termination_condition_value
                elif self.termination_condition_type == "pct_improvement":
                    # This doesnt work very well, because a lot of times
                    # the one-step improvement is zero
                    last_u_e = u_e.copy()
                    if self.interior_condition in ["lower_bnds", "linf"]:
                        u_e = self.squash_down_to_one_range(output_range_sim, M)
                        improvement = self.get_error(last_u_e, u_e)
                        if iteration == 0: improvement = np.inf
                    elif self.interior_condition == "convex_hull":
                        last_hull = estimated_hull.copy()

                        estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                        improvement = self.get_error(last_hull, estimated_hull)
                        #raise NotImplementedError
                    terminate = improvement <= self.termination_condition_value
                elif self.termination_condition_type == "pct_error":
                    if self.interior_condition in ["lower_bnds", "linf"]:
                        u_e = self.squash_down_to_one_range(output_range_sim, M)
                        error = self.get_error(output_range_sim, u_e)
                    elif self.interior_condition == "convex_hull":
                        estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
                        error = self.get_error(self.sim_convex_hull, estimated_hull)
                    terminate = error <= self.termination_condition_value
                else:
                    raise NotImplementedError
                #################

                if not terminate:
                    # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                    # Lines 16-17
                    for input_range_ in input_ranges_:
                        t_start = time.time()
                        output_range_, _ = propagator.get_output_range(input_range_, verbose=verbose)
                        t_end = time.time()
                        propagator_computation_time += t_end - t_start
                        num_propagator_calls += 1
                        M.append((input_range_, output_range_)) # Line 18
                else: # Lines 19-20
                    M.append((input_range_, output_range_))

                if self.make_animation:
                    u_e = self.squash_down_to_one_range(output_range_sim, M)
                    title = "# Partitions: {}, Error: {}".format(str(len(M)+len(interior_M)), str(round(error, 3)))
                    self.visualize(M, interior_M, u_e, iteration, show_input=self.show_input, show_output=self.show_output, title=title)
            iteration += 1
        #interior_M.append((input_range_new,output_range_new))


        # Line 24
        u_e = self.squash_down_to_one_range(output_range_sim, M)

        t_end_overall = time.time()

        if self.interior_condition in ["lower_bnds", "linf"]:
            estimated_range = self.squash_down_to_one_range(output_range_sim, M)
            estimated_error = self.get_error(output_range_sim, estimated_range)
        elif self.interior_condition == "convex_hull":
            estimated_hull = self.squash_down_to_convex_hull(M, self.sim_convex_hull.points)
            info["estimated_hull"] = estimated_hull
            estimated_error = self.get_error(self.sim_convex_hull, estimated_hull)

        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        info["computation_time"] =  t_end_overall - t_start_overall
        info["estimation_error"] =  estimated_error
        info["propagator_computation_time"] =  propagator_computation_time

        if self.make_animation:
            self.compile_animation(iteration)
        
        return u_e, info



class BoundarySimGuidedPartitioner(Partitioner):
    def __init__(self, N=20, tolerance_eps=0.01):
        Partitioner.__init__(self)
        self.N= N
        self.tolerance_eps= tolerance_eps

    def get_output_range(self, input_range, propagator):
        input_shape = input_range.shape[:-1]
        num_propagator_calls = 0

        interior_M = []
        M=[]
        M_inside=[]
        info = {}
        boundary_shape="convex"
        sect_method = 'max'
        num_inputs = input_range.shape[0]
        num_partitions = (self.N)*np.ones((num_inputs,), dtype=int)
        slope = np.divide((input_range[:,1] - input_range[:,0]), num_partitions)
        sampling_cell = np.empty(((self.N)**2,num_inputs,2))
        output_range_ = np.empty(((self.N)**2,num_inputs,2))
        sampled_output_ = np.empty(((self.N)**2,num_inputs))
        sampled_output_boundary_ = np.empty(((self.N)**2))

   
        input_range_ = np.empty((self.N**2,num_inputs,2))

        element=-1
        for idx in product(*[range(num) for num in num_partitions]):
            element = element+1
            input_range_[element,:,0] = input_range[:,0]+np.multiply(idx, slope)
            input_range_[element,:,1] = input_range[:,0]+np.multiply(np.array(idx)+1, slope)
            sampled_input_= np.random.uniform(input_range_[element,:,0], input_range_[element,:,1], (1,num_inputs,))
            sampled_output_[element,:] =  propagator.forward_pass( propagator.forward_pass(sampled_input_))
            sampled_output_boundary_[element] =0;
           # output_range_[element,:,:],_= propagator.get_output_range(input_range_[element,:,:])
           # num_propagator_calls += 1

        if boundary_shape=="convex":
            boundary_points= getboundary(sampled_output_,0.0)
        else:
            boundary_points= getboundary(sampled_output_,0.4)

        for i in range(self.N**2):
            if sampled_output_[i,:] in boundary_points:
                sampled_output_boundary_[i]=1
                propagator.get_output_range(input_range_[i,:,:])
                num_propagator_calls += 1
                M.append((input_range_[i,:,:], output_range_[i,:,:], sampled_output_boundary_[i]))# (Line 4)  

            else:
                M_inside.append((input_range_[i,:,:]))    

        

        output_range_sim = np.empty_like(output_range_[0,:,:])
        output_range_sim[:,0] = np.min(sampled_output_, axis=0)
        output_range_sim[:,1] = np.max(sampled_output_, axis=0)
            
        u_e = np.empty_like(output_range_sim)
        u_e[:,0] = np.inf
        u_e[:,1] = -np.inf


        M_e = np.empty_like(output_range_sim)
        M_e[:,0] = np.inf
        M_e[:,1] = -np.inf

        while len(M) != 0:
            input_range_, output_range_,sampled_output_boundary_ = M.pop(0) # Line 9
            if np.all((output_range_sim[:,0] - output_range_[:,0]) <= 0) and \
            np.all((output_range_sim[:,1] - output_range_[:,1]) >= 0) or sampled_output_boundary_==0:
            # Line 11
                tmp = np.dstack([u_e, output_range_])
                u_e[:,1] = np.max(tmp[:,1,:], axis=1)
                u_e[:,0] = np.min(tmp[:,0,:], axis=1)
                interior_M.append((input_range_, output_range_))
            else:
            # Line 14
                if np.max(input_range_[:,1] - input_range_[:,0]) > self.tolerance_eps:
                # Line 15
                    input_ranges_ = sect(input_range_, 2, select=sect_method)
                # Lines 16-17
                    for input_range_ in input_ranges_:
                        output_range_,_= propagator.get_output_range(input_range_)
                        num_propagator_calls += 1
   
                        M.append((input_range_, output_range_,sampled_output_boundary_)) # Line 18
                else: # Lines 19-20
                    interior_M.append((input_range_, output_range_))
                    break


        if len(M) > 0:
            M_numpy = np.dstack([output_range_ for (_, output_range_,_) in M])
            M_range = np.empty_like(u_e)
            M_range[:,1] = np.max(M_numpy[:,1,:], axis=1)
            M_range[:,0] = np.min(M_numpy[:,0,:], axis=1)
    
            tmp1 = np.dstack([u_e, M_range])
            u_e[:,1] = np.max(tmp1[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp1[:,0,:], axis=1)
        M1=[]
        if len(M_inside) > 0:
            for (input_range) in M_inside:
                output_range_,_= propagator.get_output_range(input_range_)
                num_propagator_calls += 1
                M2 = np.dstack([M_e, output_range_])
                M_e[:,1] = np.max(M2[:,1,:], axis=1)
                M_e[:,0] = np.min(M2[:,0,:], axis=1)

            tmp2 = np.dstack([u_e, M_e])
            u_e[:,1] = np.max(tmp2[:,1,:], axis=1)
            u_e[:,0] = np.min(tmp2[:,0,:], axis=1)
    

     
        info["all_partitions"] = M+interior_M
        info["exterior_partitions"] = M
        info["interior_partitions"] = interior_M
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = len(M) + len(interior_M)
        return u_e, info

