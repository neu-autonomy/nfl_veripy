import numpy as np
from partition.Partitioner import Partitioner, UniformPartitioner
import pypoman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
# from closed_loop.nn import control_nn
from itertools import product
from closed_loop.utils import init_state_range_to_polytope
from closed_loop.ClosedLoopConstraints import PolytopeInputConstraint, LpInputConstraint, PolytopeOutputConstraint, LpOutputConstraint
from copy import deepcopy

class ClosedLoopPartitioner(Partitioner):
    def __init__(self, dynamics):
        Partitioner.__init__(self)
        self.dynamics = dynamics

    def get_one_step_reachable_set(self, input_constraint, output_constraint, propagator):
        output_constraint, info = propagator.get_one_step_reachable_set(input_constraint, deepcopy(output_constraint))
        return output_constraint, info

    def get_reachable_set(self, input_constraint, output_constraint, propagator, t_max):
        output_constraint_, info = propagator.get_reachable_set(input_constraint, deepcopy(output_constraint), t_max)
        
        # TODO: this is repeated from UniformPartitioner... make more universal
        if isinstance(output_constraint, PolytopeOutputConstraint):
            reachable_set_ = [o.b for o in output_constraint_]
            if output_constraint.b is None:
                output_constraint.b = np.stack(reachable_set_)

            tmp = np.dstack([output_constraint.b, np.stack(reachable_set_)])
            output_constraint.b = np.max(tmp, axis=-1)
            
            # ranges.append((input_range_, reachable_set_))
        elif isinstance(output_constraint, LpOutputConstraint):
            reachable_set_ = [o.range for o in output_constraint_]
            if output_constraint.range is None:
                output_constraint.range = np.stack(reachable_set_)

            tmp = np.stack([output_constraint.range, np.stack(reachable_set_)], axis=-1)
            output_constraint.range[...,0] = np.min(tmp[...,0,:], axis=-1)
            output_constraint.range[...,1] = np.max(tmp[...,1,:], axis=-1)

            # ranges.append((input_range_, np.stack(reachable_set_)))
        else:
            raise NotImplementedError
        prob_list =None
        return output_constraint, info,prob_list

    def get_output_range(self, input_constraint, output_constraint):

        if isinstance(output_constraint, PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
            t_max = len(b_out)
        elif isinstance(output_constraint, LpOutputConstraint):
            output_range = output_constraint.range
            output_p = output_constraint.p
            t_max = len(output_range)
        else:
            raise NotImplementedError
        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
            num_states = A_inputs.shape[-1]
        elif isinstance(input_constraint, LpInputConstraint):
            input_range = input_constraint.range
            input_p = input_constraint.p
            num_states = input_range.shape[0]
        else:
            raise NotImplementedError
        return output_range

    def get_error(self, input_constraint,output_constraint , propagator, t_max):
        
        if isinstance(input_constraint, LpInputConstraint):
            output_estimated_range = output_constraint.range
           # t_max = len(output_estimated_range)
            output_range_exact = self.get_sampled_out_range(input_constraint, propagator, t_max , num_samples =1000)
            error = 0
            for t in range(int(t_max/self.dynamics.dt)):                 
                true_area = np.product(output_range_exact[t][...,1] - output_range_exact[t][...,0])
                estimated_area = np.product(output_estimated_range[t][...,1] - output_estimated_range[t][...,0])
                error +=(estimated_area - true_area) / true_area
        else:
            raise NotImplementedError
        final_error = (estimated_area - true_area) / true_area
        avg_error = error/t_max
        return final_error,avg_error
        
    def get_sampled_out_range(self, input_constraint, propagator, t_max =5, num_samples =1000):
        return self.dynamics.get_sampled_output_range(input_constraint, t_max,  num_samples, controller=propagator.network)
    def setup_visualization_multiple(self, input_constraint, output_constraint, propagator, input_dims_, prob_list =None, show_samples=True, outputs_to_highlight=None, color='g', line_style='-'):
        input_dims = input_dims_
        if isinstance(output_constraint, PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
            t_max = len(b_out)
        elif isinstance(output_constraint, LpOutputConstraint):
            output_range = output_constraint.range
            output_p = output_constraint.p
            output_prob = prob_list
            t_max = len(output_range)
        else:
            raise NotImplementedError
        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
            num_states = A_inputs.shape[-1]
            output_prob = prob_list

        elif isinstance(input_constraint, LpInputConstraint):
            input_range = input_constraint.range
            input_p = input_constraint.p
            num_states = input_range.shape[0]
            output_prob = prob_list
        else:
            raise NotImplementedError



        # scale = 0.05
        # x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
        # y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
        # self.animate_axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
        # self.animate_axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
       
       # if show_samples:
        #    self.dynamics.show_samples(t_max*self.dynamics.dt, input_constraint, ax=self.animate_axes, controller=propagator.network, input_dims= input_dims_)

        # # Make a rectangle for the Exact boundaries
        # sampled_outputs = self.get_sampled_outputs(input_range, propagator)
        #if show_samples:
         #    self.animate_axes.scatter(sampled_outputs[...,output_dims[0]], sampled_outputs[...,output_dims[1]], c='k', marker='.', zorder=2,
         #        label="Sampled States")

        linewidth = 2
        if show_samples:
            self.dynamics.show_samples(t_max*self.dynamics.dt, input_constraint, ax=self.animate_axes, controller=propagator.network, input_dims=input_dims)

        # Initial state set
        init_state_color = 'k'

        if isinstance(input_constraint, PolytopeInputConstraint):
            # TODO: this doesn't use the computed input_dims...
            try:
                vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
            except:
                print("[warning] Can't visualize polytopic input constraints for >2 states. Need to implement this to it extracts input_dims.")
                raise NotImplementedError
            self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                color=color, linewidth=linewidth, linestyle = line_style, label='Initial States')
        elif isinstance(input_constraint, LpInputConstraint):
            rect = Rectangle(input_range[input_dims,0], input_range[input_dims[0],1]-input_range[input_dims[0],0], input_range[input_dims[1],1]-input_range[input_dims[1],0],
                            fc='none', linewidth=linewidth, linestyle =line_style, edgecolor=init_state_color)
            self.animate_axes.add_patch(rect)
            # self.default_patches[1].append(rect)
        else:
            raise NotImplementedError
      
        linewidth=1.5
        # Reachable sets
        if prob_list is None:
           fc_color='none'
        else:
           fc_color='none'
           alpha=0.17
        if isinstance(output_constraint, PolytopeOutputConstraint):
            # TODO: this doesn't use the computed input_dims...
            for i in range(len(b_out)):
                vertices = pypoman.compute_polygon_hull(A_out, b_out[i])
                self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                    color=color, label='$\mathcal{R}_'+str(i+1)+'$')
        elif isinstance(output_constraint, LpOutputConstraint):
            if prob_list is None:
                for output_range_ in output_range:
                    rect = Rectangle(output_range_[input_dims,0], output_range_[input_dims[0],1]-output_range_[input_dims[0],0], output_range_[input_dims[1],1]-output_range_[input_dims[1],0],
                       fc=fc_color, linewidth=linewidth, linestyle = line_style,edgecolor=color)
                    self.animate_axes.add_patch(rect)

            else:    
                for output_range_,prob in zip(output_range,prob_list):
                    fc_color=cm.get_cmap('Greens')(prob)
                    rect = Rectangle(output_range_[input_dims,0], output_range_[input_dims[0],1]-output_range_[input_dims[0],0], output_range_[input_dims[1],1]-output_range_[input_dims[1],0],
                        fc= fc_color, alpha = alpha ,linewidth=linewidth, linestyle = line_style,edgecolor=None)
                    self.animate_axes.add_patch(rect)

                 

        else:
            raise NotImplementedError

        
        # self.default_patches = [[], []]
        # self.default_lines = [[], []]
        # self.default_patches[0] = [input_rect]
        
        # # Exact output range
        # color = 'black'
        # linewidth = 3
        # if self.interior_condition == "linf":
        #     output_range_exact = self.samples_to_range(sampled_outputs)
        #     output_range_exact_ = output_range_exact[self.output_dims_]
        #     rect = Rectangle(output_range_exact_[:2,0], output_range_exact_[0,1]-output_range_exact_[0,0], output_range_exact_[1,1]-output_range_exact_[1,0],
        #                     fc='none', linewidth=linewidth,edgecolor=color,
        #                     label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].add_patch(rect)
        #     self.default_patches[1].append(rect)
        # elif self.interior_condition == "lower_bnds":
        #     output_range_exact = self.samples_to_range(sampled_outputs)
        #     output_range_exact_ = output_range_exact[self.output_dims_]
        #     line1 = self.animate_axes[1].axhline(output_range_exact_[1,0], linewidth=linewidth,color=color,
        #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     line2 = self.animate_axes[1].axvline(output_range_exact_[0,0], linewidth=linewidth,color=color)
        #     self.default_lines[1].append(line1)
        #     self.default_lines[1].append(line2)
        # elif self.interior_condition == "convex_hull":
        #     from scipy.spatial import ConvexHull
        #     self.true_hull = ConvexHull(sampled_outputs)
        #     self.true_hull_ = ConvexHull(sampled_outputs[...,output_dims].squeeze())
        #     line = self.animate_axes[1].plot(
        #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[0]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[0]]),
        #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[1]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[1]]),
        #         color=color, linewidth=linewidftypeth,
        #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.default_lines[1].append(line[0])
        # else:
        #     raise NotImplementedError
    def setup_visualization(self, input_constraint, output_constraint, propagator, prob_list =None, show_samples=True, outputs_to_highlight=None, inputs_to_highlight=None):
        if isinstance(output_constraint, PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
            t_max = len(b_out)
        elif isinstance(output_constraint, LpOutputConstraint):
            output_range = output_constraint.range
            output_p = output_constraint.p
            output_prob = prob_list
            t_max = len(output_range)
        else:
            raise NotImplementedError
        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
            num_states = A_inputs.shape[-1]
            output_prob = prob_list

        elif isinstance(input_constraint, LpInputConstraint):
            input_range = input_constraint.range
            input_p = input_constraint.p
            num_states = input_range.shape[0]
            output_prob = prob_list
        else:
            raise NotImplementedError

        self.animate_fig, self.animate_axes = plt.subplots(1,1)

        if inputs_to_highlight is None:
            # Automatically detect which input dims to show based on input_range
            # num_input_dimensions_to_plot = 2
            # input_shape = A_inputs.
            # lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
            # flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
            # flat_dims.sort()
            input_dims = [[0], [1]]
            # input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
            input_names = ["State: {}".format(input_dims[0][0]), "State: {}".format(input_dims[1][0])]
        else:
            input_dims = [x['dim'] for x in inputs_to_highlight]
            input_names = [x['name'] for x in inputs_to_highlight]
      #  self.input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])
        self.input_dims_ = input_dims




        if outputs_to_highlight is None:
            # Automatically detect which input dims to show based on input_range
            # num_input_dimensions_to_plot = 2
            # input_shape = A_inputs.
            # lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
            # flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
            # flat_dims.sort()
            output_dims = [[0], [1]]
            # input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
            output_names = ["State: {}".format(output_dims[0][0]), "State: {}".format(output_dims[1][0])]
        else:
            output_dims = [x['dim'] for x in outputs_to_highlight]
            output_names = [x['name'] for x in outputs_to_highlight]
      #  self.input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])
        self.output_dims_ = output_dims
        # scale = 0.05
        # x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
        # y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
        # self.animate_axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
        # self.animate_axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
        self.animate_axes.set_xlabel(input_names[0])
        self.animate_axes.set_ylabel(input_names[1])

        if show_samples:
            self.dynamics.show_samples(t_max*self.dynamics.dt, input_constraint, ax=self.animate_axes, controller=propagator.network, input_dims=input_dims)

        # # Make a rectangle for the Exact boundaries
        # sampled_outputs = self.get_sampled_outputs(input_range, propagator)
        # if show_samples:
        #     self.animate_axes.scatter(sampled_outputs[...,output_dims[0]], sampled_outputs[...,output_dims[1]], c='k', marker='.', zorder=2,
        #         label="Sampled States")

        # Initial state set
        if prob_list is None:
           color = 'k'
        else:
           color = None
        if isinstance(input_constraint, PolytopeInputConstraint):
            # TODO: this doesn't use the computed input_dims...
            try:
                vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
            except:
                print("[warning] Can't visualize polytopic input constraints for >2 states. Need to implement this to it extracts input_dims.")
                raise NotImplementedError
            self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                color=color, label='Initial States')
        elif isinstance(input_constraint, LpInputConstraint):
            rect = Rectangle(input_range[input_dims,0], input_range[input_dims[0],1]-input_range[input_dims[0],0], input_range[input_dims[1],1]-input_range[input_dims[1],0],
                            fc='none', linewidth=3,edgecolor=color)
            self.animate_axes.add_patch(rect)
            # self.default_patches[1].append(rect)
        else:
            raise NotImplementedError
      
        # Reachable sets
        if prob_list is None:
           color = 'blue'
           fc_color='none'
        else:
           color = 'blue'
           fc_color='none'
           alpha=0.17
        if isinstance(output_constraint, PolytopeOutputConstraint):
            # TODO: this doesn't use the computed input_dims...
            for i in range(len(b_out)):
                vertices = pypoman.compute_polygon_hull(A_out, b_out[i])
                self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                    color=color, label='$\mathcal{R}_'+str(i+1)+'$')
        elif isinstance(output_constraint, LpOutputConstraint):
            if prob_list is None:
                for output_range_ in output_range:
                    rect = Rectangle(output_range_[input_dims,0], output_range_[input_dims[0],1]-output_range_[input_dims[0],0], output_range_[input_dims[1],1]-output_range_[input_dims[1],0],
                       fc=fc_color, linewidth=3,edgecolor=color)
                    self.animate_axes.add_patch(rect)

            else:    
                for output_range_,prob in zip(output_range,prob_list):
                    fc_color=cm.get_cmap('Greens')(prob)
                    rect = Rectangle(output_range_[input_dims,0], output_range_[input_dims[0],1]-output_range_[input_dims[0],0], output_range_[input_dims[1],1]-output_range_[input_dims[1],0],
                        fc= fc_color, alpha = alpha ,linewidth=3,edgecolor=None)
                    self.animate_axes.add_patch(rect)

                 

        else:
            raise NotImplementedError


        # self.default_patches = [[], []]
        # self.default_lines = [[], []]
        # self.default_patches[0] = [input_rect]
        
        # # Exact output range
        # color = 'black'
        # linewidth = 3
        # if self.interior_condition == "linf":
        #     output_range_exact = self.samples_to_range(sampled_outputs)
        #     output_range_exact_ = output_range_exact[self.output_dims_]
        #     rect = Rectangle(output_range_exact_[:2,0], output_range_exact_[0,1]-output_range_exact_[0,0], output_range_exact_[1,1]-output_range_exact_[1,0],
        #                     fc='none', linewidth=linewidth,edgecolor=color,
        #                     label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].add_patch(rect)
        #     self.default_patches[1].append(rect)
        # elif self.interior_condition == "lower_bnds":
        #     output_range_exact = self.samples_to_range(sampled_outputs)
        #     output_range_exact_ = output_range_exact[self.output_dims_]
        #     line1 = self.animate_axes[1].axhline(output_range_exact_[1,0], linewidth=linewidth,color=color,
        #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     line2 = self.animate_axes[1].axvline(output_range_exact_[0,0], linewidth=linewidth,color=color)
        #     self.default_lines[1].append(line1)
        #     self.default_lines[1].append(line2)
        # elif self.interior_condition == "convex_hull":
        #     from scipy.spatial import ConvexHull
        #     self.true_hull = ConvexHull(sampled_outputs)
        #     self.true_hull_ = ConvexHull(sampled_outputs[...,output_dims].squeeze())
        #     line = self.animate_axes[1].plot(
        #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[0]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[0]]),
        #         np.append(sampled_outputs[self.true_hull_.vertices][...,output_dims[1]], sampled_outputs[self.true_hull_.vertices[0]][...,output_dims[1]]),
        #         color=color, linewidth=linewidth,
        #         label="True Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.default_lines[1].append(line[0])
        # else:
        #     raise NotImplementedError

    def visualize(self, M, interior_M, output_constraint, prob_list =None, iteration=0):
        if isinstance(output_constraint, PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
        elif isinstance(output_constraint, LpOutputConstraint):
            pass
        else:
            raise NotImplementedError

        # self.animate_axes.patches = self.default_patches[0].copy()
        # self.animate_axes.lines = self.default_lines[0].copy()
        input_dims_ = self.input_dims_
        output_dims_ = self.output_dims_


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
            if prob_list is None:
               color = 'grey'
            else:
               color = 'none'

        
            if isinstance(output_constraint, PolytopeOutputConstraint):
                for i in range(len(output_range_)):
                    vertices = pypoman.compute_polygon_hull(A_out, output_range_[i])
                    self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
                        color=color, label='$\mathcal{R}_'+str(i+1)+'$')
            elif isinstance(output_constraint, LpOutputConstraint):
                for output_range__ in output_range_:
                    rect = Rectangle(output_range__[:2,0], output_range__[output_dims_[0][0],1]-output_range__[output_dims_[0][0],0], output_range__[output_dims_[1][0],1]-output_range__[output_dims_[1][0],0],
                                fc='none', linewidth=1,edgecolor=color)
                    self.animate_axes.add_patch(rect)
                pass
            else:
                raise NotImplementedError

            color = "tab:purple"
            rect = Rectangle(input_range_[:,0], input_range_[input_dims_[0][0],1]-input_range_[input_dims_[0][0],0], input_range_[input_dims_[1][0],1]-input_range_[input_dims_[1][0],0],
                    fc='none', linewidth=1,edgecolor=color)
            self.animate_axes.add_patch(rect)

            # vertices = pypoman.compute_polygon_hull(A_out, input_range[i])
            # bnd_color = 'k--'
            # self.animate_axes.plot([v[0] for v in vertices]+[vertices[0][0]], [v[1] for v in vertices]+[vertices[0][1]],
            #     bnd_color, label='$\mathcal{R}_'+str(i+1)+'$')

        # # Rectangles that are within the sim pts
        # for (input_range_, output_range_) in interior_M:
        #     output_range__ = output_range_[self.output_dims_]
        #     rect = Rectangle(output_range__[:2,0], output_range__[0,1]-output_range__[0,0], output_range__[1,1]-output_range__[1,0],
        #             fc='none', linewidth=1,edgecolor='tab:purple')
        #     self.animate_axes[1].add_patch(rect)

        #     input_range__ = input_range_[input_dims_]
        #     rect = Rectangle(input_range__[:,0], input_range__[0,1]-input_range__[0,0], input_range__[1,1]-input_range__[1,0],
        #             fc='none', linewidth=1,edgecolor='tab:purple')
        #     self.animate_axes[0].add_patch(rect)

        # linewidth = 2
        # color = 'tab:green'
        # if self.interior_condition == "linf":
        #     # Make a rectangle for the estimated boundaries
        #     output_range_estimate = self.squash_down_to_one_range(u_e, M)
        #     output_range_estimate_ = output_range_estimate[self.output_dims_]
        #     rect = Rectangle(output_range_estimate_[:2,0], output_range_estimate_[0,1]-output_range_estimate_[0,0], output_range_estimate_[1,1]-output_range_estimate_[1,0],
        #                     fc='none', linewidth=linewidth,edgecolor=color,
        #                     label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].add_patch(rect)
        # elif self.interior_condition == "lower_bnds":
        #     output_range_estimate = self.squash_down_to_one_range(u_e, M)
        #     output_range_estimate_ = output_range_estimate[self.output_dims_]
        #     self.animate_axes[1].axhline(output_range_estimate_[1,0],
        #         linewidth=linewidth,color=color,
        #         label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
        #     self.animate_axes[1].axvline(output_range_estimate_[0,0],
        #         linewidth=linewidth,color=color)
        # elif self.interior_condition == "convex_hull":
        #     from scipy.spatial import ConvexHull
        #     M_ = [(input_range_, output_range_[self.output_dims_]) for (input_range_, output_range_) in M]
        #     hull = self.squash_down_to_convex_hull(M_, self.true_hull_.points)
        #     self.animate_axes[1].plot(
        #         np.append(hull.points[hull.vertices,0], hull.points[hull.vertices[0],0]),
        #         np.append(hull.points[hull.vertices,1], hull.points[hull.vertices[0],1]),
        #         color=color, linewidth=linewidth,
        #         label="Estimated Bounds ({})".format(label_dict[self.interior_condition]))
        # else:
        #     raise NotImplementedError

        # if self.show_animation:
        #     plt.pause(0.01)

        # animation_save_dir = "{}/results/tmp/".format(os.path.dirname(os.path.abspath(__file__)))
        # os.makedirs(animation_save_dir, exist_ok=True)
        # plt.savefig(animation_save_dir+"tmp_{}.png".format(str(iteration).zfill(6)))


class ClosedLoopNoPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)

class ClosedLoopUniformPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=16):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    def get_one_step_reachable_set(self, input_constraint, output_constraint, propagator, num_partitions=None):
        reachable_set, info, prob = self.get_reachable_set(input_constraint, output_constraint, propagator, t_max=1, num_partitions=num_partitions)
        return reachable_set, info, prob

    def get_reachable_set(self, input_constraint, output_constraint, propagator, t_max, num_partitions=None):

        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # only used to compute slope in non-closedloop manner...
            input_polytope_verts = pypoman.duality.compute_polytope_vertices(A_inputs, b_inputs)
            input_range = np.empty((A_inputs.shape[1],2))
            input_range[:,0] = np.min(np.stack(input_polytope_verts), axis=0)
            input_range[:,1] = np.max(np.stack(input_polytope_verts), axis=0)

        elif isinstance(input_constraint, LpInputConstraint):
            input_range = input_constraint.range
        else:
            raise NotImplementedError

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
        reachable_set = None

        for element in product(*[range(num) for num in num_partitions.flatten()]):
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[...,0] = input_range[...,0]+np.multiply(element_, slope)
            input_range_[...,1] = input_range[...,0]+np.multiply(element_+1, slope)

            if isinstance(input_constraint, PolytopeInputConstraint):
                # This is a disaster hack to partition polytopes
                A_rect, b_rect = init_state_range_to_polytope(input_range_)
                rectangle_verts = pypoman.polygon.compute_polygon_hull(A_rect, b_rect)
                input_polytope_verts = pypoman.polygon.compute_polygon_hull(A_inputs, b_inputs)
                partition_verts = pypoman.intersection.intersect_polygons(input_polytope_verts, rectangle_verts)
                A_inputs_, b_inputs_ = pypoman.duality.compute_polytope_halfspaces(partition_verts)
                input_constraint_ = input_constraint.__class__(A_inputs_, b_inputs_)
            elif isinstance(input_constraint, LpInputConstraint):
                input_constraint_ = input_constraint.__class__(range=input_range_, p=input_constraint.p)
            else:
                raise NotImplementedError

            output_constraint_, info= propagator.get_reachable_set(input_constraint_, deepcopy(output_constraint), t_max)
            num_propagator_calls += t_max

            if isinstance(output_constraint, PolytopeOutputConstraint):
                reachable_set_ = [o.b for o in output_constraint_]
                if output_constraint.b is None:
                    output_constraint.b = np.stack(reachable_set_)

                tmp = np.dstack([output_constraint.b, np.stack(reachable_set_)])
                output_constraint.b = np.max(tmp, axis=-1)
                
                ranges.append((input_range_, reachable_set_))
            elif isinstance(output_constraint, LpOutputConstraint):
                reachable_set_ = [o.range for o in output_constraint_]
                if output_constraint.range is None:
                    output_constraint.range = np.stack(reachable_set_)
  
                tmp = np.stack([output_constraint.range, np.stack(reachable_set_)], axis=-1)
               
                output_constraint.range[...,0] = np.min(tmp[...,0,:], axis=-1)
                output_constraint.range[...,1] = np.max(tmp[...,1,:], axis=-1)
                ranges.append((input_range_, np.stack(reachable_set_)))
            else:
                raise NotImplementedError

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)
        prob_list =None
        return output_constraint, info,prob_list
class ClosedLoopProbabilisticPartitioner(ClosedLoopPartitioner):

    def __init__(self, dynamics, num_partitions=16):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    def get_one_step_reachable_set(self, input_constraint, output_constraint, propagator, num_partitions=None):
        reachable_set, info, prob = self.get_reachable_set(input_constraint, output_constraint, propagator, t_max=1, num_partitions=num_partitions)
        return reachable_set, info, prob

    def get_reachable_set(self, input_constraint, output_constraint, propagator, t_max, num_partitions=None):

        if isinstance(input_constraint, PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # only used to compute slope in non-closedloop manner...
            input_polytope_verts = pypoman.duality.compute_polytope_vertices(A_inputs, b_inputs)
            input_range = np.empty((A_inputs.shape[1],2))
            input_range[:,0] = np.min(np.stack(input_polytope_verts), axis=0)
            input_range[:,1] = np.max(np.stack(input_polytope_verts), axis=0)
           
        elif isinstance(input_constraint, LpInputConstraint):
            input_range = input_constraint.range
        else:
            raise NotImplementedError
        likelihood_set=[]
        input_range_0 = input_range
        info = {}
        num_propagator_calls = 0
        offset = 0
        input_shape = input_range.shape[:-1]
        sampled_output_ranges = self.get_sampled_out_range(input_constraint, propagator, t_max, num_samples =50)
        output_constraint_=[]

        if num_partitions is None:
           num_partitions = 7
        output_w_likelihood_onestep = [(input_range,1)]
        num_layers = t_max*num_partitions
       # slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)
        for t in range(t_max):
            output_w_likelihood =[]   

            sampled_output_range = sampled_output_ranges[t,:,:]
            for (input_range,likelihood_output) in output_w_likelihood_onestep:
                ranges = []
                reachable_set = None
                output_constraint_0 = None
       
                if isinstance(input_constraint, PolytopeInputConstraint):
                    raise NotImplementedError
              
                elif isinstance(input_constraint, LpInputConstraint):
                    input_constraint_ = input_constraint.__class__(range=input_range, p=input_constraint.p)
                else:
                    raise NotImplementedError

                output_constraint_init, info_ = propagator.get_reachable_set(input_constraint_, deepcopy(output_constraint), 1)
         
                num_propagator_calls+=1
                if likelihood_output is None:
                    likelihood_output =np.empty(num_partitions)

                if t==0:
                    idx =num_partitions
                    range_diff = output_constraint_init[0].range-sampled_output_range
                    layer_slope = range_diff/num_partitions
                    output_constraint_0 =sampled_output_range
                    for layer_idx in range(num_partitions):
                        output_constraint_0=layer_slope+output_constraint_0
                        output_w_likelihood.append((output_constraint_0,1-layer_idx/num_partitions-offset))
                else:
                   

 
                    range_diff = output_constraint_init[0].range-sampled_output_range
                    range_diff[...,0] = np.min(range_diff[...,0,:], axis=-1)
                    range_diff[...,1] = np.max(range_diff[...,1,:], axis=-1)
                    likelihood_output*= 1-min(1,np.max(range_diff)+offset)             
                    output_w_likelihood.append((output_constraint_init[0].range,likelihood_output))
         
        ## compare sampled output with the estimated one
        ## partition the area between estimated and sampled into layers
        ### assign gaussian distribution to the layers
        ## each layer is propagated with associated probability

            if t!=0:
                output_w_likelihood.append((sampled_output_range,1-offset) )   
            output_w_likelihood_onestep = output_w_likelihood.copy()   
        if isinstance(output_constraint, PolytopeOutputConstraint):
            raise NotImplementedError
        elif isinstance(output_constraint, LpOutputConstraint):
            #reachable_set_ = output_constraint_
            reachable_set_=[]
            output_onestep_range = output_w_likelihood_onestep.copy()
            while  output_w_likelihood_onestep!=[]:
                reach_set,likelihood_final=output_w_likelihood_onestep.pop()
                reachable_set_.append(reach_set)
                likelihood_set.append(likelihood_final)
            if output_constraint.range is None:
                output_constraint.range= np.stack(reachable_set_)
            tmp = np.stack([output_constraint.range, np.stack(reachable_set_)], axis=-1)
            output_constraint.range[...,0] = np.min(tmp[...,0,:], axis=-1)
            output_constraint.range[...,1] = np.max(tmp[...,1,:], axis=-1)
            ranges.append((input_range_0, np.stack(reachable_set_)))
        else:
            raise NotImplementedError

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)
        return output_constraint, info, likelihood_set