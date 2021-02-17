import numpy as np
import nn_partition.partitioners as partitioners
import pypoman
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import nn_closed_loop.constraints as constraints
from copy import deepcopy


class ClosedLoopPartitioner(partitioners.Partitioner):
    def __init__(self, dynamics):
        partitioners.Partitioner.__init__(self)
        self.dynamics = dynamics

    def get_one_step_reachable_set(
        self, input_constraint, output_constraint, propagator
    ):
        output_constraint, info = propagator.get_one_step_reachable_set(
            input_constraint, deepcopy(output_constraint)
        )
        return output_constraint, info

    def get_reachable_set(
        self, input_constraint, output_constraint, propagator, t_max
    ):
        output_constraint_, info = propagator.get_reachable_set(
            input_constraint, deepcopy(output_constraint), t_max
        )

        # TODO: this is repeated from UniformPartitioner... make more universal
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            reachable_set_ = [o.b for o in output_constraint_]
            if output_constraint.b is None:
                output_constraint.b = np.stack(reachable_set_)

            tmp = np.dstack([output_constraint.b, np.stack(reachable_set_)])
            output_constraint.b = np.max(tmp, axis=-1)

            # ranges.append((input_range_, reachable_set_))
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            reachable_set_ = [o.range for o in output_constraint_]
            if output_constraint.range is None:
                output_constraint.range = np.stack(reachable_set_)

            tmp = np.stack(
                [output_constraint.range, np.stack(reachable_set_)], axis=-1
            )
            output_constraint.range[..., 0] = np.min(tmp[..., 0, :], axis=-1)
            output_constraint.range[..., 1] = np.max(tmp[..., 1, :], axis=-1)

            # ranges.append((input_range_, np.stack(reachable_set_)))
        else:
            raise NotImplementedError

        return output_constraint, info

    def get_output_range(self, input_constraint, output_constraint):

        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
            t_max = len(b_out)
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            output_range = output_constraint.range
            output_p = output_constraint.p
            t_max = len(output_range)
        else:
            raise NotImplementedError
        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
            num_states = A_inputs.shape[-1]
        elif isinstance(input_constraint, constraints.LpInputConstraint):
            input_range = input_constraint.range
            input_p = input_constraint.p
            num_states = input_range.shape[0]
        else:
            raise NotImplementedError
        return output_range

    def get_error(
        self, input_constraint, output_constraint, propagator, t_max
    ):

        if isinstance(input_constraint, constraints.LpInputConstraint):
            output_estimated_range = output_constraint.range
            # t_max = len(output_estimated_range)
            output_range_exact = self.get_sampled_out_range(
                input_constraint, propagator, t_max, num_samples=1000
            )
            error = 0
            for t in range(int(t_max / self.dynamics.dt)):
                true_area = np.product(
                    output_range_exact[t][..., 1]
                    - output_range_exact[t][..., 0]
                )
                estimated_area = np.product(
                    output_estimated_range[t][..., 1]
                    - output_estimated_range[t][..., 0]
                )
                error += (estimated_area - true_area) / true_area
        else:
            raise NotImplementedError
        final_error = (estimated_area - true_area) / true_area
        avg_error = error / t_max
        return final_error, avg_error

    def get_sampled_out_range(
        self, input_constraint, propagator, t_max=5, num_samples=1000
    ):
        return self.dynamics.get_sampled_output_range(
            input_constraint, t_max, num_samples, controller=propagator.network
        )

    def setup_visualization_multiple(
        self,
        input_constraint,
        output_constraint,
        propagator,
        input_dims_,
        prob_list=None,
        show_samples=True,
        outputs_to_highlight=None,
        color="g",
        line_style="-",
    ):
        input_dims = input_dims_
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
            t_max = len(b_out)
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            output_range = output_constraint.range
            output_p = output_constraint.p
            output_prob = prob_list
            t_max = len(output_range)
        else:
            raise NotImplementedError
        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
            num_states = A_inputs.shape[-1]
            output_prob = prob_list

        elif isinstance(input_constraint, constraints.LpInputConstraint):
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
        # if show_samples:
        #    self.animate_axes.scatter(sampled_outputs[...,output_dims[0]], sampled_outputs[...,output_dims[1]], c='k', marker='.', zorder=2,
        #        label="Sampled States")

        linewidth = 2
        if show_samples:
            self.dynamics.show_samples(
                t_max * self.dynamics.dt,
                input_constraint,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=input_dims,
            )

        # Initial state set
        init_state_color = "k"

        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            # TODO: this doesn't use the computed input_dims...
            try:
                vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
            except:
                print(
                    "[warning] Can't visualize polytopic input constraints for >2 states. Need to implement this to it extracts input_dims."
                )
                raise NotImplementedError
            self.animate_axes.plot(
                [v[0] for v in vertices] + [vertices[0][0]],
                [v[1] for v in vertices] + [vertices[0][1]],
                color=color,
                linewidth=linewidth,
                linestyle=line_style,
                label="Initial States",
            )
        elif isinstance(input_constraint, constraints.LpInputConstraint):
            rect = Rectangle(
                input_range[input_dims, 0],
                input_range[input_dims[0], 1] - input_range[input_dims[0], 0],
                input_range[input_dims[1], 1] - input_range[input_dims[1], 0],
                fc="none",
                linewidth=linewidth,
                linestyle=line_style,
                edgecolor=init_state_color,
            )
            self.animate_axes.add_patch(rect)
            # self.default_patches[1].append(rect)
        else:
            raise NotImplementedError

        linewidth = 1.5
        # Reachable sets
        if prob_list is None:
            fc_color = "none"
        else:
            fc_color = "none"
            alpha = 0.17
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            # TODO: this doesn't use the computed input_dims...
            for i in range(len(b_out)):
                vertices = pypoman.compute_polygon_hull(A_out, b_out[i])
                self.animate_axes.plot(
                    [v[0] for v in vertices] + [vertices[0][0]],
                    [v[1] for v in vertices] + [vertices[0][1]],
                    color=color,
                    label="$\mathcal{R}_" + str(i + 1) + "$",
                )
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            if prob_list is None:
                for output_range_ in output_range:
                    rect = Rectangle(
                        output_range_[input_dims, 0],
                        output_range_[input_dims[0], 1]
                        - output_range_[input_dims[0], 0],
                        output_range_[input_dims[1], 1]
                        - output_range_[input_dims[1], 0],
                        fc=fc_color,
                        linewidth=linewidth,
                        linestyle=line_style,
                        edgecolor=color,
                    )
                    self.animate_axes.add_patch(rect)

            else:
                for output_range_, prob in zip(output_range, prob_list):
                    fc_color = cm.get_cmap("Greens")(prob)
                    rect = Rectangle(
                        output_range_[input_dims, 0],
                        output_range_[input_dims[0], 1]
                        - output_range_[input_dims[0], 0],
                        output_range_[input_dims[1], 1]
                        - output_range_[input_dims[1], 0],
                        fc=fc_color,
                        alpha=alpha,
                        linewidth=linewidth,
                        linestyle=line_style,
                        edgecolor=None,
                    )
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

    def setup_visualization(
        self,
        input_constraint,
        output_constraint,
        propagator,
        show_samples=True,
        outputs_to_highlight=None,
        inputs_to_highlight=None,
        aspect="auto",
    ):
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
            t_max = len(b_out)
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            output_range = output_constraint.range
            output_p = output_constraint.p
            t_max = len(output_range)
        else:
            raise NotImplementedError
        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b
            num_states = A_inputs.shape[-1]

        elif isinstance(input_constraint, constraints.LpInputConstraint):
            input_range = input_constraint.range
            input_p = input_constraint.p
            num_states = input_range.shape[0]
        else:
            raise NotImplementedError

        self.animate_fig, self.animate_axes = plt.subplots(1, 1)

        if inputs_to_highlight is None:
            # Automatically detect which input dims to show based on input_range
            # num_input_dimensions_to_plot = 2
            # input_shape = A_inputs.
            # lengths = input_range[...,1].flatten() - input_range[...,0].flatten()
            # flat_dims = np.argpartition(lengths, -num_input_dimensions_to_plot)[-num_input_dimensions_to_plot:]
            # flat_dims.sort()
            input_dims = [[0], [1]]
            # input_dims = [np.unravel_index(flat_dim, input_range.shape[:-1]) for flat_dim in flat_dims]
            input_names = [
                "State: {}".format(input_dims[0][0]),
                "State: {}".format(input_dims[1][0]),
            ]
        else:
            input_dims = [x["dim"] for x in inputs_to_highlight]
            input_names = [x["name"] for x in inputs_to_highlight]
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
            output_names = [
                "State: {}".format(output_dims[0][0]),
                "State: {}".format(output_dims[1][0]),
            ]
        else:
            output_dims = [x["dim"] for x in outputs_to_highlight]
            output_names = [x["name"] for x in outputs_to_highlight]
        #  self.input_dims_ = tuple([tuple([input_dims[j][i] for j in range(len(input_dims))]) for i in range(len(input_dims[0]))])
        self.output_dims_ = output_dims
        # scale = 0.05
        # x_off = max((input_range[input_dims[0]+(1,)] - input_range[input_dims[0]+(0,)])*(scale), 1e-5)
        # y_off = max((input_range[input_dims[1]+(1,)] - input_range[input_dims[1]+(0,)])*(scale), 1e-5)
        # self.animate_axes[0].set_xlim(input_range[input_dims[0]+(0,)] - x_off, input_range[input_dims[0]+(1,)]+x_off)
        # self.animate_axes[0].set_ylim(input_range[input_dims[1]+(0,)] - y_off, input_range[input_dims[1]+(1,)]+y_off)
        self.animate_axes.set_xlabel(input_names[0])
        self.animate_axes.set_ylabel(input_names[1])

        self.animate_axes.set_aspect(aspect)

        if show_samples:
            self.dynamics.show_samples(
                t_max * self.dynamics.dt,
                input_constraint,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=input_dims,
            )

        # Initial state set
        color = "tab:gray"
        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            # TODO: this doesn't use the computed input_dims...
            try:
                vertices = pypoman.compute_polygon_hull(A_inputs, b_inputs)
            except:
                print(
                    "[warning] Can't visualize polytopic input constraints for >2 states. Need to implement this to it extracts input_dims."
                )
                raise NotImplementedError
            self.animate_axes.plot(
                [v[0] for v in vertices] + [vertices[0][0]],
                [v[1] for v in vertices] + [vertices[0][1]],
                color=color,
                label="Initial States",
            )
        elif isinstance(input_constraint, constraints.LpInputConstraint):
            rect = Rectangle(
                input_range[input_dims, 0],
                input_range[input_dims[0], 1] - input_range[input_dims[0], 0],
                input_range[input_dims[1], 1] - input_range[input_dims[1], 0],
                fc="none",
                linewidth=3,
                edgecolor=color,
            )
            self.animate_axes.add_patch(rect)
            # self.default_patches[1].append(rect)
        else:
            raise NotImplementedError

        # Reachable sets
        color = "tab:blue"
        fc_color = "None"
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            # TODO: this doesn't use the computed input_dims...
            for i in range(len(b_out)):
                vertices = pypoman.compute_polygon_hull(A_out, b_out[i])
                self.animate_axes.plot(
                    [v[0] for v in vertices] + [vertices[0][0]],
                    [v[1] for v in vertices] + [vertices[0][1]],
                    color=color,
                    label="$\mathcal{R}_" + str(i + 1) + "$",
                )
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            for output_range_ in output_range:
                rect = Rectangle(
                    output_range_[input_dims, 0],
                    output_range_[input_dims[0], 1]
                    - output_range_[input_dims[0], 0],
                    output_range_[input_dims[1], 1]
                    - output_range_[input_dims[1], 0],
                    fc=fc_color,
                    linewidth=3,
                    edgecolor=color,
                )
                self.animate_axes.add_patch(rect)
        else:
            raise NotImplementedError

    def visualize(self, M, interior_M, output_constraint, iteration=0):
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            A_out = output_constraint.A
            b_out = output_constraint.b
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
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
                input_label = "Cell of Partition"
                output_label = "One Cell's Estimated Bounds"
                first = False
            else:
                input_label = None
                output_label = None
            color = "grey"

            if isinstance(
                output_constraint, constraints.PolytopeOutputConstraint
            ):
                for i in range(len(output_range_)):
                    vertices = pypoman.compute_polygon_hull(
                        A_out, output_range_[i]
                    )
                    self.animate_axes.plot(
                        [v[0] for v in vertices] + [vertices[0][0]],
                        [v[1] for v in vertices] + [vertices[0][1]],
                        color=color,
                        label="$\mathcal{R}_" + str(i + 1) + "$",
                    )
            elif isinstance(output_constraint, constraints.LpOutputConstraint):
                for output_range__ in output_range_:
                    rect = Rectangle(
                        output_range__[:2, 0],
                        output_range__[output_dims_[0][0], 1]
                        - output_range__[output_dims_[0][0], 0],
                        output_range__[output_dims_[1][0], 1]
                        - output_range__[output_dims_[1][0], 0],
                        fc="none",
                        linewidth=1,
                        edgecolor=color,
                    )
                    self.animate_axes.add_patch(rect)
                pass
            else:
                raise NotImplementedError

            color = "tab:purple"
            rect = Rectangle(
                input_range_[:, 0],
                input_range_[input_dims_[0][0], 1]
                - input_range_[input_dims_[0][0], 0],
                input_range_[input_dims_[1][0], 1]
                - input_range_[input_dims_[1][0], 0],
                fc="none",
                linewidth=1,
                edgecolor=color,
            )
            self.animate_axes.add_patch(rect)
