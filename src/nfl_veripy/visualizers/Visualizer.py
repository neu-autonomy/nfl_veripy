class Visualizer:
    def __init__(
        self,
        dynamics: dynamics.Dynamics,
    ):
        self.dynamics = dynamics

        # Animation-related flags
        self.make_animation: bool = False
        self.show_animation: bool = False
        self.tmp_animation_save_dir = "{}/../../results/tmp_animation/".format(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.animation_save_dir = "{}/../../results/animations/".format(
            os.path.dirname(os.path.abspath(__file__))
        )

    def get_tmp_animation_filename(self, iteration):
        filename = self.tmp_animation_save_dir + "tmp_{}.png".format(
            str(iteration).zfill(6)
        )
        return filename

    def setup_visualization(
        self,
        initial_set: constraints.SingleTimestepConstraint,
        t_max: int,
        propagator: propagators.ClosedLoopPropagator,
        show_samples: bool = True,
        show_samples_from_cells: bool = True,
        show_trajectories: bool = False,
        axis_labels: list = ["$x_0$", "$x_1$"],
        axis_dims: list = [0, 1],
        aspect: str = "auto",
        initial_set_color: Optional[str] = None,
        initial_set_zorder: Optional[int] = None,
        extra_set_color: Optional[str] = None,
        extra_set_zorder: Optional[int] = None,
        sample_zorder: Optional[int] = None,
        sample_colors: Optional[str] = None,
        extra_constraint: Optional[
            constraints.SingleTimestepConstraint
        ] = None,
        plot_lims: Optional[list] = None,
        controller_name: Optional[str] = None,
    ) -> None:
        self.default_patches: list = []
        self.default_lines: list = []

        self.axis_dims = axis_dims

        if len(axis_dims) == 2:
            projection = None
            self.plot_2d = True
            self.linewidth = 2
        elif len(axis_dims) == 3:
            projection = "3d"
            self.plot_2d = False
            self.linewidth = 1
            aspect = "auto"

        self.animate_fig, self.animate_axes = plt.subplots(
            1, 1, subplot_kw=dict(projection=projection)
        )
        if controller_name is not None:
            from nfl_veripy.utils.controller_generation import (
                display_ground_robot_control_field,
            )

            display_ground_robot_control_field(
                name=controller_name, ax=self.animate_axes
            )

        # if controller_name is not None:
        #     from nfl_veripy.utils.controller_generation import (
        #         display_ground_robot_DI_control_field,
        #     )

        #     display_ground_robot_DI_control_field(
        #         name=controller_name, ax=self.animate_axes
        #     )

        self.animate_axes.set_aspect(aspect)

        if show_samples:
            self.dynamics.show_samples(
                t_max * self.dynamics.dt,
                initial_set,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=axis_dims,
                zorder=sample_zorder,
                colors=sample_colors,
            )

        if show_samples_from_cells:
            for initial_set_cell in initial_set.cells:
                self.dynamics.show_samples(
                    t_max * self.dynamics.dt,
                    initial_set_cell,
                    ax=self.animate_axes,
                    controller=propagator.network,
                    input_dims=axis_dims,
                    zorder=sample_zorder,
                    colors=sample_colors,
                )

        if show_trajectories:
            self.dynamics.show_trajectories(
                t_max * self.dynamics.dt,
                initial_set,
                ax=self.animate_axes,
                controller=propagator.network,
                input_dims=axis_dims,
                zorder=sample_zorder,
                colors=sample_colors,
            )

        self.animate_axes.set_xlabel(axis_labels[0])
        self.animate_axes.set_ylabel(axis_labels[1])
        if not self.plot_2d:
            self.animate_axes.set_zlabel(axis_labels[2])

        # # Plot the initial state set's boundaries
        # if initial_set_color is None:
        #     initial_set_color = "tab:grey"
        # rect = initial_set.plot(
        #     self.animate_axes,
        #     axis_dims,
        #     initial_set_color,
        #     zorder=initial_set_zorder,
        #     linewidth=self.linewidth,
        #     plot_2d=self.plot_2d,
        # )
        # self.default_patches += rect

        if show_samples_from_cells:
            for cell in initial_set.cells:
                rect = initial_set_cell.plot(
                    self.animate_axes,
                    axis_dims,
                    initial_set_color,
                    zorder=initial_set_zorder,
                    linewidth=self.linewidth,
                    plot_2d=self.plot_2d,
                )
                self.default_patches += rect

        if extra_set_color is None:
            extra_set_color = "tab:red"
        # if extra_constraint[0] is not None:
        #     for i in range(len(extra_constraint)):
        #         rect = extra_constraint[i].plot(
        #             self.animate_axes,
        #             input_dims,
        #             extra_set_color,
        #             zorder=extra_set_zorder,
        #             linewidth=self.linewidth,
        #             plot_2d=self.plot_2d,
        #         )
        #         self.default_patches += rect

    def visualize(  # type: ignore
        self,
        M: list,
        interior_M: list,
        reachable_sets: constraints.MultiTimestepConstraint,
        iteration: int = 0,
        title: Optional[str] = None,
        reachable_set_color: Optional[str] = None,
        reachable_set_zorder: Optional[int] = None,
        reachable_set_ls: Optional[str] = None,
        dont_tighten_layout: bool = False,
        plot_lims: Optional[str] = None,
    ) -> None:
        # Bring forward whatever default items should be in the plot
        # (e.g., MC samples, initial state set boundaries)
        for item in self.default_patches + self.default_lines:
            if isinstance(item, Patch):
                self.animate_axes.add_patch(item)
            elif isinstance(item, Line2D):
                self.animate_axes.add_line(item)

        self.plot_reachable_sets(
            reachable_sets,
            self.axis_dims,
            reachable_set_color=reachable_set_color,
            reachable_set_zorder=reachable_set_zorder,
            reachable_set_ls=reachable_set_ls,
        )

        if plot_lims is not None:
            import ast

            plot_lims_arr = np.array(ast.literal_eval(plot_lims))
            plt.xlim(plot_lims_arr[0])
            plt.ylim(plot_lims_arr[1])

        # Do auxiliary stuff to make sure animations look nice
        if title is not None:
            plt.suptitle(title)

        if (iteration == 0 or iteration == -1) and not dont_tighten_layout:
            plt.tight_layout()

        if self.show_animation:
            plt.pause(0.01)

        if self.make_animation and iteration is not None:
            os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
            filename = self.get_tmp_animation_filename(iteration)
            plt.savefig(filename)

        if self.make_animation and not self.plot_2d:
            # Make an animated 3d view
            os.makedirs(self.tmp_animation_save_dir, exist_ok=True)
            for i, angle in enumerate(range(-100, 0, 2)):
                self.animate_axes.view_init(30, angle)
                filename = self.get_tmp_animation_filename(i)
                plt.savefig(filename)
            self.compile_animation(i, delete_files=True, duration=0.2)

    def plot_reachable_sets(
        self,
        constraint: constraints.MultiTimestepConstraint,
        dims: list,
        reachable_set_color: Optional[str] = None,
        reachable_set_zorder: Optional[int] = None,
        reachable_set_ls: Optional[str] = None,
        reachable_set_lw: Optional[int] = None,
    ):
        if reachable_set_color is None:
            reachable_set_color = "tab:blue"
        if reachable_set_ls is None:
            reachable_set_ls = "-"
        if reachable_set_lw is None:
            reachable_set_lw = self.linewidth
        fc_color = "None"
        constraint.plot(
            self.animate_axes,
            dims,
            reachable_set_color,
            fc_color=fc_color,
            zorder=reachable_set_zorder,
            plot_2d=self.plot_2d,
            linewidth=reachable_set_lw,
            ls=reachable_set_ls,
        )

    def plot_partition(self, constraint, dims, color):
        # This if shouldn't really be necessary -- someone is calling
        # self.plot_partitions with something other than a
        # (constraint, ___) element in M?
        if isinstance(constraint, np.ndarray):
            constraint = constraints.LpConstraint(range=constraint)

        constraint.plot(
            self.animate_axes, dims, color, linewidth=1, plot_2d=self.plot_2d
        )

    def plot_partitions(
        self,
        M: list[tuple[constraints.SingleTimestepConstraint, np.ndarray]],
        dims: list,
    ) -> None:
        # first = True
        for input_constraint, output_range in M:
            # Next state constraint of that cell
            output_constraint_ = constraints.LpConstraint(range=output_range)
            self.plot_partition(output_constraint_, dims, "grey")

            # Initial state constraint of that cell
            self.plot_partition(input_constraint, dims, "tab:red")

    def compile_animation(
        self, iteration, delete_files=False, start_iteration=0, duration=0.1
    ):
        filenames = [
            self.get_tmp_animation_filename(i)
            for i in range(start_iteration, iteration)
        ]
        images = []
        for filename in filenames:
            try:
                image = imageio.imread(filename)
            except FileNotFoundError:
                # not every iteration has a plot
                continue
            images.append(image)
            if filename == filenames[-1]:
                for i in range(10):
                    images.append(imageio.imread(filename))
            if delete_files:
                os.remove(filename)

        # Save the gif in a new animations sub-folder
        os.makedirs(self.animation_save_dir, exist_ok=True)
        animation_filename = (
            self.animation_save_dir + self.get_animation_filename()
        )
        imageio.mimsave(animation_filename, images, duration=duration)
        optimize(animation_filename)  # compress gif file size

    def get_animation_filename(self):
        animation_filename = self.__class__.__name__ + ".gif"
        return animation_filename
