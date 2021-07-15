## Reproduce Figures from journal paper (extending ICRA 21)

### Backprojection

```bash
python -m nn_closed_loop.example_backward \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator \
	--state_feedback \
	--show_plot \
	--boundaries polytope \
	--num_partitions "[10, 10]"
```

You can change `--num_partitions` to get the various fidelities.

2x2 | 4x4 | 8x8
------------ | ------------- | -------------
![2x2](docs/_static/journal/backreach/double_integrator_None_CROWN_polytope_8_partitions_2_2.png) | ![4x4](docs/_static/journal/backreach/double_integrator_None_CROWN_polytope_8_partitions_4_4.png) | ![8x8](docs/_static/journal/backreach/double_integrator_None_CROWN_polytope_8_partitions_8_8.png)


### 3D Quadrotor Plot

Within `example.py`, we set `inputs_to_highlight` to have 3 components by default for the quadrotor system, which tells the plotting scripts to make a 3D plot (rather than the 2D plots for the double integrator):
```bash
python -m nn_closed_loop.example \
	--partitioner None \
	--propagator CROWN \
	--system quadrotor \
	--state_feedback \
	--t_max 1.2 \
	--save_plot --show_plot \
	--boundaries lp \
	--plot_aspect equal
```
Note that this doesn't affect the reachable set calculations (all dimensions' rechability is computed), just the visualization.

You can change `--num_partitions` to get the various fidelities.

State Feedback | Output Feedback
------------ | -------------
![2x2](docs/_static/journal/3d_quadrotor/quadrotor_None_CROWN_tmax_1.2_lp_8_state_feedback.png) | ![4x4](docs/_static/journal/3d_quadrotor/quadrotor_None_CROWN_tmax_1.2_lp_8_output_feedback.png)

If you want to get an animated 3D plot, add the `--make_animation` flag:

![animation](docs/_static/journal/3d_quadrotor/ClosedLoopNoPartitioner.gif)

### Partitioner Comparison

You can change the `--partitioner` flag to get various reachable set estimates:
```bash
python -m nn_closed_loop.example \
	--partitioner GreedySimGuided \
	--propagator CROWN \
	--system double_integrator \
	--state_feedback \
	--t_max 5 \
	--skip_show_plot \
	--make_animation
```

You can change which timestep GSG optimizes for by going into `ClosedLoopGreedySimGuidedPartitioner.py` method `grab_from_M` and changing the commented value (sorry for the major hack).

UnGuided | SimGuided | GreedySimGuided-0 | GreedySimGuided-4
------------ | ------------- | ------------ | -------------
![UnGuided](docs/_static/journal/partitioners/ClosedLoopUnGuidedPartitioner.gif) | ![SimGuided](docs/_static/journal/partitioners/ClosedLoopSimGuidedPartitioner.gif) | ![GreedySimGuided-0](docs/_static/journal/partitioners/ClosedLoopGreedySimGuidedPartitioner0.gif) | ![GreedySimGuided-4](docs/_static/journal/partitioners/ClosedLoopGreedySimGuidedPartitioner4.gif)

