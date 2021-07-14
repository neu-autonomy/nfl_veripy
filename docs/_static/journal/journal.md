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

If you want to get an animated 3D plot, add the `--make_animation` flag:

![animation](docs/_static/journal/3d_quadrotor/ClosedLoopNoPartitioner.gif)