## Reproduce Figures from submitted ICRA 2021 Paper

### Figure 3

Reach-SDP (Final Step Error: `206`):
```bash
python -m closed_loop.example \
	--partitioner None \
	--propagator SDP \
	--system double_integrator_mpc \
	--state_feedback \
	--t_max 5 \
	--save_plot --skip_show_plot
```
Note: If SDP isn't working, consider using a different solver and/or making it verbose (look for `prob.solve(verbose=False, solver=cp.MOSEK)` in `ClosedLoopSDPPropagator`).

Reach-SDP-Partition (Final Step Error: `19.35`):
```bash
python -m closed_loop.example \
	--partitioner Uniform \
	--propagator SDP \
	--system double_integrator_mpc \
	--state_feedback \
	--t_max 5 \
	--save_plot --skip_show_plot
```

Reach-LP (Final Step Error: `848`):
```bash
python -m closed_loop.example \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator_mpc \
	--state_feedback \
	--t_max 5 \
	--save_plot --skip_show_plot
```

Reach-LP-Partition (Final Step Error: `19.87`):
```bash
python -m closed_loop.example \
	--partitioner Uniform \
	--propagator CROWN \
	--system double_integrator_mpc \
	--state_feedback \
	--t_max 5 \
	--save_plot --skip_show_plot
```

Add the `--estimate_runtime` flag to any of those to see how long it takes to run each algorithm (avg. over 5 trials) on your machine!

Reach-SDP | Reach-SDP-Partition | Reach-LP | Reach-LP-Partition
------------ | ------------- | ------------- | -------------
![Reach-SDP](docs/_static/icra21/fig_3/double_integrator_mpc_None_SDP.png) | ![Reach-SDP-Partition](docs/_static/icra21/fig_3/double_integrator_mpc_Uniform_SDP.png) | ![Reach-LP](docs/_static/icra21/fig_3/double_integrator_mpc_None_CROWN.png) | ![Reach-LP-Partition](docs/_static/icra21/fig_3/double_integrator_mpc_Uniform_CROWN.png)

### Figure 4

For Fig. 4b, use a polytope description of the reachable sets with different numbers of facets (change `--num_polytope_facets`!):
```bash
python -m closed_loop.example \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator_mpc \
	--state_feedback \
	--t_max 8 \
	--save_plot --skip_show_plot \
	--boundaries polytope \
	--num_polytope_facets 4 \
	--init_state_range "[[-2., -1.5], [0.4, 0.8]]"
```

4-Polytope (Rectangle) | 8-Polytope | 35-Polytope
------------ | ------------- | -------------
![4-Polytope](docs/_static/icra21/fig_4/double_integrator_mpc_None_CROWN_tmax_4.0_polytope_4.png) | ![8-Polytope](docs/_static/icra21/fig_4/double_integrator_mpc_None_CROWN_tmax_4.0_polytope_8.png) | ![35-Polytope](docs/_static/icra21/fig_4/double_integrator_mpc_None_CROWN_tmax_4.0_polytope_35.png)

### Figure 5