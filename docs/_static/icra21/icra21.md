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


