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
![2x2](docs/_static/journal/backreach/double_integrator_None_CROWN_polytope_8_partitions_2_2.png) | ![4x4](docs/_static/journal/backreach/double_integrator_None_CROWN_polytope_8_partitions_4_4.png) | ![8x8](docs/_static/journal/backreach/double_integrator_None_CROWN_polytope_8_partitions_8_8.png