# `nn_partition`

This code is based on ideas from the following paper:
* Michael Everett, Golnaz Habibi, Jonathan P. How, ["Robustness Analysis of Neural Networks via Efficient Partitioning with Applications in Control Systems"](https://doi.org/10.1109/LCSYS.2020.3045323), IEEE LCSS 2020 & ACC 2021.

## About

We introduce the concepts of `Analyzer`, `Propagator`, and `Partitioner` in our LCSS/ACC '21 paper and implement several instances of each concept as a starting point.
This modular view on NN robustness analysis essentially defines an API that decouples each component.
This decoupling enables improvements in either `Propagator` or `Partitioner` algorithms to have a wide impact across many analysis/verification problems.

![nn_partition](/docs/_static/lcss21/animations/GreedySimGuidedPartitioner.gif)

## Reproduce Figures from LCSS/ACC 2021 Paper

### Figure 4

Figure 4a (Lower Bounds):
```bash
python -m nn_partition.example \
	--partitioner GreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition lower_bnds \
	--model random_weights \
	--activation relu \
	--show_input --show_output
```

Figure 4b (Linf Ball):
```bash
python -m nn_partition.example \
	--partitioner GreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition linf \
	--model random_weights \
	--activation relu \
	--show_input --show_output
```


Figure 4c (Convex Hull):
```bash
python -m nn_partition.example \
	--partitioner GreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model random_weights \
	--activation relu \
	--show_input --show_output
```

Fig 4a | Fig 4b | Fig 4c
------------ | ------------- | -------------
Lower Bounds | Linf Ball | Convex Hull
![Fig. 4a](/docs/_static/lcss21/fig_4/random_weights_relu_GreedySimGuided_CROWN_LIRPA_interior_condition_lower_bnds_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png) | ![Fig. 4b](/docs/_static/lcss21/fig_4/random_weights_relu_GreedySimGuided_CROWN_LIRPA_interior_condition_linf_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png) | ![Fig. 4c](/docs/_static/lcss21/fig_4/random_weights_relu_GreedySimGuided_CROWN_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png)

### Figure 5

Figure 5a (SG+IBP):
```bash
python -m nn_partition.example \
	--partitioner SimGuided \
	--propagator IBP_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model random_weights \
	--activation relu \
	--input_plot_labels None None \
	--show_input --skip_show_output \
	--input_plot_aspect equal
```

Figure 5b (SG+CROWN):
```bash
python -m nn_partition.example \
	--partitioner SimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model random_weights \
	--activation relu \
	--input_plot_labels None None \
	--show_input --skip_show_output \
	--input_plot_aspect equal
```

Figure 5c (GSG+CROWN):
```bash
python -m nn_partition.example \
	--partitioner GreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model random_weights \
	--activation relu \
	--input_plot_labels None None \
	--show_input --skip_show_output \
	--input_plot_aspect equal
```

Figure 5d (GSG+CROWN):
```bash
python -m nn_partition.example \
	--partitioner AdaptiveGreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model random_weights \
	--activation relu \
	--input_plot_labels None None \
	--show_input --skip_show_output \
	--input_plot_aspect equal
```

Fig 5a | Fig 5b | Fig 5c | Fig 5d
------------ | ------------- | ------------- |  -------------
SG+IBP | SG+CROWN | GSG+IBP | GSG+CROWN
![Fig. 5a](/docs/_static/lcss21/fig_5/random_weights_relu_SimGuided_IBP_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png) | ![Fig. 5b](/docs/_static/lcss21/fig_5/random_weights_relu_SimGuided_CROWN_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png) | ![Fig. 5c](/docs/_static/lcss21/fig_5/random_weights_relu_GreedySimGuided_CROWN_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png) | ![Fig. 5d](/docs/_static/lcss21/fig_5/random_weights_relu_AdaptiveGreedySimGuided_CROWN_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png)


### Figure 6

Figure 6a (SG+IBP):
```bash
python -m nn_partition.example \
	--partitioner SimGuided \
	--propagator IBP_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model robot_arm \
	--activation tanh \
	--output_plot_labels x y \
	--output_plot_aspect equal \
	--skip_show_input
```

Figure 6b (AGSG+CROWN):
```bash
python -m nn_partition.example \
	--partitioner AdaptiveGreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition convex_hull \
	--model robot_arm \
	--activation tanh \
	--output_plot_labels x y \
	--output_plot_aspect equal \
	--skip_show_input
```

Fig 6a | Fig 6b |
------------ | -------------
SG+IBP | AGSG+CROWN
![Fig. 6a](/docs/_static/lcss21/fig_6/robot_arm_tanh_SimGuided_IBP_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png) | ![Fig. 6b](/docs/_static/lcss21/fig_6/robot_arm_tanh_AdaptiveGreedySimGuided_CROWN_LIRPA_interior_condition_convex_hull_num_simulations_10000.0_termination_condition_type_time_budget_termination_condition_value_2.0.png)

### Figure 7

This figure unfortunately requires code for the RL implementation that is under IP protection from our research sponsor.

### Figure 8

Info coming soon...

### Animations

For any of the above examples, you can add the `--make_animation` flag which will save a `.gif` in `results/animations/`, e.g.,
```bash
python -m nn_partition.example \
	--partitioner GreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 5 \
	--interior_condition lower_bnds \
	--model random_weights \
	--activation relu \
	--show_input --show_output \
	--make_animation
```

will produce something like this:
![animation](/docs/_static/lcss21/animations/GSG_CROWN_random_relu_lowerbnds.gif)

