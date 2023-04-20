# `Backward Reachability for Neural Feedback Loops`

This code is based on ideas from the following paper:
* Nicholas Rober, Michael Everett, Jonathan P. How, ["Backward Reachability Analysis for Neural Feedback Loops"](https://arxiv.org/abs/2204.08319).

## About

We discuss a technique that can be used to determine over-approximations to the backprojection set, i.e., the set of states that lead to a given target set, for neural feedback loops.

## Reproduce Figures from CDC 2022 Paper

### Figure 3

Figure 3a (BReach-LP vs ReBReach-LP):

```bash
python -m nfl_veripy.example_backward \
    --propagator CROWNNStep \
    --partitioner None \
    --state_feedback \
    --show_plot \
    --overapprox \
    --show_BReach \
    --system double_integrator \
    --t_max 5 \
    --num_partitions [4,4] \
    --plot_lims [[-3.8,5.64],[-0.64,2.5]] \
    --show_convex_hulls \
    --show_BReach
```

Figure 3b (BReach-LP vs ReBReach-LP Error Trend):

```bash
python -m nfl_veripy.backward_experiments
```



Fig 3a | Fig 3b
------------ | -------------
BReach-LP vs ReBReach-LP | Error Trend
![Fig. 3a](/docs/_static/cdc22/fig3/double_integrator_r1.png) | ![Fig. 3b](/docs/_static/cdc22/fig3/runtime_vs_error_2022_05_30__16_03_07_timestep.png)

**Notice that figure 3b will not appear in a pop-up window, but rather is stored in nfl_veripy/results/logs**

Figure 4a:

```bash
python -m nfl_veripy.example \
    --propagator CROWN \
    --partitioner Uniform \
    --state_feedback \
    --show_plot \
    --system ground_robot \
    --controller complex_potential_field \
    --t_max 9 \
    --plot_lims [[-7.2,3],[-7.2,7.2]] \
    --num_partitions [4,4] \
    --init_state_range [[-5.5,-4.5],[.5,1.5]] \
    --final_state_range [[-1,1],[-1,1]] \
    --show_policy \
    --show_trajectories \
    --show_obs
```

Figure 4b:

```bash
python -m nfl_veripy.example \
    --propagator CROWN \
    --partitioner Uniform \
    --state_feedback \
    --show_plot \
    --system ground_robot \
    --controller complex_potential_field \
    --t_max 9 \
    --plot_lims [[-7.2,3],[-7.2,7.2]] \
    --num_partitions [4,4] \
    --init_state_range [[-5.5,-4.5],[-0.5,0.5]] \
    --final_state_range [[-1,1],[-1,1]] \
    --show_trajectories \
    --show_obs
```

Figure 4c:

```bash
python -m nfl_veripy.example_backward \
    --propagator CROWN \
    --partitioner None \
    --state_feedback \
    --show_plot \
    --overapprox \
    --system ground_robot \
    --controller complex_potential_field \
    --t_max 9 \
    --plot_lims [[-7.2,3],[-7.2,7.2]] \
    --num_partitions [4,4] \
    --init_state_range [[-5.5,-4.5],[-0.5,0.5]] \
    --final_state_range [[-1,1],[-1,1]] \
    --show_trajectories
```

Fig 4a | Fig 4b | Fig 4c
------------ | ------------- | -------------
Nominal Forward Reachability | Bifurcating Forward Reachability | Bifurcating Backward Reachability
![Fig. 4a](/docs/_static/cdc22/fig4/forward_reach_nominal_r2.png) | ![Fig. 4b](/docs/_static/cdc22/fig4/forward_reach_bifurcation_r2.png) | ![Fig. 4c](/docs/_static/cdc22/fig4/backward_reach_bifurcation_r2.png)

Figure 5:

```bash
python -m nfl_veripy.example_backward \
    --propagator CROWNNStep \
    --partitioner None \
    --state_feedback \
    --show_plot \
    --overapprox \
    --system ground_robot \
    --controller buggy_complex_potential_field \
    --t_max 6 \
    --plot_lims [[-7.2,3],[-7.2,7.2]] \
    --num_partitions [8,8] \
    --init_state_range [[-5.5,-4.5],[.5,1.5]] \
    --final_state_range [[-1,1],[-1,1]] \
    --show_policy \
    --show_trajectories
```

![Fig. 5](/docs/_static/cdc22/fig5/backward_lgr_buggy_demo_r1.png)
