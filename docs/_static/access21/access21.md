# `nfl_veripy`

## About

* Michael Everett, Golnaz Habibi, Chuangchuang Sun, Jonathan P. How, ["Reachability Analysis of Neural Feedback Loops"](https://arxiv.org/pdf/2108.04140.pdf), in review.
* Michael Everett, Golnaz Habibi, Jonathan P. How, ["Efficient Reachability Analysis for Closed-Loop Systems with Neural Network Controllers"](https://arxiv.org/pdf/2101.01815.pdf), ICRA 2021.

Since NNs are rarely deployed in isolation, we developed a framework for analyzing closed-loop systems that employ NN control policies.
The `nfl_veripy` codebase follows a similar API as the `nn_partition` package, leveraging analogous `ClosedLoopAnalyzer`, `ClosedLoopPropagator` and `ClosedLoopPartitioner` concepts.
The typical problem statement is: given a known initial state set (and a known dynamics model), compute bounds on the reachable sets for N steps into the future.
These bounds provide a safety guarantee for autonomous systems employing NN controllers, as they guarantee that the system will never enter parts of the state space outside of the reachable set bounds.

Reach-LP-Partition | Reach-LP w/ Polytopes
----- | -----
![nn_partition_polytope](/docs/_static/icra21/other/double_integrator_Uniform_CROWN_tmax_5.0_lp_8.png) | ![nn_partition_polytope](/docs/_static/icra21/other/double_integrator_None_CROWN_tmax_4.0_polytope_35.png)


![nfl_veripy](/docs/_static/access21/partitions/ClosedLoopGreedySimGuidedPartitioner4.gif)


## Reproduce Figures from IEEE Access '21 paper (extending ICRA '21 paper)

### Backprojection

```bash
python -m nfl_veripy.example_backward \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator \
	--state_feedback \
	--show_plot \
	--boundaries polytope \
	--num_partitions "[10, 10]" \
	--plot_lims "[[1.75,3.25],[-0.3,1.25]]"
```

By default, this computes an under-approximation. You can add the `--overapprox` flag to change this to compute an over-approximation.

You can change `--num_partitions` to get the various fidelities.

2x2 | 4x4 | 8x8 | 16x16
------------ | ------------- | ------------- | -------------
![2x2](/docs/_static/access21/backreach/double_integrator_None_CROWN_polytope_8_partitions_2_2.png) | ![4x4](/docs/_static/access21/backreach/double_integrator_None_CROWN_polytope_8_partitions_4_4.png) | ![8x8](/docs/_static/access21/backreach/double_integrator_None_CROWN_polytope_8_partitions_8_8.png) | ![16x16](/docs/_static/access21/backreach/double_integrator_None_CROWN_polytope_8_partitions_16_16.png)



### 3D Quadrotor Plot

Within `example.py`, we set `inputs_to_highlight` to have 3 components by default for the quadrotor system, which tells the plotting scripts to make a 3D plot (rather than the 2D plots for the double integrator):
```bash
python -m nfl_veripy.example \
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
![2x2](/docs/_static/access21/3d_quadrotor/quadrotor_None_CROWN_tmax_1.2_lp_8_state_feedback.png) | ![4x4](/docs/_static/access21/3d_quadrotor/quadrotor_None_CROWN_tmax_1.2_lp_8_output_feedback.png)

If you want to get an animated 3D plot, add the `--make_animation` flag:

![animation](/docs/_static/access21/3d_quadrotor/ClosedLoopNoPartitioner.gif)

### Partitioner Comparison

You can change the `--partitioner` flag to get various reachable set estimates:
```bash
python -m nfl_veripy.example \
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
![UnGuided](/docs/_static/access21/partitions/ClosedLoopUnGuidedPartitioner.gif) | ![SimGuided](/docs/_static/access21/partitions/ClosedLoopSimGuidedPartitioner.gif) | ![GreedySimGuided-0](/docs/_static/access21/partitions/ClosedLoopGreedySimGuidedPartitioner0.gif) | ![GreedySimGuided-4](/docs/_static/access21/partitions/ClosedLoopGreedySimGuidedPartitioner4.gif)

### Compare Reach-LP and Reach-SDP

In `nfl_veripy/experiments.py`, at the bottom check that these are uncommented:

```python
# Like Fig 3 in ICRA21 paper
c = CompareRuntimeVsErrorTable()
c.run()
c.plot()  # 3A: table
c.plot_reachable_sets()  # 3B: overlay reachable sets
c.plot_error_vs_timestep()  # 3C: error vs timestep
```

Running `python -m nfl_veripy.experiments` will generate:
- a `.pkl` file in `nfl_veripy/results/logs`, which will then be loaded to generate...
- this table output (and the latex version of the table)
```txt
Algorithm                     Runtime [s]            Error
----------------------------  -------------------  -------
Reach-SDP~\cite{hu2020reach}  $42.571 \pm 0.538$       207
Reach-SDP-Partition           $670.257 \pm 2.913$       12
Reach-LP                      $0.017 \pm 0.000$       1590
Reach-LP-Partition            $0.263 \pm 0.001$         34
```
- and the following two plots in the same directory as the `.pkl` file:

Reachable Sets | Error per Timestep
------------ | -------------
![reachable](/docs/_static/access21/reachlp_vs_reachsdp/runtime_vs_error_2021_07_21__12_33_20_reachable.png) | ![SimGuided](/docs/_static/access21/reachlp_vs_reachsdp/runtime_vs_error_2021_07_21__12_33_20_timestep.png)


### Compare Linear Program and Closed-Form solution timings

In `nfl_veripy/experiments.py`, at the bottom check that these are uncommented:

```python
c = CompareLPvsCF(system="double_integrator")
c.run()
c.plot()
```

Running `python -m nfl_veripy.experiments` will generate:
- a `.pkl` file in `nfl_veripy/results/logs`, which will then be loaded to generate...
- this table output (and the latex version of the table)
```txt
      1                  4                  16
----  -----------------  -----------------  -----------------
L.P.  $0.229 \pm 0.018$  $0.856 \pm 0.046$  $3.308 \pm 0.091$
C.F.  $0.017 \pm 0.000$  $0.066 \pm 0.000$  $0.265 \pm 0.002$
```

You can change `system="quadrotor"` to see the corresponding table for the 6D quadrotor system.

### Other Systems

#### Duffing

```bash
python -m nfl_veripy.example --partitioner None --propagator CROWN --system duffing --state_feedback --t_max 0.3
```
will output this plot:
![duffing](/docs/_static/access21/systems/duffing_None_CROWN_tmax_0.3_lp_8.png)

#### ISS
```bash
python -m nfl_veripy.example --partitioner None --propagator CROWN --system iss --state_feedback --t_max 0.21
```
will output this plot:
![iss](/docs/_static/access21/systems/iss_None_CROWN_tmax_0.2_lp_8.png)
