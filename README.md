### About

This repository provides Python implementations for the robustness analysis tools in some of our recent papers. This research is supported by Ford Motor Company.

#### `nn_partition`

* Michael Everett, Golnaz Habibi, Jonathan P. How, ["Robustness Analysis of Neural Networks via Efficient Partitioning with Applications in Control Systems"](https://doi.org/10.1109/LCSYS.2020.3045323), IEEE LCSS 2020 & ACC 2021.

We introduce the concepts of `Analyzer`, `Propagator`, and `Partitioner` in our LCSS/ACC '21 paper and implement several instances of each concept as a starting point.
This modular view on NN robustness analysis essentially defines an API that decouples each component.
This decoupling enables improvements in either `Propagator` or `Partitioner` algorithms to have a wide impact across many analysis/verification problems.

![nn_partition](/docs/_static/lcss21/animations/GreedySimGuidedPartitioner.gif)

#### `nn_closed_loop`

* Michael Everett, Golnaz Habibi, Chuangchuang Sun, Jonathan P. How, ["Reachability Analysis of Neural Feedback Loops"](https://arxiv.org/pdf/2108.04140.pdf), in review.
* Michael Everett, Golnaz Habibi, Jonathan P. How, ["Efficient Reachability Analysis for Closed-Loop Systems with Neural Network Controllers"](https://arxiv.org/pdf/2101.01815.pdf), ICRA 2021.

Since NNs are rarely deployed in isolation, we developed a framework for analyzing closed-loop systems that employ NN control policies.
The `nn_closed_loop` codebase follows a similar API as the `nn_partition` package, leveraging analogous `ClosedLoopAnalyzer`, `ClosedLoopPropagator` and `ClosedLoopPartitioner` concepts.
The typical problem statement is: given a known initial state set (and a known dynamics model), compute bounds on the reachable sets for N steps into the future.
These bounds provide a safety guarantee for autonomous systems employing NN controllers, as they guarantee that the system will never enter parts of the state space outside of the reachable set bounds.

Reach-LP-Partition | Reach-LP w/ Polytopes
----- | -----
![nn_partition_polytope](/docs/_static/icra21/other/double_integrator_Uniform_CROWN_tmax_5.0_lp_8.png) | ![nn_partition_polytope](/docs/_static/icra21/other/double_integrator_None_CROWN_tmax_4.0_polytope_35.png)


![nn_closed_loop](/docs/_static/journal/partitions/ClosedLoopGreedySimGuidedPartitioner4.gif)

---

We build on excellent open-source repositories from the neural network analysis community. These repositories are imported as Git submodules or re-implemented in Python here, with some changes to reflect the slightly different problem statements:
* [`auto_LIRPA`](https://github.com/KaidiXu/auto_LiRPA)
* [`crown_ibp`](https://github.com/huanzhang12/CROWN-IBP)
* [`robust_nn`](https://github.com/arobey1/RobustNN)
* [`nnv`](https://github.com/verivital/nnv)

### Get the code

```bash
git clone --recursive <this_repo>
```

### Install

You *might* need to install these dependencies on Linux (for `cvxpy`'s SCS solver and to generate reasonably sized animation files) (did not need to on OSX):
```bash
sudo apt-get install libblas-dev liblapack-dev gifsicle
```

Create a `virtualenv` for this repo:
```bash
python -m virtualenv venv
source venv/bin/activate
```

Install the various python packages in this repo:
```bash
python -m pip install -e crown_ibp 
python -m pip install -e auto_LiRPA
python -m pip install -e robust_sdp
python -m pip install -e nn_partition
python -m pip install -e nn_closed_loop
```

You're good to go!

### Simple Examples

Try running a simple example where the Analyzer computes bounds on the NN output (given bounds on the NN input):
```bash
python -m nn_partition.example \
	--partitioner GreedySimGuided \
	--propagator CROWN_LIRPA \
	--term_type time_budget \
	--term_val 2 \
	--interior_condition lower_bnds \
	--model random_weights \
	--activation relu \
	--show_input --show_output --show_plot
```

Or, compute reachable sets for a closed-loop system with a pre-trained NN control policy:
```bash
python -m nn_closed_loop.example \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator \
	--state_feedback \
	--t_max 5 \
	--show_plot
```

Or, compute backward reachable sets for a closed-loop system with a pre-trained NN control policy:
```bash
python -m nn_closed_loop.example_backward \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator \
	--state_feedback \
	--show_plot --boundaries polytope
```

### Jupyter Notebooks

Please see the `jupyter_notebooks` folder for an interactive version of the above examples.

### Replicate plots from the papers:

* LCSS/ACC '21: [README](/docs/_static/lcss21/lcss21.md)
* ICRA '21: [README](/docs/_static/icra21/icra21.md)
* Journal: [README](/docs/_static/journal/journal.md)

### If you find this code useful, please consider citing:
For the partitioning-only code (LCSS/ACC '21):
```
@article{everett2020robustness,
  title={Robustness Analysis of Neural Networks via Efficient Partitioning with Applications in Control Systems},
  author={Everett, Michael and Habibi, Golnaz and How, Jonathan P},
  journal={IEEE Control Systems Letters},
  year={2021},
  publisher={IEEE},
  doi={10.1109/LCSYS.2020.3045323}
}
```

For the closed-loop system analysis code (ICRA '21):
```
@inproceedings{Everett21_ICRA,
    Author = {Michael Everett and Golnaz Habibi and Jonathan P. How},
    Booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    Title = {Efficient Reachability Analysis for Closed-Loop Systems with Neural Network Controllers},
    Year = {2021},
    Url = {https://arxiv.org/pdf/2101.01815.pdf},
    }
```
and/or:
```
@article{Everett21_journal,
    Author = {Michael Everett and Golnaz Habibi and Chuangchuang Sun and Jonathan P. How},
    Title = {Reachability Analysis of Neural Feedback Loops},
    Year = {2021},
    Url = {https://arxiv.org/pdf/2101.01815.pdf},
    }
```

### TODOS:

- [x] ICRA Fig 3 as single script
- [x] ICRA Fig 3b make pkl
- [x] ICRA Fig 3c from pkl
- [x] get animation working for ICRA

Someday soon...
- [ ] add rtdocs (auto-fill code snippets from test files)
- [ ] LCSS Fig 8
- [ ] Replicate LCSS Table 6b
- [ ] Replicate LCSS Table I
- [ ] ICRA Fig 4a make pkl
- [ ] ICRA Fig 4a from pkl
- [ ] ICRA Fig 4b as single script
- [ ] ICRA Fig 4b load correct model