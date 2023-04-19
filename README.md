## Updates

- **2023-04-13:** Add new jax-based propagators, including some from [`DRIP` paper](https://arxiv.org/abs/2212.04646). Cleaned up implementation of BReach-LP and HyBReach-LP from OJCSYS paper.
- **2022-06-20:** Add new backprojection code from [`BReach-LP` paper](https://arxiv.org/abs/2204.08319). More info [here](/docs/_static/cdc22/cdc22.md)
- **2022-05-09:** Add new N-Step `ClosedLoopPropagator`. Rather than recursively computing reachable sets (suffers from the wrapping effect), we see improved performance by solving an LP directly for the reachable set N steps in the future. You can experiment with this using the `CROWNNStep` flag in `nn_closed_loop/example.py`.
- **2022-05-09:** Add new MILP-based `ClosedLoopPropagator`, using [`OVERT`](https://github.com/sisl/OVERTVerify.jl). Note that this component requires a Julia installation, and we pass data between Python and Julia using a lightweight local HTTP server. More info [here](/docs/_static/other.md).

## About

### `nn_partition`: Open-Loop Analysis (NNs in Isolation)

**Handles problems such as:**
- Given a set of possible NN inputs and a trained NN, compute outer bounds on the set of possible NN outputs.

For more info, please see [this README](/docs/_static/access21/access21.md).

### `nn_closed_loop`: Closed-Loop Analysis (NNs in feedback loops) -- includes Reach-LP and BReach-LP

**Handles problems such as:**
- Given a set of possible initial states, a trained NN controller, and a known dynamics model, compute outer bounds on the set of possible future states (**forward reachable sets**).
- Given a set of terminal states, a trained NN controller, and a known dynamics model, compute inner/outer bounds on the set of possible initial states that will/won't lead to the terminal state set (**backprojection sets**).

For more info, please see [this README](/docs/_static/access21/access21.md) and [this README](/docs/_static/cdc22/cdc22.md).

## Setup

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
* IEEE Access '21: [README](/docs/_static/access21/access21.md)
* CDC '22: [README](/docs/_static/cdc22/cdc22.md)
* ACC '23 (to appear): Coming soon!
* OJCSYS '23 (to appear): Coming soon!
* LCSS '23 (to appear): Coming soon!

### If you find this code useful, please consider citing our work:

* Everett, M. (2021, December). Neural network verification in control. In 2021 60th IEEE Conference on Decision and Control (CDC) (pp. 6326-6340). IEEE.
* Everett, M., Habibi, G., & How, J. P. (2020). Robustness analysis of neural networks via efficient partitioning with applications in control systems. IEEE Control Systems Letters, 5(6), 2114-2119.
* Everett, M., Habibi, G., Sun, C., & How, J. P. (2021). Reachability analysis of neural feedback loops. IEEE Access, 9, 163938-163953.
* Everett, M., Habibi, G., & How, J. P. (2021, May). Efficient reachability analysis of closed-loop systems with neural network controllers. In 2021 IEEE International Conference on Robotics and Automation (ICRA) (pp. 4384-4390). IEEE.
* Rober, N., Everett, M., & How, J. P. (2022, December). Backward reachability analysis for neural feedback loops. In 2022 IEEE 61st Conference on Decision and Control (CDC) (pp. 2897-2904). IEEE.
* Rober, N., Everett, M., Zhang, S., & How, J. P. (2022). A Hybrid Partitioning Strategy for Backward Reachability of Neural Feedback Loops. arXiv preprint arXiv:2210.07918.
* Rober, N., Katz, S. M., Sidrane, C., Yel, E., Everett, M., Kochenderfer, M. J., & How, J. P. (2023). Backward reachability analysis of neural feedback loops: Techniques for linear and nonlinear systems. IEEE Open Journal of Control Systems.
* Everett, M., Bunel, R., & Omidshafiei, S. (2023). Drip: Domain refinement iteration with polytopes for backward reachability analysis of neural feedback loops. IEEE Control Systems Letters.


## Acknowledgements

This research is supported by Ford Motor Company.

We build on excellent open-source repositories from the neural network analysis community. These repositories are imported as Git submodules or re-implemented in Python here, with some changes to reflect the slightly different problem statements:
* [`auto_LIRPA`](https://github.com/KaidiXu/auto_LiRPA)
* [`crown_ibp`](https://github.com/huanzhang12/CROWN-IBP)
* [`robust_nn`](https://github.com/arobey1/RobustNN)
* [`nnv`](https://github.com/verivital/nnv)
* [`jax_verify`](https://github.com/deepmind/jax_verify)


## TODOS:

- [ ] add rtdocs (auto-fill code snippets from test files)
- [ ] add installation instructions & tests for julia code
