### About

This repository provides Analyzer, Propagator, and Partitioner classes from the LCSS/ACC '21 paper.
We import `auto_LIRPA`, `crown_ibp` and `robust_sdp` codebases from open source repositories (as Git submodules) and implement `Partitioner` methods from Xiang '17 and Xiang '20 in Python (open source implementation is in Matlab), along with our own methods from the paper.

### Get the code

```bash
git clone --recursive <this_repo>
```

### Install

Create a `virtualenv` for this repo:
```bash
python -m virtualenv venv
source venv/bin/activate
```

Install the various python packages in this repo:
```bash
python -m pip install -e crown_ibp auto_LIRPA robust_sdp partition closed_loop
```

### Simple Examples

You're good to go! Try running a simple example where the Analyzer computes bounds on the NN output (given bounds on the NN input):
```bash
python -m partition.example \
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
python -m closed_loop.example \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator_mpc \
	--state_feedback \
	--t_max 5 \
	--show_plot
```

### Replicate plots from the papers:

* LCSS/ACC '21: [README](docs/_static/lcss21/lcss21.md)
* ICRA '21 (submission): [README](docs/_static/icra21/icra21.md)

### If you find this code useful, please consider citing:
For the partitioning-only code (LCSS/ACC '21):
```
@article{everett2020robustness,
  title={Robustness Analysis of Neural Networks via Efficient Partitioning with Applications in Control Systems},
  author={Everett, Michael and Habibi, Golnaz and How, Jonathan P},
  journal={IEEE Control Systems Letters},
  year={2020},
  publisher={IEEE},
  doi={10.1109/LCSYS.2020.3045323}
}
```

For the closed-loop system analysis code (ICRA '21 subm.):
```
@inproceedings{Everett21_ICRA,
    Author = {Michael Everett and Golnaz Habibi and Jonathan P. How},
    Booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    Title = {Efficient Reachability Analysis for Closed-Loop Systems with Neural Network Controllers},
    Year = {2021},
    Pages = {(in review)},
    Url = {https://arxiv.org/pdf/2101.01815.pdf},
    }
```

### TODOS:

- [x] Choices in analyzer argparse
- [x] move partitioners, propagators to separate dirs
- [x] move cartpole, pend, quadrotor files elsewhere
- [x] move MNIST data to right place
- [x] merge in closed_loop branch
- [x] Fig 3b individual images
- [x] Fig 3a individual table
- [x] Replicate LCSS Fig 4
- [x] Replicate LCSS Fig 5
- [x] Replicate LCSS Fig 6
- [x] Replicate ICRA Fig 3b individuals + table
- [x] Replicate ICRA Fig 4b individuals
- [x] Replicate ICRA Fig 5
- [ ] publish crown_ibp, auto-Lirpa forks
- [ ] setup ci and simple tests to run the various expts
- [ ] add citation to papers, add description of repo to top of readme
- [ ] add license & copyright?
- [ ] setup sync with github

Someday soon...
- [ ] add rtdocs
- [ ] get animation working
- [ ] LCSS Fig 8
- [ ] Replicate LCSS Table 6b
- [ ] Replicate LCSS Table I
- [ ] ICRA Fig 3 as single script
- [ ] ICRA Fig 3c make pkl
- [ ] ICRA Fig 3c from pkl
- [ ] ICRA Fig 4a make pkl
- [ ] ICRA Fig 4a from pkl
- [ ] ICRA Fig 4b as single script
- [ ] ICRA Fig 4b load correct model
- [ ] ICRA Fig 5 axes names & spacings

