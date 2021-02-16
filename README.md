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

### Replicate plots from the papers:

* LCSS/ACC '21: [README](docs/_static/lcss21/lcss21.md)
* ICRA '21 (submission): [README](docs/_static/icra21/icra21.md)


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

---

### Figure 6
To reproduce this figure, please run:
```bash
python -m partition.experiments
```
This will iterate through combinations of `Partitioner`, `Propagator` and termination conditions and produce a timestamped `.pkl` file in `results` containing a `pandas` dataframe with the results.

To plot the data from this dataframe (it should happen automatically by running `experiments.py`), comment out `df = experiment()` in `experiments.py` and it will just load the latest dataframe and plot it.
You can switch which x axis to use with the argument of `plot()`.

---

## Older stuff

### getting julia code to work

- Install julia via https://julialang.org/downloads
- Be sure to create a pointer to the executable (https://julialang.org/downloads/platform/)
```bash
# For OSX:
ln -s /Applications/Julia-<version>.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
```
- Install NeuralVerification.jl from Stanford's github
```bash
julia
# Press `]`
add https://github.com/sisl/NeuralVerification.jl
```
- Install pyjulia
```bash
python -m pip install julia
```
