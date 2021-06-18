## Newer stuff

```bash
python -m nn_closed_loop.example_backward \
	--partitioner None \
	--propagator CROWN \
	--system double_integrator \
	--state_feedback \
	--t_max 0.9 \
	--show_plot --boundaries polytope
```


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

### Examples

See the implementation of Xiang 2017 and Xiang 2020:
```bash
python -m partition.xiang
```

This will pop up with:
- randomly initialized DNN & 25 uniform partitioned inputs (from Xiang 2017)
- robot arm example using CROWN & IBP and Uniform & Simulation-Guided partitioning (from Xiang 2020)

### Figure 6
To reproduce this figure, please run:
```bash
python -m partition.experiments
```
This will iterate through combinations of `Partitioner`, `Propagator` and termination conditions and produce a timestamped `.pkl` file in `results` containing a `pandas` dataframe with the results.

To plot the data from this dataframe (it should happen automatically by running `experiments.py`), comment out `df = experiment()` in `experiments.py` and it will just load the latest dataframe and plot it.
You can switch which x axis to use with the argument of `plot()`.