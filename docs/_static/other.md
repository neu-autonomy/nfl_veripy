## Other Stuff

### OVERT.jl

- Install julia (v1.5) via https://julialang.org/downloads

To use OVERT, first start a julia session, then start the HTTP server that hosts an OVERT query endpoint:
```bash
julia> include("/home/mfe/code/nn_robustness_analysis/run_overt.jl")
```
There won't be any output but you should now have a server at `http://127.0.0.1:8000/` so you can send a `POST` request to call OVERT.
Note that the reason for having an HTTP server is that compiling julia libraries can take a while and you don't want to wait for this every time you run the python script.

Now you can run OVERT as a propagator, just like CROWN or SDP:
```bash
python -m nn_closed_loop.example \
    --partitioner None \
    --propagator OVERT \
    --system double_integrator \
    --state_feedback \
    --t_max 5 \
    --show_plot
```




## Older stuff

### getting julia code to work

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