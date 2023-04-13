## Other Stuff

### OVERT.jl

- Install julia (v1.5) via https://julialang.org/downloads

To use OVERT, first start a julia session, then start the HTTP server that hosts an OVERT query endpoint:
```bash
julia> include("<path_to_repo>/nn_closed_loop/nn_closed_loop/utils/run_overt.jl")
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

### Run tests locally

```bash
docker build -t nnrob -f docker/Dockerfile .
docker run -v $PWD:/home/nn_robustness_analysis nnrob:latest python -m nn_closed_loop.tests.test
```

### Run new closed-loop partitioner example

```bash
python -m nn_closed_loop.example_backward \
    --partitioner None \
    --propagator CROWN \
    --system double_integrator \
    --state_feedback \
    --t_max 2 \
    --show_plot --overapprox --skip_save_plot
```

## mypy debugging

Only the first time, get the type stubs for some installed pip pkgs using the typeshed (most that don't live in here are ignored in `mypy.ini`): 
```bash
mypy --install-types
```

Run this to check types of entire nn_closed_loop pkg (excluding a few files as defined in `mypy.ini`):
```bash
cd nn_closed_loop
mypy -p nn_closed_loop
```


