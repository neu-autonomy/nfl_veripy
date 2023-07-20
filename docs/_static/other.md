## Other Stuff


### [deprecated] Jupyter Notebooks

Please see the `jupyter_notebooks` folder for an interactive version of the above examples.

## packaging info

```bash
python -m pip install build twine

# update version number in pyproject.toml (e.g., 0.0.2a0 to test out a candidate 0.0.2 release)
cd nfl_veripy
python -m build
python -m twine upload --repository testpypi dist/nfl_veripy-0.0.2a0*

# in a separate terminal...
python -m virtualenv nfl_veripy_pypi_venv
source nfl_veripy_pypi_venv/bin/activate
python -m pip install "jax_verify @ git+https://gitlab.com/neu-autonomy/certifiable-learning/jax_verify.git" "crown_ibp @ git+https://gitlab.com/neu-autonomy/certifiable-learning/crown_ibp.git"
python -m pip install --extra-index-url https://test.pypi.org/simple/ nfl-veripy==0.0.2a0
python -m nfl_veripy.example --config example_configs/icra21/fig3_reach_lp.yaml

# if that works, build & release the pkg to the real pypi:
# - update version number in pyproject.toml (e.g., 0.0.2)
python -m build
python -m twine upload dist/nfl_veripy-0.0.2*

```

Aside: We acknowledge that the method for installing `jax_verify` and `crown_ibp` (dependencies of `nfl_veripy`) in the non-developer setup way is a little unconventional.
It would be better to simply include these as dependencies of `nfl_veripy` and let pip find those packages, but (a) those packages are not available (or are too outdated) on PyPI, and (b) it is not allowed to include dependencies with direct URLs when releasing a package on PyPI.
If there's a better way of doing this we would love to hear about it!


### OVERT.jl

- Install julia (v1.5) via https://julialang.org/downloads

To use OVERT, first start a julia session, then start the HTTP server that hosts an OVERT query endpoint:
```bash
julia> include("<path_to_repo>/nfl_veripy/nfl_veripy/utils/run_overt.jl")
```
There won't be any output but you should now have a server at `http://127.0.0.1:8000/` so you can send a `POST` request to call OVERT.
Note that the reason for having an HTTP server is that compiling julia libraries can take a while and you don't want to wait for this every time you run the python script.

Now you can run OVERT as a propagator, just like CROWN or SDP:
```bash
python -m nfl_veripy.example \
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
docker run -v $PWD:/home/nn_robustness_analysis nnrob:latest python -m nfl_veripy.tests.test
```

### Run new closed-loop partitioner example

```bash
python -m nfl_veripy.example_backward \
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

Run this to check types of entire nfl_veripy pkg (excluding a few files as defined in `mypy.ini`):
```bash
cd nfl_veripy
mypy -p nfl_veripy
```


