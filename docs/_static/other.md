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
