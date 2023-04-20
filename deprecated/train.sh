#!/bin/bash

# Load python virtualenv
virtualenv --python=/usr/bin/python3.8 venv
source venv/bin/activate
pip3.8 install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:/home/nrober/code/nn_robustness_analysis/nfl_veripy

# run ...!
python3.8 -m nfl_veripy.example_backward \
--propagator CROWN \
--partitioner None \
--state_feedback \
--show_plot \
--overapprox \
--system double_integrator \
--t_max 5 \
--final_state_range [[4.5,5.0],[-0.25,0.25]] \
--plot_lims [[-3.8,5.64],[-0.64,2.5]] \
--num_partitions 9 \
--show_convex_hulls \
--show_BReach \
--estimate_runtime