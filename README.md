
### Get the code

```bash
git clone --recursive git@gitlab.com:mit-acl/ford_ugvs/robustness_analysis.git
```

### Install

In the root of this directory:
```bash
python -m pip install -e .
python -m pip install -e crown_ibp
python -m pip install -e partition
python -m pip install -e robust_sdp
```

Now you can import things like:
```bash
>>> from reach_lp.reach_lp import reachLP_1
>>> import crown_ibp.bound_layers
```

### Examples

See the implementation of Xiang 2017 and Xiang 2020:
```bash
python -m partition.xiang
```

This will pop up with:
- their randomly initialized DNN & 25 uniform partitioned inputs (from Xiang 2017)
- their robot arm example using CROWN & IBP and Uniform & Simulation-Guided partitioning (from Xiang 2020)


### tmp
```bash
source ~/code/gym-collision-avoidance/venv/bin/activate
```
