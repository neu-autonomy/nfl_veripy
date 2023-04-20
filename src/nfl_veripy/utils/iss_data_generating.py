import nfl_veripy.constraints as constraints
import numpy as np
from nfl_veripy.dynamics import ISS

dynamics = ISS()
init_state_range = 1 * np.ones((dynamics.n, 2))
init_state_range[:, 0] = init_state_range[:, 0] - 0.5
init_state_range[:, 1] = init_state_range[:, 1] + 0.5

iss_xs, iss_us = dynamics.collect_data(
    t_max=0.5,
    input_constraint=constraints.LpInputConstraint(
        p=np.inf, range=init_state_range
    ),
    num_samples=20,
)
print(iss_xs.shape, iss_us.shape)
np.savetxt("iss_xs.csv", iss_xs, delimiter=",")
np.savetxt("iss_us.csv", iss_us, delimiter=",")
# with open(dir_path + "/datasets/double_integrator/xs.pkl", "wb") as f:
#     pickle.dump(xs, f)
# with open(dir_path + "/datasets/double_integrator/us.pkl", "wb") as f:
#     pickle.dump(us, f)
