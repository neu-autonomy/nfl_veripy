import numpy as np
from nfl_veripy.dynamics import Duffing

duff = Duffing()
dim = 2
N = 100
K = 80
duff_xs = np.zeros((N * K, dim), dtype=float)
duff_us = np.zeros((N * K, 1), dtype=float)
bound = np.array([[2.49, 2.51], [1.49, 1.51]])
bound = np.array([[2.45, 2.55], [1.45, 1.55]])
for i in range(N):
    duff_xs_p = np.zeros((K, dim), dtype=float)
    duff_us_p = np.zeros((K, 1), dtype=float)
    duff_xs_p[0, :] = np.random.uniform(
        low=bound[:, 0], high=bound[:, 1], size=(1, dim)
    )
    for k in range(K):
        if k >= 0 and k <= 10:
            duff_us_p[k] = 0.5 * k
        elif k <= 40:
            duff_us_p[k] = 5 - 0.5 * (k - 10) / 3
        else:
            duff_us_p[k] = 0
        if k <= K - 2:
            duff_xs_p[k + 1, :] = duff.dynamics_step(
                duff_xs_p[k, :], duff_us_p[k, :]
            )

    duff_xs[i * K : (i + 1) * K, :] = duff_xs_p
    duff_us[i * K : (i + 1) * K, :] = duff_us_p

print(duff_xs.shape)
print(duff_us.shape)
np.savetxt("duff_xs.csv", duff_xs, delimiter=",")
np.savetxt("duff_us.csv", duff_us, delimiter=",")


# xs = np.reshape(duff_xs, (-1, K, dim))
# us = np.reshape(duff_us, (-1, K, 1))
# import matplotlib.pyplot as plt
# for i in range(N):
#     plt.plot(xs[i, 1:, 0], xs[i, 1:, 1])
# plt.show()

# return xs, us
