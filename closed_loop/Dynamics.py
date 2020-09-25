import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches as ptch
from tqdm import tqdm
# from reach_lp.nn import control_nn
import pypoman
from closed_loop.mpc import control_mpc
from scipy.linalg import solve_discrete_are

from closed_loop.nn_bounds import BoundClosedLoopController
from closed_loop.ClosedLoopConstraints import PolytopeInputConstraint, LpInputConstraint, PolytopeOutputConstraint, LpOutputConstraint
import torch

class Dynamics:
    def __init__(self, At, bt, ct, u_min=None, u_max=None, dt=1.0):
        self.At = At
        self.bt = bt
        self.ct = ct
        self.num_states, self.num_inputs = bt.shape

        # Min/max control inputs
        self.u_min = u_min
        self.u_max = u_max

        self.dt = dt

    def colors(self, t_max):
        return [cm.get_cmap("tab10")(i) for i in range(t_max+1)]

    def show_samples(self, t_max, input_constraint, save_plot=False, ax=None, show=False, controller='mpc'):
        if ax is None:
            ax = plt.figure()

        xs, us = self.collect_data(t_max, input_constraint, num_samples=1000, controller=controller)

        num_runs, num_timesteps, num_states = xs.shape
        colors = self.colors(num_timesteps)
        for t in range(num_timesteps):
            ax.scatter(xs[:,t,0], xs[:,t,1], color=colors[t])

        # if isinstance(input_constraint, PolytopeInputConstraint):
        
        # elif isinstance(input_constraint, LpInputConstraint):
        #     if input_constraint.p == np.inf:
        #         # Input state rectangle
        #         rect = ptch.Rectangle(init_state_range[:,0],
        #             init_state_range[0,1]-init_state_range[0,0], 
        #             init_state_range[1,1]-init_state_range[1,0],
        #             fill=False, ec='k')
        #         ax.add_patch(rect)
        #     else:
        #         raise NotImplementedError

        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')

        if save_plot:
            ax.savefig(plot_name)

        if show:
            plt.show()

    def collect_data(self, t_max, input_constraint, num_samples=2420, controller='mpc'):
        xs, us = self.run(t_max, input_constraint, num_samples, collect_data=True, controller=controller)
        return xs, us

    def run(self, t_max, input_constraint, num_samples=100, collect_data=False, clip_control=True, controller='mpc'):
        np.random.seed(0)
        num_timesteps = int((t_max)/self.dt)+1

        if collect_data:
            np.random.seed(1)
            num_runs = int(num_samples / num_timesteps)
            xs = np.zeros((num_runs, num_timesteps, self.num_states))
            us = np.zeros((num_runs, num_timesteps, self.num_inputs))

        # Initial state
        if isinstance(input_constraint, LpInputConstraint):
            if input_constraint.p == np.inf:
                xs[:,0,:] = np.random.uniform(
                    low=input_constraint.range[:,0], 
                    high=input_constraint.range[:,1],
                    size=(num_runs, self.num_states))
            else:
                raise NotImplementedError
        elif isinstance(input_constraint, PolytopeInputConstraint):
            init_state_range = input_constraint.to_linf()
            xs[:,0,:] = np.random.uniform(low=init_state_range[:,0], high=init_state_range[:,1], size=(num_runs, self.num_states))
            within_constraint_inds = np.where(np.all((np.dot(input_constraint.A, xs[:,0,:].T) - np.expand_dims(input_constraint.b, axis=-1)) <= 0, axis=0))
            xs = xs[within_constraint_inds]
            us = us[within_constraint_inds]
        else:
            raise NotImplementedError

        this_colors = self.colors(t_max).copy()

        t = 0
        step = 0
        while t < t_max:
            t += self.dt
            if controller == 'mpc':
                u = self.control_mpc(x0=x[step,:])
            elif isinstance(controller, BoundClosedLoopController):
                u = self.control_nn(x=xs[:,step,:], model=controller)
            else:
                raise NotImplementedError
            if clip_control and (self.u_min is not None or self.u_max is not None):
                u_raw = u
                u = np.clip(u, self.u_min, self.u_max)
            xs[:,step+1,:] = (np.dot(self.At, xs[:, step, :].T) + np.dot(self.bt, u.T)).T
            us[:,step,:] = u
            step += 1
        if collect_data:
            return xs, us

        # for run in range(num_runs):
        #     # Initial state
        #     x = np.zeros((num_timesteps, self.num_states))
        #     x[0,:] = np.random.uniform(
        #         low=init_state_range[:,0], 
        #         high=init_state_range[:,1])
        #     u_clipped = np.zeros((num_timesteps, self.num_inputs))
        #     this_colors = self.colors(t_max).copy()

        #     t = 0
        #     step = 0
        #     while t < t_max:
        #         t += self.dt
        #         if controller == 'mpc':
        #             u = self.control_mpc(x0=x[step,:])
        #         elif isinstance(controller, BoundClosedLoopController):
        #             u = self.control_nn(x=x[step,:], model=controller)
        #         else:
        #             raise NotImplementedError
        #         if clip_control and (self.u_min is not None or self.u_max is not None):
        #             u_raw = u
        #             u = np.clip(u, self.u_min, self.u_max)
        #             # if u != u_raw:
        #             #     this_colors[step+1] = [0,0,0]
        #         if collect_data:
        #             xs[run, step, :] = x[step,:]
        #             us[run, step, :] = u
        #         x[step+1,:] = np.dot(self.At, x[step, :]) + np.dot(self.bt, u)
        #         step += 1
        # if collect_data:
        #     return xs, us


class DoubleIntegrator(Dynamics):
    def __init__(self):

        At = np.array([[1, 1],[0, 1]])
        bt = np.array([[0.5], [1]])
        ct = np.array([0., 0.]).T

        u_min = -100.
        u_max = 100.

        Dynamics.__init__(self, At=At, bt=bt, ct=ct, u_min=u_min, u_max=u_max)

        # LQR-MPC parameters
        self.Q = np.eye(2)
        self.R = 1
        self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def control_nn(self, x, model):
        if x.ndim == 1:
            batch_x = np.expand_dims(x, axis=0)
        else:
            batch_x = x
        us = model.forward(torch.Tensor(batch_x)).data.numpy()
        return us

    def control_mpc(self, x0):
        return control_mpc(x0, self.At, self.bt, self.ct, self.Q, self.R, self.Pinf, self.u_min, self.u_max, n_mpc=10, debug=False)


if __name__ == '__main__':
    dynamics = DoubleIntegrator()
    t_max = 3
    init_state_range = np.array([ # (num_inputs, 2)
                      [2.5, 3.0], # x0min, x0max
                      [-0.25, 0.25], # x1min, x1max
    ])
    dynamics.show_samples(t_max, init_state_range, save_plot=False, ax=None, show=False, controller='nn')

