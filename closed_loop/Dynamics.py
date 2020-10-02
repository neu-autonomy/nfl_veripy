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
    def __init__(self, At, bt, ct, u_limits=None, dt=1.0):
        self.At = At
        self.bt = bt
        self.ct = ct
        self.num_states, self.num_inputs = bt.shape

        # Min/max control inputs
        self.u_limits = u_limits

        self.dt = dt

    def control_nn(self, x, model):
        if x.ndim == 1:
            batch_x = np.expand_dims(x, axis=0)
        else:
            batch_x = x
        us = model.forward(torch.Tensor(batch_x)).data.numpy()
        return us

    def colors(self, t_max):
        return [cm.get_cmap("tab10")(i) for i in range(t_max+1)]
 
    def get_sampled_output_range(self, input_constraint, t_max =5, num_samples= 1000, controller='mpc'):

        xs, us = self.collect_data(t_max, input_constraint, num_samples, controller=controller)
         
        num_runs, num_timesteps, num_states = xs.shape

        if isinstance(input_constraint, PolytopeInputConstraint):
            raise NotImplementedError
        elif isinstance(input_constraint, LpInputConstraint):
            sampled_range= np.zeros((num_timesteps-1,num_states,2))
            for t in range(1,num_timesteps):
                sampled_range[t-1,:,0] = np.min(xs[:,t,:], axis =0)
                sampled_range[t-1,:,1] = np.max(xs[:,t,:], axis =0)
        else:
            raise NotImplementedError
     
        return sampled_range
    
    def show_samples(self, t_max, input_constraint, save_plot=False, ax=None, show=False, controller='mpc', input_dims=[[0],[1]]):
        if ax is None:
            ax = plt.subplot()

        xs, us = self.collect_data(t_max, input_constraint, num_samples=1000, controller=controller)

        num_runs, num_timesteps, num_states = xs.shape
        colors = self.colors(num_timesteps)

        for t in range(num_timesteps):
            ax.scatter(xs[:,t,input_dims[0]], xs[:,t,input_dims[1]], color=colors[t])

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

        ax.set_xlabel('$x_'+str(input_dims[0][0])+'$')
        ax.set_ylabel('$x_'+str(input_dims[1][0])+'$')

        if save_plot:
            ax.savefig(plot_name)

        if show:
            plt.show()

    def collect_data(self, t_max, input_constraint, num_samples=2420, controller='mpc'):
        xs, us = self.run(t_max, input_constraint, num_samples, collect_data=True, controller=controller)
        return xs, us

    def run(self, t_max, input_constraint, num_samples=100, collect_data=False, clip_control=False, controller='mpc'):
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
                # import pdb; pdb.set_trace()
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

        t = 0
        step = 0
        while t < t_max:
            t += self.dt
            if controller == 'mpc':
                u = self.control_mpc(x0=x[step,:])
            elif isinstance(controller, BoundClosedLoopController) or isinstance(controller, torch.nn.Sequential):
                u = self.control_nn(x=xs[:,step,:], model=controller)
            else:
                raise NotImplementedError
            if clip_control and (self.u_limits is not None):
                u = np.clip(u, self.u_limits[:,0], self.u_limits[:,1])

            xs[:,step+1,:] = self.dynamics_step(xs[:, step, :], u)

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

        self.continuous_time = False

        At = np.array([[1, 1],[0, 1]])
        bt = np.array([[0.5], [1]])
        ct = np.array([0., 0.]).T

        u_limits = np.array([
            [-100., 100.]
        ])

        Dynamics.__init__(self, At=At, bt=bt, ct=ct, u_limits=u_limits)

        # LQR-MPC parameters
        self.Q = np.eye(2)
        self.R = 1
        self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def control_mpc(self, x0):
        return control_mpc(x0, self.At, self.bt, self.ct, self.Q, self.R, self.Pinf, self.u_min, self.u_max, n_mpc=10, debug=False)

    def dynamics_step(self, xs, us):
        # Dynamics are already discretized:
        return (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct

class Quadrotor(Dynamics):
    def __init__(self):

        self.continuous_time = True

        g = 9.8 # m/s^2

        At = np.zeros((6,6))
        At[0][3] = 1
        At[1][4] = 1
        At[2][5] = 1

        bt = np.zeros((6,3))
        bt[3][0] = g
        bt[4][1] = -g
        bt[5][2] = 1

        ct = np.zeros((6,))
        ct[-1] = -g
        # ct = np.array([0., 0., 0. ,0., 0., -g]).T

        u_limits = np.array([
            [-np.pi/9, np.pi/9],
            [-np.pi/9, np.pi/9],
            [0, 2*g],
            ])

        dt = 0.1

        Dynamics.__init__(self, At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt)

        # # LQR-MPC parameters
        # self.Q = np.eye(2)
        # self.R = 1
        # self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def dynamics_step(self, xs, us):

        xs_t1 = xs + self.dt*self.dynamics(xs, us)

        # h = self.dt
        # # Dynamics are for x_dot = Ax+Bu+c, need to compute x_{t+1}:
        # k1 = h*self.dynamics(xs, us)
        # k2 = h*self.dynamics(xs + k1/2, us)
        # k3 = h*self.dynamics(xs + k2/2, us)
        # k4 = h*self.dynamics(xs + k3, us)
        # # k1 = h*self.dynamics(xs, us)
        # # k2 = h*self.dynamics(xs+h/2, us + k1/2)
        # # k3 = h*self.dynamics(xs+h/2, us + k2/2)
        # # k4 = h*self.dynamics(xs+h, us + k3)
        # xs_t1 = xs + k1/6 + k2/3 + k3/3 + k4/6

        # print("xs_t1:", xs_t1)
        # print("xs_t1_:", xs_t1_)
        # print('--')

        return xs_t1

    def dynamics(self, xs, us):
        # # TODO: Add ct back in!!!!!!!
        # return ((np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T)
        return ((np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct)

if __name__ == '__main__':
    from closed_loop.nn import load_model

    # dynamics = DoubleIntegrator()
    # init_state_range = np.array([ # (num_inputs, 2)
    #                   [2.5, 3.0], # x0min, x0max
    #                   [-0.25, 0.25], # x1min, x1max
    # ])
    dynamics = Quadrotor()

    # init_state_range = np.array([
    #     [ 4.74399948,  4.84599972],
    #     [ 4.64899969,  4.75099993],
    #     [ 2.94900012,  3.05099988],
    #     [-0.16668373, -0.11418372],
    #     [-0.42657009, -0.37431771],
    #     [-0.07524291, -0.04822937],
    #     ])

    init_state_range = np.array([
        [ 4.7447899, 4.65043755, 2.94919611, 0.9173471, -0.03881739, -0.003111],
        [ 4.8456064, 4.75059748, 3.04982631, 0.9377732, -0.01473783, 0.0167838],
        ]).T

    init_state_range = np.array([ # (num_inputs, 2)
                  [4.65,4.65,2.95,0.94,-0.01,-0.01], # x0min, x0max
                  [4.75,4.75,3.05,0.96,0.01,0.01] # x1min, x1max
    ]).T
    goal_state_range = np.array([
                          [3.7,2.5,1.2],
                          [4.1,3.5,2.6]            
    ]).T
    controller = load_model(name='quadrotor')
    t_max = 3*dynamics.dt
    input_constraint = LpInputConstraint(range=init_state_range, p=np.inf)
    dynamics.show_samples(t_max, input_constraint, save_plot=False, ax=None, show=True, controller=controller)

