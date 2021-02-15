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
import os
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

class Dynamics:
    def __init__(self, At, bt, ct, u_limits=None, dt=1.0, c=None, sensor_noise=None, process_noise=None):

        # State dynamics
        self.At = At
        self.bt = bt
        self.ct = ct
        self.num_states, self.num_inputs = bt.shape

        # Observation Dynamics and Noise
        if c is None:
            c = np.eye(self.num_states)
        self.c = c
        self.num_outputs = self.c.shape[0]
        self.sensor_noise = sensor_noise
        self.process_noise = process_noise

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

    def observe_step(self, xs):
        obs = np.dot(xs, self.c.T)
        if self.sensor_noise is not None:
            noise = np.random.uniform(
                low=self.sensor_noise[:,0],
                high=self.sensor_noise[:,1],
                size=xs.shape,
            )
            obs += noise
        return obs

    def dynamics_step(self, xs, us):
        raise NotImplementedError

    def colors(self, t_max):
        return [cm.get_cmap("tab20")(i) for i in range(t_max+1)]
 
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
            ax.scatter(xs[:,t,input_dims[0]], xs[:,t,input_dims[1]], color=colors[t],s =4)

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

    def run(self, t_max, input_constraint, num_samples=100, collect_data=False, clip_control=True, controller='mpc'):
        np.random.seed(0)
        num_timesteps = int((t_max+self.dt + np.finfo(float).eps)/(self.dt))
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

        t = 0
        step = 0
        while t < t_max:
            print('--- t={} ---'.format(t))

            # Observe system (using observer matrix, possibly adding measurement noise)
            obs = self.observe_step(xs[:,step,:])

            # Compute Control
            if controller == 'mpc':
                u = self.control_mpc(x0=obs)
            elif isinstance(controller, BoundClosedLoopController) or isinstance(controller, torch.nn.Sequential):
                u = self.control_nn(x=obs, model=controller)
            else:
                raise NotImplementedError
            if clip_control and (self.u_limits is not None):
                u = np.clip(u, self.u_limits[:,0], self.u_limits[:,1])

            # Step through dynamics (possibly adding process noise)
            xs[:,step+1,:] = self.dynamics_step(xs[:, step, :], u)

            us[:,step,:] = u
            step += 1
            t += self.dt +np.finfo(float).eps

        if collect_data:
            return xs.reshape(-1, self.num_states), us.reshape(-1, self.num_inputs)

class DoubleIntegrator(Dynamics):
    def __init__(self):

        self.continuous_time = False

        At = np.array([[1, 1],[0, 1]])
        bt = np.array([[0.5], [1]])
        ct = np.array([0., 0.]).T

        # u_limits = None
        u_limits = np.array([
            [-1., 1.] # (u0_min, u0_max)
        ])

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits)

        # LQR-MPC parameters
        self.Q = np.eye(2)
        self.R = 1
        self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def control_mpc(self, x0):
        return control_mpc(x0, self.At, self.bt, self.ct, self.Q, self.R, self.Pinf, self.u_limits[:,0], self.u_limits[:,1], n_mpc=10, debug=False)

    def dynamics_step(self, xs, us):
        # Dynamics are already discretized:
        xs_t1 = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:,0],
                high=self.process_noise[:,1],
                size=xs.shape,
            )
            xs_t1 += noise
        return xs_t1

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

        # u_limits = None
        u_limits = np.array([
            [-np.pi/9, np.pi/9],
            [-np.pi/9, np.pi/9],
            [0, 2*g],
        ])

        dt = 0.1

        super().__init__(At=At, bt=bt, ct=ct, u_limits=u_limits, dt=dt)

        # # LQR-MPC parameters
        # self.Q = np.eye(2)
        # self.R = 1
        # self.Pinf = solve_discrete_are(self.At, self.bt, self.Q, self.R)

    def dynamics_step(self, xs, us):
        return xs + self.dt*self.dynamics(xs, us)

    def dynamics(self, xs, us):
        xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
        if self.process_noise is not None:
            noise = np.random.uniform(
                low=self.process_noise[:,0],
                high=self.process_noise[:,1],
                size=xs.shape,
            )
            xdot += noise
        return xdot

class DoubleIntegratorOutputFeedback(DoubleIntegrator):
    def __init__(self, process_noise=None, sensing_noise=None):
        super().__init__()
        # self.process_noise = np.array([
        #     [-0.5,0.5],
        #     [-0.01,0.01],
        # ])
        if process_noise is None:
            self.process_noise = 0*0.1*np.dstack([-np.ones(self.num_states), np.ones(self.num_states)])[0]
        else:
            self.process_noise = process_noise*np.dstack([-np.ones(self.num_states), np.ones(self.num_states)])[0]

        # self.sensor_noise = np.array([
        #     [-0.8,0.8],
        #     [-0.0,0.0],
        # ])

        if sensing_noise is None:
            self.sensor_noise = 0*0.1*np.dstack([-np.ones(self.num_outputs), np.ones(self.num_outputs)])[0]
        else:
            self.sensor_noise = sensing_noise*np.dstack([-np.ones(self.num_outputs), np.ones(self.num_outputs)])[0]
        print()
class QuadrotorOutputFeedback(Quadrotor):
    def __init__(self):
        super().__init__(process_noise=None, sensing_noise=None)
        if process_noise is None:
            self.process_noise = 0*0.05*np.dstack([-np.ones(self.num_states), np.ones(self.num_states)])[0]
        else:
            self.process_noise = process_noise*0.05*np.dstack([-np.ones(self.num_states), np.ones(self.num_states)])[0]
        
        if sensing_noise is None:
    
            self.sensor_noise = 0*0.01*np.dstack([-np.ones(self.num_outputs), np.ones(self.num_outputs)])[0]
        else:    
            self.process_noise = sensing_noise*np.dstack([-np.ones(self.num_states), np.ones(self.num_states)])[0]

if __name__ == '__main__':
    from closed_loop.nn import load_model

    dynamics = DoubleIntegrator()
    init_state_range = np.array([ # (num_inputs, 2)
                      [-2.0, -0.5], # x0min, x0max
                      [0.2, 0.8], # x1min, x1max
    ])
    xs, us = dynamics.collect_data(t_max=10, input_constraint=LpInputConstraint(p=np.inf, range=init_state_range),
        num_samples=2420)
    print(xs.shape, us.shape)
    with open(dir_path+'/datasets/double_integrator/xs.pkl', 'wb') as f:
        pickle.dump(xs, f)
    with open(dir_path+'/datasets/double_integrator/us.pkl', 'wb') as f:
        pickle.dump(us, f)

    # # dynamics = DoubleIntegrator()
    # # init_state_range = np.array([ # (num_inputs, 2)
    # #                   [2.5, 3.0], # x0min, x0max
    # #                   [-0.25, 0.25], # x1min, x1max
    # # ])
    # # controller = load_model(name='double_integrator_mpc')

    # dynamics = QuadrotorOutputFeedback()
    # init_state_range = np.array([ # (num_inputs, 2)
    #               [4.65,4.65,2.95,0.94,-0.01,-0.01], # x0min, x0max
    #               [4.75,4.75,3.05,0.96,0.01,0.01] # x1min, x1max
    # ]).T
    # goal_state_range = np.array([
    #                       [3.7,2.5,1.2],
    #                       [4.1,3.5,2.6]            
    # ]).T
    # controller = load_model(name='quadrotor')
    # t_max = 15*dynamics.dt
    # input_constraint = LpInputConstraint(range=init_state_range, p=np.inf)
    # dynamics.show_samples(t_max, input_constraint, save_plot=False, ax=None, show=True, controller=controller)

