from nn_closed_loop.utils.utils import range_to_polytope
import numpy as np
import platform
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('MACOSX')
import matplotlib.pyplot as plt
from matplotlib import cm

from nn_closed_loop.utils.nn_bounds import BoundClosedLoopController
import nn_closed_loop.constraints as constraints
import torch
import os
import pickle
from colour import Color


dir_path = os.path.dirname(os.path.realpath(__file__))


class Dynamics:
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
        x_limits=None,
    ):

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
            
        self.x_limits = x_limits
        self.dt = dt

        self.name = self.__class__.__name__

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
                low=self.sensor_noise[:, 0],
                high=self.sensor_noise[:, 1],
                size=xs.shape,
            )
            obs += noise
        return obs

    def dynamics_step(self, xs, us):
        raise NotImplementedError

    def colors(self, t_max):
        return [cm.get_cmap(self.cmap_name)(i) for i in range(t_max + 1)]

    def get_sampled_output_range(
        self, input_constraint, t_max=5, num_samples=1000, controller="mpc",
        output_constraint=None
    ):

        xs, us = self.collect_data(
            t_max,
            input_constraint,
            num_samples,
            controller=controller,
            merge_cols=False,
        )

        num_runs, num_timesteps, num_states = xs.shape

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            # hack: just return all the sampled pts for error calculator
            sampled_range = xs
            # num_facets = output_constraint.A.shape[0]
            # all_pts = np.dot(output_constraint.A, xs.T.reshape(num_states, -1))
            # all_pts = all_pts.reshape(num_facets, num_runs, num_timesteps)
            # all_pts = all_pts[..., 1:]  # drop zeroth timestep
            # sampled_range = np.max(all_pts, axis=1).T
        elif isinstance(input_constraint, constraints.LpConstraint):
            sampled_range = np.zeros((num_timesteps - 1, num_states, 2))
            for t in range(1, num_timesteps):
                sampled_range[t - 1, :, 0] = np.min(xs[:, t, :], axis=0)
                sampled_range[t - 1, :, 1] = np.max(xs[:, t, :], axis=0)
        else:
            raise NotImplementedError

        return sampled_range

    def get_state_and_next_state_samples(
        self, input_constraint, t_max=1, num_samples=1000, controller="mpc",
        output_constraint=None,
    ):

        xs, us = self.collect_data(
            t_max,
            input_constraint,
            num_samples,
            controller=controller,
            merge_cols=False,
        )

        return xs[:, 0, :], xs[:, 1, :]

    def get_true_backprojection_set(self, backreachable_set, target_set, t_max=1, controller="mpc"):
        
        
        xs, _ = self.collect_data(
            t_max,
            backreachable_set,
            num_samples=1e6,
            controller=controller,
            merge_cols=False,
        )
        if isinstance(target_set, constraints.PolytopeConstraint):
            A, b = target_set.A, target_set.b[0]
        elif isinstance(target_set, constraints.LpConstraint):
            A, b = range_to_polytope(target_set.range)
        else:
            raise NotImplementedError

        # Find which of the xt+t_max points actually end up in the target set
        within_constraint_inds = np.where(
            np.all(
                (
                    np.dot(A, xs[:, -1, :].T)
                    - np.expand_dims(b, axis=-1)
                )
                <= 0,
                axis=0,
            )
        )
        x_samples_inside_backprojection_set = xs[within_constraint_inds]

        return x_samples_inside_backprojection_set

    def show_samples(
        self,
        t_max,
        input_constraint,
        save_plot=False,
        ax=None,
        show=False,
        controller="mpc",
        input_dims=[[0], [1]],
        zorder=1,
        xs=None,
        colors=None,
    ):
        if ax is None:
            if len(input_dims) == 2:
                projection = None
            elif len(input_dims) == 3:
                projection = '3d'
            ax = plt.subplot(projection=projection)

        if xs is None:
            xs, us = self.collect_data(
                t_max,
                input_constraint,
                num_samples=10000,
                controller=controller,
                merge_cols=False,
            )

        num_runs, num_timesteps, num_states = xs.shape

        if colors is None:
            colors = self.colors(num_timesteps)

        for t in range(num_timesteps):
            ax.scatter(
                *[xs[:, t, i] for i in input_dims],
                color=colors[t],
                s=4,
                zorder=zorder,
            )

        ax.set_xlabel("$x_" + str(input_dims[0][0]) + "$")
        ax.set_ylabel("$x_" + str(input_dims[1][0]) + "$")
        if len(input_dims) == 3:
            ax.set_zlabel("$x_" + str(input_dims[2][0]) + "$")

        if save_plot:
            ax.savefig(plot_name)

        if show:
            plt.show()

    def show_trajectories(
        self,
        t_max,
        input_constraint,
        save_plot=False,
        ax=None,
        show=False,
        controller="mpc",
        input_dims=[[0], [1]],
        zorder=1,
        xs=None,
        colors=None,
    ):
        # import pdb; pdb.set_trace()
        if ax is None:
            if len(input_dims) == 2:
                projection = None
            elif len(input_dims) == 3:
                projection = '3d'
            ax = plt.subplot(projection=projection)

        num_trajectories = 100
        if xs is None:
            xs, us = self.collect_data(
                t_max,
                input_constraint,
                num_samples=num_trajectories*(t_max+self.dt)/self.dt,
                controller=controller,
                merge_cols=False,
            )

        num_runs, num_timesteps, num_states = xs.shape

        if colors is None:
            colors = self.colors(num_timesteps)

        
        orange = Color("orange")
        colors = list(orange.range_to(Color("purple"),num_timesteps))
        # import pdb; pdb.set_trace()
        for traj in range(num_runs):
            if len(input_dims) == 2:
                for seg in range(num_timesteps-1):
                    ax.plot(
                        xs[traj, seg:seg+2, 0],
                        xs[traj, seg:seg+2, 1],
                        color=colors[seg].hex_l,
                        zorder=zorder,
                    )
            elif len(input_dims) == 3:
                for seg in range(num_timesteps-1):
                    ax.plot(
                        xs[traj, seg:seg+2, 0],
                        xs[traj, seg:seg+2, 1],
                        xs[traj, seg:seg+2, 2],
                        color=colors[seg].hex_l,
                        zorder=zorder,
                    )

    def collect_data(
        self,
        t_max,
        input_constraint,
        num_samples=2420,
        controller="mpc",
        merge_cols=True,
    ):
        xs, us = self.run(
            t_max,
            input_constraint,
            num_samples,
            collect_data=True,
            controller=controller,
            merge_cols=merge_cols,
        )
        return xs, us

    def run(
        self,
        t_max,
        input_constraint,
        num_samples=100,
        collect_data=False,
        clip_control=True,
        controller="mpc",
        merge_cols=False,
    ):

        np.random.seed(0)
        num_timesteps = int(
            (t_max + self.dt + np.finfo(float).eps) / (self.dt)
        )
        
        if collect_data:
            np.random.seed(1)
            num_runs = int(num_samples / num_timesteps)
            xs = np.zeros((num_runs, num_timesteps, self.num_states))
            us = np.zeros((num_runs, num_timesteps, self.num_inputs))
        
        # Initial state
        if isinstance(input_constraint, constraints.LpConstraint):
            if input_constraint.p == np.inf:
                xs[:, 0, :] = np.random.uniform(
                    low=input_constraint.range[:, 0],
                    high=input_constraint.range[:, 1],
                    size=(num_runs, self.num_states),
                )
            else:
                raise NotImplementedError
        elif isinstance(input_constraint, constraints.PolytopeConstraint):
            init_state_range = input_constraint.to_linf()
            if isinstance(init_state_range, list):
                # For backreachability, We will have N polytope input 
                # constraints, so sample from those N sets individually then 
                # merge to get (xs, us)

                # want total of num_runs samples, so allocate a (roughly)
                # equal number of "runs" to each polytope
                num_runs_ = np.append(np.arange(0, num_runs, num_runs // len(init_state_range)), num_runs)
                for i in range(len(init_state_range)):
                    # Sample a handful of points
                    xs_ = np.random.uniform(
                        low=init_state_range[i][:, 0],
                        high=init_state_range[i][:, 1],
                        size=(num_runs_[i+1]-num_runs_[i], self.num_states),
                    )
                    # check which of those are within this polytope
                    within_constraint_inds = np.where(
                        np.all(
                            (
                                np.dot(input_constraint.A[i], xs_.T)
                                - np.expand_dims(input_constraint.b[i], axis=-1)
                            )
                            <= 0,
                            axis=0,
                        )
                    )

                    # append polytope-satisfying samples to xs__
                    if i == 0:
                        xs__ = xs_[within_constraint_inds]
                    else:
                        xs__ = np.vstack([xs__, xs_[within_constraint_inds]])

                # assign things so (xs, us) end up as the right shape
                us = np.zeros((xs__.shape[0], num_timesteps, self.num_inputs))
                xs = np.zeros((xs__.shape[0], num_timesteps, self.num_states))
                xs[:, 0, :] = xs__
            else:
                # For forward reachability...
                # sample num_runs pts from within the state range (box)
                # and drop all the points that don't satisfy the polytope
                # constraint
                xs[:, 0, :] = np.random.uniform(
                    low=init_state_range[:, 0],
                    high=init_state_range[:, 1],
                    size=(num_runs, self.num_states),
                )
                within_constraint_inds = np.where(
                    np.all(
                        (
                            np.dot(input_constraint.A, xs[:, 0, :].T)
                            - np.expand_dims(input_constraint.b, axis=-1)
                        )
                        <= 0,
                        axis=0,
                    )
                )
                xs = xs[within_constraint_inds]
                us = us[within_constraint_inds]
        else:
            raise NotImplementedError

        t = 0
        step = 0
        while t < t_max:

            # Observe system (using observer matrix,
            # possibly adding measurement noise)
            obs = self.observe_step(xs[:, step, :])

            # Compute Control
            if controller == "mpc":
                u = self.control_mpc(x0=obs)
            elif isinstance(
                controller, BoundClosedLoopController
            ) or isinstance(controller, torch.nn.Sequential):
                u = self.control_nn(x=obs, model=controller)
            else:
                raise NotImplementedError
            if clip_control and (self.u_limits is not None):
                u = np.clip(u, self.u_limits[:, 0], self.u_limits[:, 1])

            # Step through dynamics (possibly adding process noise)
            xs[:, step + 1, :] = self.dynamics_step(xs[:, step, :], u)

            us[:, step, :] = u
            step += 1
            t += self.dt + np.finfo(float).eps

        if merge_cols:
            return xs.reshape(-1, self.num_states), us.reshape(
                -1, self.num_inputs
            )
        else:
            return xs, us


class ContinuousTimeDynamics(Dynamics):
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
        x_limits=None
    ):
        super().__init__(At, bt, ct, u_limits, dt, c, sensor_noise, process_noise, x_limits)
        self.continuous_time = True

    def dynamics(self, xs, us):
        if isinstance(xs,np.ndarray): # For tracking MC samples
            xdot = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
            if self.process_noise is not None:
                noise = np.random.uniform(
                    low=self.process_noise[:, 0],
                    high=self.process_noise[:, 1],
                    size=xs.shape,
                )
                xdot += noise
        else: # For solving LP
            xdot = self.At@xs + self.bt@us + self.ct
        return xdot

    def dynamics_step(self, xs, us):
        xs_t1 = xs + self.dt * self.dynamics(xs, us)
        
        return xs_t1


class DiscreteTimeDynamics(Dynamics):
    def __init__(
        self,
        At,
        bt,
        ct,
        u_limits=None,
        dt=1.0,
        c=None,
        sensor_noise=None,
        process_noise=None,
        x_limits=None,
    ):
        super().__init__(At, bt, ct, u_limits, dt, c, sensor_noise, process_noise, x_limits)
        self.continuous_time = False

    def dynamics_step(self, xs, us):
        if isinstance(xs, np.ndarray): # For tracking MC samples
            xs_t1 = (np.dot(self.At, xs.T) + np.dot(self.bt, us.T)).T + self.ct
            if self.process_noise is not None:
                noise = np.random.uniform(
                    low=self.process_noise[:, 0],
                    high=self.process_noise[:, 1],
                    size=xs.shape,
                )
                xs_t1 += noise
        else: # For solving LP
            xs_t1 = self.At@xs + self.bt@us + self.ct

        return xs_t1


if __name__ == "__main__":

    from nn_closed_loop.dynamics.DoubleIntegrator import DoubleIntegrator
    dynamics = DoubleIntegrator()
    init_state_range = np.array([
        # (num_inputs, 2)
        [2.5, 3.0],  # x0min, x0max
        [-0.25, 0.25],  # x1min, x1max
    ])
    xs, us = dynamics.collect_data(
        t_max=10,
        input_constraint=constraints.LpConstraint(
            p=np.inf, range=init_state_range
        ),
        num_samples=2420,
    )
    print(xs.shape, us.shape)
    system = "double_integrator"
    with open(dir_path + "/../../datasets/{}/xs.pkl".format(system), "wb") as f:
        pickle.dump(xs, f)
    with open(dir_path + "/../../datasets/{}/us.pkl".format(system), "wb") as f:
        pickle.dump(us, f)

    # from nn_closed_loop.utils.nn import load_model

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
    # input_constraint = LpConstraint(range=init_state_range, p=np.inf)
    # dynamics.show_samples(t_max, input_constraint, save_plot=False, ax=None, show=True, controller=controller)
