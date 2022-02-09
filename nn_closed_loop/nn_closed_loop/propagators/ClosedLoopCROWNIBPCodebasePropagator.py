from .ClosedLoopPropagator import ClosedLoopPropagator
import numpy as np
import pypoman
import nn_closed_loop.constraints as constraints
import torch
from nn_closed_loop.utils.utils import range_to_polytope
import cvxpy as cp
from itertools import product


class ClosedLoopCROWNIBPCodebasePropagator(ClosedLoopPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )

    def torch2network(self, torch_model):
        from nn_closed_loop.utils.nn_bounds import BoundClosedLoopController

        torch_model_cl = BoundClosedLoopController.convert(
            torch_model, dynamics=self.dynamics, bound_opts=self.params
        )
        return torch_model_cl

    def forward_pass(self, input_data):
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_one_step_reachable_set(self, input_constraint, output_constraint):

        if isinstance(input_constraint, constraints.PolytopeConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # Get bounds on each state from A_inputs, b_inputs
            try:
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(A_inputs, b_inputs)
                )
            except:
                # Sometimes get arithmetic error... this may fix it
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(
                        A_inputs, b_inputs + 1e-6
                    )
                )
            x_max = np.max(vertices, 0)
            x_min = np.min(vertices, 0)
            norm = np.inf
        elif isinstance(input_constraint, constraints.LpConstraint):
            x_min = input_constraint.range[..., 0]
            x_max = input_constraint.range[..., 1]
            norm = input_constraint.p
            A_inputs = None
            b_inputs = None
        else:
            raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeConstraint):
            A_out = output_constraint.A
            num_facets = A_out.shape[0]
            bs = np.zeros((num_facets))
        elif isinstance(output_constraint, constraints.LpConstraint):
            A_out = np.eye(x_min.shape[0])
            num_facets = A_out.shape[0]
            ranges = np.zeros((num_facets, 2))
        else:
            raise NotImplementedError

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        prev_state_max = torch.Tensor([x_max])
        prev_state_min = torch.Tensor([x_min])
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor([self.dynamics.sensor_noise[:, 1]])
            nn_input_min += torch.Tensor([self.dynamics.sensor_noise[:, 0]])

        # Compute the NN output matrices (for the input constraints)
        num_control_inputs = self.dynamics.bt.shape[1]
        C = torch.eye(num_control_inputs).unsqueeze(0)
        # import pdb; pdb.set_trace()
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
            method_opt=self.method_opt,
            norm=norm,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )

        print("x_max: {}".format(x_max))
        print("x_min: {}".format(x_min))
        upper_A_numpy = upper_A.detach().numpy()
        upper_sum_b_numpy = upper_sum_b.detach().numpy()
        lower_A_numpy = lower_A.detach().numpy()
        lower_sum_b_numpy = lower_sum_b.detach().numpy()
        umax = np.maximum(upper_A_numpy@x_max+upper_sum_b_numpy, upper_A_numpy@x_min+upper_sum_b_numpy)
        umin = np.minimum(lower_A_numpy@x_max+lower_sum_b_numpy, lower_A_numpy@x_min+lower_sum_b_numpy)
        print("upper: {}".format(umax))
        print("lower: {}".format(umin))
        # import pdb; pdb.set_trace()

        for i in range(num_facets):
            # For each dimension of the output constraint (facet/lp-dimension):
            # compute a bound of the NN output using the pre-computed matrices
            if A_out is None:
                A_out_torch = None
            else:
                A_out_torch = torch.Tensor([A_out[i, :]])

            # CROWN was initialized knowing dynamics, no need to pass them here
            # (unless they've changed, e.g., time-varying At matrix)
            (
                A_out_xt1_max,
                A_out_xt1_min,
            ) = self.network.compute_bound_from_matrices(
                lower_A,
                lower_sum_b,
                upper_A,
                upper_sum_b,
                prev_state_max,
                prev_state_min,
                norm,
                A_out_torch,
                A_in=A_inputs,
                b_in=b_inputs,
            )

            if isinstance(
                output_constraint, constraints.PolytopeConstraint
            ):
                bs[i] = A_out_xt1_max
            elif isinstance(output_constraint, constraints.LpConstraint):
                ranges[i, 0] = A_out_xt1_min
                ranges[i, 1] = A_out_xt1_max
            else:
                raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, constraints.LpConstraint):
            output_constraint.range = ranges
        else:
            raise NotImplementedError

        # Auxiliary info
        nn_matrices = {
            'lower_A': lower_A.data.numpy()[0],
            'upper_A': upper_A.data.numpy()[0],
            'lower_sum_b': lower_sum_b.data.numpy()[0],
            'upper_sum_b': upper_sum_b.data.numpy()[0],
        }

        return output_constraint, {'nn_matrices': nn_matrices}

    def get_one_step_backprojection_set(
        self,
        output_constraint,
        input_constraint,
        num_partitions=None,
        overapprox=False,
    ):
        # Given an output_constraint, compute the input_constraint
        # that ensures that starting from within the input_constraint
        # will lead to a state within the output_constraint

        info = {}

        # Extract elementwise bounds on xt1 from the lp-ball or polytope constraint
        if isinstance(output_constraint, constraints.PolytopeConstraint):
            A_t1 = output_constraint.A
            b_t1 = output_constraint.b[0]

            # Get bounds on each state from A_t1, b_t1
            try:
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(A_t1, b_t1)
                )
            except:
                # Sometimes get arithmetic error... this may fix it
                vertices = np.stack(
                    pypoman.compute_polytope_vertices(
                        A_t1, b_t1 + 1e-6
                    )
                )
            xt1_max = np.max(vertices, 0)
            xt1_min = np.min(vertices, 0)
            norm = np.inf
        elif isinstance(output_constraint, constraints.LpConstraint):
            xt1_min = output_constraint.range[..., 0]
            xt1_max = output_constraint.range[..., 1]
            norm = output_constraint.p
            A_t1 = None
            b_t1 = None
        else:
            raise NotImplementedError

        '''
        Step 1: 
        Find backreachable set: all the xt for which there is
        some u in U that leads to a state xt1 in output_constraint
        '''

        if self.dynamics.u_limits is None:
            u_min = -np.inf
            u_max = np.inf
        else:
            u_min = self.dynamics.u_limits[:, 0]
            u_max = self.dynamics.u_limits[:, 1]

        num_states = xt1_min.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]
        
        xt = cp.Variable(xt1_min.shape+(2,))
        ut = cp.Variable(num_control_inputs)

        A_t = np.eye(xt1_min.shape[0])
        num_facets = A_t.shape[0]
        coords = np.empty((2*num_states, num_states))
        # import pdb; pdb.set_trace()

        # For each dimension of the output constraint (facet/lp-dimension):
        # compute a bound of the NN output using the pre-computed matrices
        xt = cp.Variable(xt1_min.shape)
        ut = cp.Variable(num_control_inputs)
        constrs = []
        constrs += [u_min <= ut]
        constrs += [ut <= u_max]
        constrs += [self.dynamics.At@xt + self.dynamics.bt@ut <= xt1_max]
        constrs += [self.dynamics.At@xt + self.dynamics.bt@ut >= xt1_min]
        A_t_i = cp.Parameter(num_states)
        obj = A_t_i@xt
        min_prob = cp.Problem(cp.Minimize(obj), constrs)
        max_prob = cp.Problem(cp.Maximize(obj), constrs)
        for i in range(num_facets):
            A_t_i.value = A_t[i, :]
            min_prob.solve()
            coords[2*i, :] = xt.value
            max_prob.solve()
            coords[2*i+1, :] = xt.value

        # min/max of each element of xt in the backreachable set
        ranges = np.vstack([coords.min(axis=0), coords.max(axis=0)]).T

        backreachable_set = constraints.LpConstraint(range=ranges)
        info['backreachable_set'] = backreachable_set
        from copy import deepcopy
        info['target_set'] = deepcopy(output_constraint)

        '''
        Step 2: 
        Partition the backreachable set (xt).
        For each cell in the partition:
        - relax the NN (use CROWN to compute matrices for affine bounds)
        - use the relaxed NN to compute bounds on xt1
        - use those bounds to define constraints on xt, and if valid, add
            to input_constraint
        '''

        # Setup the partitions
        if num_partitions is None:
            num_partitions = np.array([10, 10])
        input_range = ranges
        input_shape = input_range.shape[:-1]
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), num_partitions
        )

        # Set an empty Constraint that will get filled in
        input_constraint = constraints.PolytopeConstraint(A=[], b=[])
        ut_max = -np.inf*np.ones(num_control_inputs)
        ut_min = np.inf*np.ones(num_control_inputs)
        xt_range_max = -np.inf*np.ones(xt1_min.shape)
        xt_range_min = np.inf*np.ones(xt1_min.shape)
        upper_A_max, lower_A_min = -np.inf*np.ones((num_control_inputs, num_states)),  np.inf*np.ones((num_control_inputs, num_states))
        upper_sum_b_max, lower_sum_b_min = -np.inf*np.ones(num_control_inputs),  np.inf*np.ones(num_control_inputs)

        # Iterate through each partition
        for element in product(
            *[range(num) for num in num_partitions.flatten()]
        ):
            # Compute this partition's min/max xt values
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[..., 0] = input_range[..., 0] + np.multiply(
                element_, slope
            )
            input_range_[..., 1] = input_range[..., 0] + np.multiply(
                element_ + 1, slope
            )
            ranges = input_range_

            # Because there might sensor noise, the NN could see a different
            # set of states than the system is actually in
            xt_min = ranges[..., 0]
            xt_max = ranges[..., 1]
            prev_state_max = torch.Tensor([xt_max])
            prev_state_min = torch.Tensor([xt_min])
            nn_input_max = prev_state_max
            nn_input_min = prev_state_min
            if self.dynamics.sensor_noise is not None:
                raise NotImplementedError
                # nn_input_max += torch.Tensor([self.dynamics.sensor_noise[:, 1]])
                # nn_input_min += torch.Tensor([self.dynamics.sensor_noise[:, 0]])

            # Compute the NN output matrices (for this xt partition)
            num_control_inputs = self.dynamics.bt.shape[1]
            C = torch.eye(num_control_inputs).unsqueeze(0)
            lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
                method_opt=self.method_opt,
                norm=norm,
                x_U=nn_input_max,
                x_L=nn_input_min,
                upper=True,
                lower=True,
                C=C,
                return_matrices=True,
            )

            # Extract numpy array from pytorch tensors
            upper_A = upper_A.detach().numpy()[0]
            lower_A = lower_A.detach().numpy()[0]
            upper_sum_b = upper_sum_b.detach().numpy()[0]
            lower_sum_b = lower_sum_b.detach().numpy()[0]
            # print("current upper: {}".format(upper_A))
            # print("current lower: {}".format(lower_A))
            # print("xt_max: {}".format(xt_max))
            # print("xt_min: {}".format(xt_min))

            if overapprox:

                # An over-approximation of the backprojection set is the set of:
                # all xt s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
                #              that leads to the target set



                # print("xt_max: {}".format(xt_max))
                # print("xt_min: {}".format(xt_min))
                # umax = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
                # umin = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)
                # print("upper: {}".format(umax))
                # print("lower: {}".format(umin))


                xt = cp.Variable(xt1_min.shape)
                ut = cp.Variable(num_control_inputs)
                constrs = []
                constrs += [xt_min <= xt]
                constrs += [xt <= xt_max]
                # constrs += [u_min <= ut]
                # constrs += [ut <= u_max]
                
                constrs += [lower_A@xt+lower_sum_b <= ut]
                constrs += [ut <= upper_A@xt+upper_sum_b]

                # if upper_A.all() == lower_A.all() and upper_sum_b.all() == lower_sum_b.all():
                #     constrs += [upper_A@xt+upper_sum_b == ut]
                # else:
                #     constrs += [lower_A@xt + lower_sum_b <= ut]
                #     constrs += [ut <= upper_A@xt + upper_sum_b]


                constrs += [self.dynamics.At@xt + self.dynamics.bt@ut <= xt1_max]
                constrs += [self.dynamics.At@xt + self.dynamics.bt@ut >= xt1_min]
                A_t_ = np.vstack([A_t, -A_t])
                num_facets = 2*num_states
                A_t_i = cp.Parameter(num_states)
                obj = A_t_i@xt
                prob = cp.Problem(cp.Maximize(obj), constrs)
                A_ = A_t_
                b_ = np.empty(2*num_states)
                for i in range(num_facets):
                    A_t_i.value = A_t_[i, :]
                    prob.solve()
                    b_[i] = prob.value
                    # if prob.status == 'infeasible' and upper_A.all() == lower_A.all() and upper_sum_b.all() == lower_sum_b.all():
                    #     print('upper and lower equal + infeasible')
                    # elif upper_A.all() == lower_A.all() and upper_sum_b.all() == lower_sum_b.all():
                    #     print('just upper and lower equal')
                    #     # print("xt_max: {}".format(xt_max))
                    #     # print("xt_min: {}".format(xt_min))
                    # elif prob.status == 'infeasible':
                    #     print('just infeasible')
                    # else:
                    #     print('feasible')

                # This cell of the backprojection set is upper-bounded by the
                # cell of the backreachable set that we used in the NN relaxation
                # ==> the polytope is the intersection (i.e., concatenation)
                # of the polytope used for relaxing the NN and the soln to the LP
                A_NN, b_NN = range_to_polytope(ranges)
                A_stack = np.vstack([A_, A_NN])
                b_stack = np.hstack([b_, b_NN])

                # Only add that polytope to the list if it's non-empty
                try:
                    pypoman.polygon.compute_polygon_hull(A_stack, b_stack+1e-10)
                    
                    vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                    xt_max_candidate = np.max(vertices, axis=0)
                    xt_min_candidate = np.min(vertices, axis=0)
                    xt_range_max = np.maximum(xt_range_max, xt_max_candidate)
                    xt_range_min = np.minimum(xt_range_min, xt_min_candidate)

                    # print(ranges)
                    # print(A_)
                    # print(A_NN)
                    # print(A_stack)
                    # print(b_)
                    # print(b_NN)
                    # print(b_stack)

                    # print("xt_max: {}".format(xt_max))
                    # print("xt_min: {}".format(xt_min))
                    ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
                    ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)
                    # print("upper_i: {}".format(ut_max))
                    # print("lower_i: {}".format(ut_min))
                    # print("upper: {}".format(ut_max_candidate))
                    # print("lower: {}".format(ut_min_candidate))
                    ut_min = np.minimum(ut_min, ut_min_candidate)
                    ut_max = np.maximum(ut_max, ut_max_candidate)
                    # import pdb; pdb.set_trace()
                    upper_A_max = np.maximum(upper_A_max, upper_A)
                    lower_A_min = np.minimum(lower_A_min, lower_A)
                    upper_sum_b_max = np.maximum(upper_sum_b_max, upper_sum_b)
                    lower_sum_b_min = np.minimum(lower_sum_b_min, lower_sum_b)
                    # print("new max: {}".format(upper_A_max))
                    # print("new min: {}".format(lower_A_min))
                    # import pdb; pdb.set_trace()
                    

                    input_constraint.A.append(A_)
                    input_constraint.b.append(b_)
                except:
                    continue

            else:
                # For our under-approximation, refer to the Access21 paper.

                # The NN matrices define three types of constraints:
                # - NN's resulting lower bnds on xt1 >= lower bnds on xt1
                # - NN's resulting upper bnds on xt1 <= upper bnds on xt1
                # - NN matrices are only valid within the partition
                A_NN, b_NN = range_to_polytope(ranges)
                A_ = np.vstack([
                        (self.dynamics.At+self.dynamics.bt@upper_A),
                        -(self.dynamics.At+self.dynamics.bt@lower_A),
                        A_NN
                    ])
                b_ = np.hstack([
                        xt1_max - self.dynamics.bt@upper_sum_b,
                        -xt1_min + self.dynamics.bt@lower_sum_b,
                        b_NN
                        ])

                # If those constraints to a non-empty set, then add it to
                # the list of input_constraints. Otherwise, skip it.
                try:
                    pypoman.polygon.compute_polygon_hull(A_, b_+1e-10)
                    input_constraint.A.append(A_)
                    input_constraint.b.append(b_)
                except:
                    continue

        # input_constraint contains lists for A, b
        x_overapprox = np.vstack((xt_range_min, xt_range_max)).T
        A_overapprox, b_overapprox = range_to_polytope(x_overapprox)
        info['backproj_overapprox'] = constraints.PolytopeConstraint(A_overapprox, b_overapprox)

        info['u_range'] = np.vstack((ut_min, ut_max)).T
        info['upper_A'] = upper_A_max
        info['lower_A'] = lower_A_min
        info['upper_sum_b'] = upper_sum_b_max
        info['lower_sum_b'] = lower_sum_b_min
        
        return input_constraint, info

    def get_N_step_backprojection_set(self,
            output_constraint,
            input_constraint,
            t_max=1,
            num_partitions=None,
            overapprox=True,
        ):
            if overapprox:
                input_constraint, infos = super().get_backprojection_set(
                    output_constraint, 
                    input_constraint, 
                    t_max,
                    num_partitions=num_partitions, 
                    overapprox=overapprox
                )

                if num_partitions is None:
                    num_partitions = np.array([10, 10])
                vertices = np.array(
                    pypoman.duality.compute_polytope_vertices(
                        infos['per_timestep'][-1]['backproj_overapprox'].A, infos['per_timestep'][-1]['backproj_overapprox'].b
                    )
                )
                tightened_constraint = constraints.PolytopeConstraint(A=[], b=[])
                ranges = np.vstack([vertices.min(axis=0), vertices.max(axis=0)]).T
                input_range = ranges
                input_shape = input_range.shape[:-1]
                slope = np.divide(
                    (input_range[..., 1] - input_range[..., 0]), num_partitions
                )

                num_states = self.dynamics.At.shape[1]
                num_control_inputs = self.dynamics.bt.shape[1]
                ut_max = -np.inf*np.ones(num_control_inputs)
                ut_min = np.inf*np.ones(num_control_inputs)
                xt_range_max = -np.inf*np.ones(num_states)
                xt_range_min = np.inf*np.ones(num_states)
                A_t = np.eye(num_states)
                from copy import deepcopy
                constr_timesteps = deepcopy(infos['per_timestep'])
                constr_timesteps.reverse()
                # import pdb; pdb.set_trace()

                # Iterate through each partition
                for element in product(
                    *[range(num) for num in num_partitions.flatten()]
                ):
                    element_ = np.array(element).reshape(input_shape)
                    input_range_ = np.empty_like(input_range)
                    input_range_[..., 0] = input_range[..., 0] + np.multiply(
                        element_, slope
                    )
                    input_range_[..., 1] = input_range[..., 0] + np.multiply(
                        element_ + 1, slope
                    )
                    ranges = input_range_
                    xt_min = ranges[..., 0]
                    xt_max = ranges[..., 1]





                    nn_input_max = torch.Tensor([xt_max])
                    nn_input_min = torch.Tensor([xt_min])
                    # norm = output_constraint.p

                    # Compute the NN output matrices (for this xt partition)
                    C = torch.eye(num_control_inputs).unsqueeze(0)
                    lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
                        method_opt=self.method_opt,
                        norm=np.inf,
                        x_U=nn_input_max,
                        x_L=nn_input_min,
                        upper=True,
                        lower=True,
                        C=C,
                        return_matrices=True,
                    )
                    upper_A = upper_A.detach().numpy()[0]
                    lower_A = lower_A.detach().numpy()[0]
                    upper_sum_b = upper_sum_b.detach().numpy()[0]
                    lower_sum_b = lower_sum_b.detach().numpy()[0]





                    
                    

                    xt = cp.Variable((num_states, int(t_max+1)))
                    ut = cp.Variable((num_control_inputs, int(t_max+1)))
                    constrs = []
                    # import pdb; pdb.set_trace()
                    constrs += [xt_min <= xt[:,0]]
                    constrs += [xt[:,0] <= xt_max]
                    # # constrs += [u_min <= ut]
                    # # constrs += [ut <= u_max]
                    # constrs += [lower_A@xt + lower_sum_b <= ut]
                    # constrs += [ut <= upper_A@xt + upper_sum_b]
                    for i in range(int(t_max)):
                        u_range = constr_timesteps[i]['u_range']
                        # Gather CROWN bounds for full backprojection overapprox
                        if i > 0:
                            upper_A = constr_timesteps[i]['upper_A']
                            lower_A = constr_timesteps[i]['lower_A']
                            upper_sum_b = constr_timesteps[i]['upper_sum_b']
                            lower_sum_b = constr_timesteps[i]['lower_sum_b']
                        # x_{t+1} must be in the next backprojection overapprox or target set
                        if i < t_max-1:
                            A_i, b_i = constr_timesteps[i+1]['backproj_overapprox'].A, constr_timesteps[i+1]['backproj_overapprox'].b
                        else:
                            A_i, b_i = output_constraint.A, output_constraint.b[0]
                        constrs += [A_i@xt[:,i+1] <= b_i]

                        # u_t bounded by CROWN bounds
                        # constrs += [u_range[:,0] <= ut[:,i]]
                        # constrs += [ut[:,i] <= u_range[:,1]]
                        # if upper_A.all() == lower_A.all() and upper_sum_b.all() == lower_sum_b.all():
                        #     constrs += [lower_A@xt[:,i]+lower_sum_b == ut[:,i]]
                        # else:
                        #     constrs += [lower_A@xt[:,i]+lower_sum_b <= ut[:,i]]
                        #     constrs += [ut[:,i] <= upper_A@xt[:,i]+upper_sum_b]
                        constrs += [lower_A@xt[:,i]+lower_sum_b <= ut[:,i]]
                        constrs += [ut[:,i] <= upper_A@xt[:,i]+upper_sum_b]

                        constrs += [self.dynamics.u_limits[:, 0] <= ut[:,i]]
                        constrs += [ut[:,i] <= self.dynamics.u_limits[:, 1]]

                        # x_t and x_{t+1} connected through system dynamics
                        constrs += [self.dynamics.At@xt[:,i] + self.dynamics.bt@ut[:,i] == xt[:,i+1]]
                        # import pdb; pdb.set_trace()

                    # upper_A = constr_timesteps[-1]['upper_A']
                    # lower_A = constr_timesteps[-1]['lower_A']
                    # upper_sum_b = constr_timesteps[-1]['upper_sum_b']
                    # lower_sum_b = constr_timesteps[-1]['lower_sum_b']
                    # constrs += [lower_A@xt[:,t_max-1]+lower_sum_b <= ut[:,t_max-1]]
                    # constrs += [ut[:,t_max-1] <= upper_A@xt[:,t_max-1]+upper_sum_b]
                    # constrs += [self.dynamics.At@xt[:,t_max-1] + self.dynamics.bt@ut[:,t_max-1] == xt[:,t_max]]
                    # constrs += [output_constraint.A@xt[:,t_max] <= output_constraint.b[0]]
                    # import pdb; pdb.set_trace()
                    A_t_ = np.vstack([A_t, -A_t])
                    num_facets = 2*num_states
                    A_t_i = cp.Parameter(num_states)
                    obj = A_t_i@xt[:,0]
                    prob = cp.Problem(cp.Maximize(obj), constrs)
                    A_ = A_t_
                    b_ = np.empty(2*num_states)
                    for i in range(num_facets):
                        A_t_i.value = A_t_[i, :]
                        prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
                        # if prob.status == 'infeasible' and upper_A.all() == lower_A.all() and upper_sum_b.all() == lower_sum_b.all():
                        #     print('upper and lower equal + infeasible')
                        # elif upper_A.all() == lower_A.all() and upper_sum_b.all() == lower_sum_b.all():
                        #     print('just upper and lower equal')
                        #     # print("xt_max: {}".format(xt_max))
                        #     # print("xt_min: {}".format(xt_min))
                        # elif prob.status == 'infeasible':
                        #     print('just infeasible')
                        # else:
                        #     print('feasible')
                        # if prob.status == 'optimal':
                        #     import pdb; pdb.set_trace()
                        # try:
                        #     prob.solve()
                        # except:
                        #     prob.solve(solver='SCS')
                        b_[i] = prob.value


                    # This cell of the backprojection set is upper-bounded by the
                    # cell of the backreachable set that we used in the NN relaxation
                    # ==> the polytope is the intersection (i.e., concatenation)
                    # of the polytope used for relaxing the NN and the soln to the LP
                    A_NN, b_NN = range_to_polytope(ranges)
                    A_stack = np.vstack([A_, A_NN])
                    b_stack = np.hstack([b_, b_NN])

                    # Only add that polytope to the list if it's non-empty
                    try:
                        pypoman.polygon.compute_polygon_hull(A_stack, b_stack+1e-10)
                        
                        vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                        xt_max_candidate = np.max(vertices, axis=0)
                        xt_min_candidate = np.min(vertices, axis=0)
                        xt_range_max = np.maximum(xt_range_max, xt_max_candidate)
                        xt_range_min = np.minimum(xt_range_min, xt_min_candidate)
                    
                        # print(ranges)
                        # print(A_)
                        # print(A_NN)
                        # print(A_stack)
                        # print(b_)
                        # print(b_NN)
                        # print(b_stack)
                        # print("xt_max: {}".format(xt_range_max))
                        # print("xt_min: {}".format(xt_range_min))
                        # ut_max_candidate = np.maximum(upper_A@xt_max+upper_sum_b, upper_A@xt_min+upper_sum_b)
                        # ut_min_candidate = np.minimum(lower_A@xt_max+lower_sum_b, lower_A@xt_min+lower_sum_b)
                        # print("upper_i: {}".format(ut_max))
                        # print("lower_i: {}".format(ut_min))
                        # print("upper: {}".format(ut_max_candidate))
                        # print("lower: {}".format(ut_min_candidate))
                        # ut_min = np.minimum(ut_min, ut_min_candidate)
                        # ut_max = np.maximum(ut_max, ut_max_candidate)

                        

                        tightened_constraint.A.append(A_)
                        tightened_constraint.b.append(b_)
                    except:
                        continue
                
                
                
                    
                
            else:
                raise NotImplementedError

            infos['tightened_constraint'] = tightened_constraint
            return input_constraint, infos



class ClosedLoopIBPPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        raise NotImplementedError
        # TODO: Write nn_bounds.py:BoundClosedLoopController:interval_range
        # (using bound_layers.py:BoundSequential:interval_range)
        self.method_opt = "interval_range"
        self.params = {}


class ClosedLoopCROWNPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": False}


class ClosedLoopCROWNLPPropagator(ClosedLoopCROWNPropagator):
    # Same as ClosedLoopCROWNPropagator but don't allow the
    # use of closed-form soln to the optimization, even if it's possible
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.params['try_to_use_closed_form'] = False


class ClosedLoopFastLinPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNIBPCodebasePropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": True}


class ClosedLoopCROWNNStepPropagator(ClosedLoopCROWNPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )

    def get_reachable_set(self, input_constraint, output_constraint, t_max):
        from copy import deepcopy

        output_constraints, infos = super().get_reachable_set(
            input_constraint, output_constraint, t_max)

        # Symbolically refine all N steps
        tightened_output_constraints = []
        tightened_infos = deepcopy(infos)
        N = len(output_constraints)
        for t in range(2, N + 1):

            # N-step analysis accepts as input:
            # [1st output constraint, {2,...,t} tightened output constraints,
            #  dummy output constraint]
            output_constraints_better = deepcopy(output_constraints[:1])
            output_constraints_better += deepcopy(tightened_output_constraints)
            output_constraints_better += deepcopy(output_constraints[:1]) # just a dummy

            output_constraint, info = self.get_N_step_reachable_set(
                input_constraint, output_constraints_better, infos['per_timestep'][:t]
            )
            tightened_output_constraints.append(output_constraint)
            tightened_infos['per_timestep'][t-1] = info

        output_constraints = output_constraints[:1] + tightened_output_constraints

        # Do N-step "symbolic" refinement
        # output_constraint, new_infos = self.get_N_step_reachable_set(
        #     input_constraint, output_constraints, infos['per_timestep']
        # )
        # output_constraints[-1] = output_constraint
        # tightened_infos = infos  # TODO: fix this...

        return output_constraints, tightened_infos

    def get_N_step_reachable_set(
        self,
        input_constraint,
        output_constraints,
        infos,
    ):

        # TODO: Get this to work for Polytopes too
        # TODO: Confirm this works with partitioners
        # TODO: pull the cvxpy out of this function
        # TODO: add back in noise, observation model

        A_out = np.eye(self.dynamics.At.shape[0])
        num_facets = A_out.shape[0]
        ranges = np.zeros((num_facets, 2))
        num_steps = len(output_constraints)

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        x_min = output_constraints[-2].range[..., 0]
        x_max = output_constraints[-2].range[..., 1]
        norm = output_constraints[-2].p
        prev_state_max = torch.Tensor([x_max])
        prev_state_min = torch.Tensor([x_min])
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor([self.dynamics.sensor_noise[:, 1]])
            nn_input_min += torch.Tensor([self.dynamics.sensor_noise[:, 0]])

        # Compute the NN output matrices (for the input constraints)
        num_control_inputs = self.dynamics.bt.shape[1]
        C = torch.eye(num_control_inputs).unsqueeze(0)
        lower_A, upper_A, lower_sum_b, upper_sum_b = self.network(
            method_opt=self.method_opt,
            norm=norm,
            x_U=nn_input_max,
            x_L=nn_input_min,
            upper=True,
            lower=True,
            C=C,
            return_matrices=True,
        )
        infos[-1]['nn_matrices'] = {
            'lower_A': lower_A.data.numpy()[0],
            'upper_A': upper_A.data.numpy()[0],
            'lower_sum_b': lower_sum_b.data.numpy()[0],
            'upper_sum_b': upper_sum_b.data.numpy()[0],
        }

        import cvxpy as cp

        num_states = self.dynamics.bt.shape[0]
        xs = []
        for t in range(num_steps+1):
            xs.append(cp.Variable(num_states))

        constraints = []

        # xt \in Xt
        constraints += [
            xs[0] <= input_constraint.range[..., 1],
            xs[0] >= input_constraint.range[..., 0]
        ]

        # xt+1 \in our one-step overbound on Xt+1
        for t in range(num_steps - 1):
            constraints += [
                xs[t+1] <= output_constraints[t].range[..., 1],
                xs[t+1] >= output_constraints[t].range[..., 0]
            ]

        # xt1 connected to xt via dynamics
        for t in range(num_steps):

            # Don't need to consider each state individually if we have Bt >= 0

            # upper_A = infos[t]['nn_matrices']['upper_A']
            # lower_A = infos[t]['nn_matrices']['lower_A']
            # upper_sum_b = infos[t]['nn_matrices']['upper_sum_b']
            # lower_sum_b = infos[t]['nn_matrices']['lower_sum_b']

            # if self.dynamics.continuous_time:
            #     constraints += [
            #         xs[t+1] <= xs[t] + self.dynamics.dt * (self.dynamics.At@xs[t]+self.dynamics.bt@(upper_A@xs[t]+upper_sum_b)+self.dynamics.ct),
            #         xs[t+1] >= xs[t] + self.dynamics.dt * (self.dynamics.At@xs[t]+self.dynamics.bt@(lower_A@xs[t]+lower_sum_b)+self.dynamics.ct),
            #     ]
            # else:
            #     constraints += [
            #         xs[t+1] <= self.dynamics.At@xs[t]+self.dynamics.bt@(upper_A@xs[t]+upper_sum_b)+self.dynamics.ct,
            #         xs[t+1] >= self.dynamics.At@xs[t]+self.dynamics.bt@(lower_A@xs[t]+lower_sum_b)+self.dynamics.ct,
            #     ]

            # Handle case of Bt ! >= 0 by adding a constraint per state
            for j in range(num_states):

                upper_A = np.where(np.tile(self.dynamics.bt[j, :], (num_states, 1)).T >= 0, infos[t]['nn_matrices']['upper_A'], infos[t]['nn_matrices']['lower_A'])
                lower_A = np.where(np.tile(self.dynamics.bt[j, :], (num_states, 1)).T >= 0, infos[t]['nn_matrices']['lower_A'], infos[t]['nn_matrices']['upper_A'])
                upper_sum_b = np.where(self.dynamics.bt[j, :] >= 0, infos[t]['nn_matrices']['upper_sum_b'], infos[t]['nn_matrices']['lower_sum_b'])
                lower_sum_b = np.where(self.dynamics.bt[j, :] >= 0, infos[t]['nn_matrices']['lower_sum_b'], infos[t]['nn_matrices']['upper_sum_b'])

                if self.dynamics.continuous_time:
                    constraints += [
                        xs[t+1][j] <= xs[t][j] + self.dynamics.dt * (self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(upper_A@xs[t]+upper_sum_b)+self.dynamics.ct[j]),
                        xs[t+1][j] >= xs[t][j] + self.dynamics.dt * (self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(lower_A@xs[t]+lower_sum_b)+self.dynamics.ct[j]),
                    ]
                else:
                    constraints += [
                        xs[t+1][j] <= self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(upper_A@xs[t]+upper_sum_b)+self.dynamics.ct[j],
                        xs[t+1][j] >= self.dynamics.At[j, :]@xs[t]+self.dynamics.bt[j, :]@(lower_A@xs[t]+lower_sum_b)+self.dynamics.ct[j],
                    ]

        A_out_facet = cp.Parameter(num_states)

        obj = cp.Maximize(A_out_facet@xs[-1])
        prob_max = cp.Problem(obj, constraints)

        obj = cp.Minimize(A_out_facet@xs[-1])
        prob_min = cp.Problem(obj, constraints)

        for i in range(num_facets):

            A_out_facet.value = A_out[i, :]

            prob_max.solve()
            A_out_xtN_max = prob_max.value

            prob_min.solve()
            A_out_xtN_min = prob_min.value

            ranges[i, 0] = A_out_xtN_min
            ranges[i, 1] = A_out_xtN_max

        from copy import deepcopy
        output_constraint = deepcopy(output_constraints[-1])
        output_constraint.range = ranges
        return output_constraint, infos
