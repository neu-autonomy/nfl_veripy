from .ClosedLoopPropagator import ClosedLoopPropagator
import numpy as np
import pypoman
import nn_closed_loop.constraints as constraints
import torch
from nn_closed_loop.utils.utils import (
    range_to_polytope,
    get_crown_matrices
)
from nn_closed_loop.utils.optimization_utils import (
    optimize_over_all_states,
    optimization_results_to_backprojection_set,
)
import cvxpy as cp
from itertools import product
from copy import deepcopy


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

        A_inputs, b_inputs, x_max, x_min, norm = input_constraint.to_reachable_input_objects()
        A_out, num_facets = output_constraint.to_fwd_reachable_output_objects(self.dynamics.num_states)

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        prev_state_max = torch.Tensor(np.array([x_max]))
        prev_state_min = torch.Tensor(np.array([x_min]))
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor(np.array([self.dynamics.sensor_noise[:, 1]]))
            nn_input_min += torch.Tensor(np.array([self.dynamics.sensor_noise[:, 0]]))

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

        for i in range(num_facets):
            # For each dimension of the output constraint (facet/lp-dimension):
            # compute a bound of the NN output using the pre-computed matrices
            if A_out is None:
                A_out_torch = None
            else:
                A_out_torch = torch.Tensor(np.array([A_out[i, :]]))

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

            output_constraint.set_bound(i, A_out_xt1_max, A_out_xt1_min)

        # Auxiliary info
        nn_matrices = {
            'lower_A': lower_A.data.numpy()[0],
            'upper_A': upper_A.data.numpy()[0],
            'lower_sum_b': lower_sum_b.data.numpy()[0],
            'upper_sum_b': upper_sum_b.data.numpy()[0],
        }

        return output_constraint, {'nn_matrices': nn_matrices}

    '''
    Inputs: 
        target_set:
            - constraints.LpConstraint: set of final states to backproject from, OR
            - [constraints.LpConstraint, ...]: [X_T, BP_{-1}, ...,]
        input_constriant: empty constraint object
        overapprox: flag to calculate over approximations of the BP set (set to true gives algorithm 1 in CDC 2022)

    Outputs: 
        input_constraint: one step BP set estimate
        info: dict with extra info (e.g. backreachable set, etc.)
    '''
    def get_one_step_backprojection_set(
        self,
        backreachable_set,
        target_sets,
        overapprox=False,
        infos=None,
    ):

        backreachable_set.crown_matrices = get_crown_matrices(
            self,
            backreachable_set,
            self.dynamics.num_inputs,
            self.dynamics.sensor_noise
        )

        if overapprox:
            backprojection_set = self.get_one_step_backprojection_set_overapprox(
                backreachable_set,
                target_sets
            )
        else:
            backprojection_set = self.get_one_step_backprojection_set_underapprox(
                backreachable_set,
                target_sets
            )

        return backprojection_set, {}

    '''
    Inputs: 
        ranges: section of backreachable set
        upper_A, lower_A, upper_sum_b, lower_sum_b: CROWN variables
        xt1max, xt1min: target set max values
        input_constraint: empty constraint object
    Outputs: 
        input_constraint: one step BP set under-approximation
    '''
    def get_one_step_backprojection_set_underapprox(
        self,
        backreachable_set,
        target_sets
    ):
        # For our under-approximation, refer to the Access21 paper.

        upper_A, lower_A, upper_sum_b, lower_sum_b = backreachable_set.crown_matrices.to_numpy()

        # The NN matrices define three types of constraints:
        # - NN's resulting lower bnds on xt1 >= lower bnds on xt1
        # - NN's resulting upper bnds on xt1 <= upper bnds on xt1
        # - NN matrices are only valid within the partition
        A_NN, b_NN = range_to_polytope(backreachable_set.range)
        A = np.vstack([
                (self.dynamics.At+self.dynamics.bt@upper_A),
                -(self.dynamics.At+self.dynamics.bt@lower_A),
                A_NN
            ])
        b = np.hstack([
                target_sets[0].range[:, 1] - self.dynamics.bt@upper_sum_b,
                -target_sets[0].range[:, 0] + self.dynamics.bt@lower_sum_b,
                b_NN
                ])

        # If those constraints to a non-empty set, then add it to
        # the list of input_constraints. Otherwise, skip it.
        try:
            pypoman.polygon.compute_polygon_hull(A, b+1e-10)
        except:
            return None

        backprojection_set = constraints.PolytopeConstraint(A=A, b=b)

        return backprojection_set

    '''
    CDC 2022 Paper Alg 1

    Inputs: 
        ranges: section of backreachable set
        crown_matrices: CROWNMatrices: CROWN variables defining upper and lower affine bounds
        target_sets: [target_set, BP_{-1}, ..., BP{-T}]
            * Note: only target_set is used for this algorithm! (BPs are discarded)
        backreachable_set: 
    Outputs: 
        backprojection_set:
    '''
    def get_one_step_backprojection_set_overapprox(
        self,
        backreachable_set,
        target_sets
    ):

        num_states, num_control_inputs = self.dynamics.bt.shape

        # An over-approximation of the backprojection set is the set of:
        # all x_t s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        xt = cp.Variable(num_states)
        ut = cp.Variable(num_control_inputs)
        constrs = []

        # Constraints to ensure that xt stays within the backreachable set
        constrs += [backreachable_set.range[:, 0] <= xt]
        constrs += [xt <= backreachable_set.range[:, 1]]

        # Constraints to ensure that ut satisfies the affine bounds
        constrs += [backreachable_set.crown_matrices.lower_A_numpy@xt+backreachable_set.crown_matrices.lower_sum_b_numpy <= ut]
        constrs += [ut <= backreachable_set.crown_matrices.upper_A_numpy@xt+backreachable_set.crown_matrices.upper_sum_b_numpy]

        # Constraints to ensure xt reaches the "target set" given ut
        # ... where target set = our best bounds on the next state set
        # (i.e., either the true target set or a backprojection set)
        constrs += [self.dynamics.dynamics_step(xt, ut) <= target_sets[-1].range[:, 1]]
        constrs += [self.dynamics.dynamics_step(xt, ut) >= target_sets[-1].range[:, 0]]

        b, status = optimize_over_all_states(xt, constrs)

        backprojection_set = optimization_results_to_backprojection_set(
            status, b, backreachable_set
        )
        return backprojection_set


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
        self.params = {"same-slope": False, "zero-lb": True}
        # self.params = {"same-slope": False}


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

    def get_backprojection_set(
        self,
        output_constraints,
        input_constraint,
        t_max=1,
        num_partitions=None,
        overapprox=True,
    ):
        input_constraint_list = []
        tightened_infos_list = []
        if not isinstance(output_constraints, list):
            output_constraint_list = [deepcopy(output_constraints)]
        else:
            output_constraint_list = deepcopy(output_constraints)

        # Initialize backprojections and u bounds (in infos)
        input_constraints, infos = super().get_backprojection_set(
            output_constraint_list, 
            input_constraint, 
            t_max,
            num_partitions=num_partitions, 
            overapprox=overapprox,
        )
        # import pdb; pdb.set_trace()
        for i in range(len(output_constraint_list)):
            tightened_input_constraints, tightened_infos = self.get_single_target_N_step_backprojection_set(output_constraint_list[i], input_constraints[i], infos[i], t_max=t_max, num_partitions=num_partitions, overapprox=overapprox)

            input_constraint_list.append(deepcopy(tightened_input_constraints))
            tightened_infos_list.append(deepcopy(tightened_infos))

        return input_constraint_list, tightened_infos_list

    def get_single_target_N_step_backprojection_set(
        self,
        output_constraint,
        input_constraints,
        infos,
        t_max=1,
        num_partitions=None,
        overapprox=True,
    ):
        # Symbolically refine all N steps
        tightened_input_constraints = []
        tightened_infos = deepcopy(infos)
        N = len(input_constraints)

        for t in range(2, N + 1):

            # N-step analysis accepts as input:
            # [1st input constraint, {2,...,t} tightened input constraints,
            #  dummy input constraint]
            input_constraints_better = deepcopy(input_constraints[:1])
            input_constraints_better += deepcopy(tightened_input_constraints)
            input_constraints_better += deepcopy([input_constraints[t-1]])  # the first pass' backproj overapprox

            input_constraint, info = self.get_N_step_backprojection_set(
                output_constraint, input_constraints_better, infos['per_timestep'][:t], overapprox=overapprox, num_partitions=num_partitions
            )
            tightened_input_constraints.append(input_constraint)
            tightened_infos['per_timestep'][t-1] = info

        input_constraints = input_constraints[:1] + tightened_input_constraints

        return input_constraints, tightened_infos

    '''
    Inputs: 
        output_constraint: target set defining the set of final states to backproject from
        input_constriant: empty constraint object
        num_partitions: array of length nx defining how to partition for each dimension
        overapprox: flag to calculate over approximations of the BP set (set to true gives algorithm 1 in CDC 2022)

    Outputs: 
        input_constraint: one step BP set estimate
        info: dict with extra info (e.g. backreachable set, etc.)
    '''
    def get_N_step_backprojection_set(
        self,
        output_constraint,
        input_constraints,
        infos,
        num_partitions=None,
        overapprox=True,
    ):

        if overapprox:
            input_constraint, info = self.get_N_step_backprojection_set_overapprox(
                output_constraint,
                input_constraints,
                infos,
                num_partitions=num_partitions,
            )
        else:
            input_constraint, info = self.get_N_step_backprojection_set_underapprox()

        return input_constraint, info

    def get_N_step_backprojection_set_underapprox(self):
        raise NotImplementedError

    def get_N_step_backprojection_set_overapprox(
        self,
        output_constraint,
        input_constraints,
        infos,
        num_partitions=None,
    ):
        # TODO: Update for new structure (6/10/22)
        raise NotImplementedError
        # import pdb; pdb.set_trace()
        # Get range of "earliest" backprojection overapprox
        vertices = np.array(
            pypoman.duality.compute_polytope_vertices(
                input_constraints[-1].A[0],
                input_constraints[-1].b[0]
            )
        )
        if isinstance(output_constraint, constraints.LpConstraint):
            tightened_constraint = constraints.LpConstraint(p=np.inf)
        elif isinstance(output_constraint, constraints.PolytopeConstraint):
            tightened_constraint = constraints.PolytopeConstraint(A=[], b=[])
        else:
            raise NotImplementedError
        ranges = np.vstack([vertices.min(axis=0), vertices.max(axis=0)]).T
        input_range = ranges

        # Partition "earliest" backproj overapprox
        if num_partitions is None:
            num_partitions = np.array([10, 10])
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), num_partitions
        )

        num_states = self.dynamics.At.shape[1]
        num_control_inputs = self.dynamics.bt.shape[1]
        num_steps = len(input_constraints)
        xt_range_max = -np.inf*np.ones(num_states)
        xt_range_min = np.inf*np.ones(num_states)
        A_facets = np.vstack([np.eye(num_states), -np.eye(num_states)])
        num_facets = A_facets.shape[0]
        input_shape = input_range.shape[:-1]

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

            # Initialize cvxpy variables
            xt = cp.Variable((num_states, num_steps+1))
            ut = cp.Variable((num_control_inputs, num_steps))
            constrs = []

            # x_{t=0} \in this partition of 0-th backreachable set
            constrs += [xt_min <= xt[:, 0]]
            constrs += [xt[:, 0] <= xt_max]

            # if self.dynamics.x_limits is not None:
            #     x_llim = self.dynamics.x_limits[:, 0]
            #     x_ulim = self.dynamics.x_limits[:, 1]
            

            # # Each xt must be in a backprojection overapprox
            # for t in range(num_steps - 1):
            #     A, b = input_constraints[t].A[0], input_constraints[t].b[0]
            #     constrs += [A@xt[:, t+1] <= b]

            # x_{t=T} must be in target set
            if isinstance(output_constraint, constraints.LpConstraint):
                goal_set_A, goal_set_b = range_to_polytope(output_constraint.range)
            elif isinstance(output_constraint, constraints.PolytopeConstraint):
                goal_set_A, goal_set_b = output_constraint.A, output_constraint.b[0]
            constrs += [goal_set_A@xt[:, -1] <= goal_set_b]

            # Each ut must not exceed CROWN bounds
            for t in range(num_steps):

                if t == 0:
                    lower_A, upper_A, lower_sum_b, upper_sum_b = self.get_crown_matrices(xt_min, xt_max, num_control_inputs)
                else:
                    # Gather CROWN bounds for full backprojection overapprox
                    # import pdb; pdb.set_trace()
                    upper_A = infos[-t-1]['upper_A']
                    lower_A = infos[-t-1]['lower_A']
                    upper_sum_b = infos[-t-1]['upper_sum_b']
                    lower_sum_b = infos[-t-1]['lower_sum_b']

                # u_t bounded by CROWN bounds
                constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
                constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]

            # Each xt must fall in the original backprojection
            for t in range(num_steps):
                constrs += [input_constraints[-t-1].range[:,0] <= xt[:,t]]
                constrs += [xt[:,t] <= input_constraints[-t-1].range[:,1]]


            # x_t and x_{t+1} connected through system dynamics
            for t in range(num_steps):
                constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]

                # if self.dynamics.x_limits is not None:
                #     x_llim = self.dynamics.x_limits[:, 0]
                #     x_ulim = self.dynamics.x_limits[:, 1]
                #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) <= x_ulim]
                #     constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) >= x_llim]

            # u_t satisfies control limits (TODO: Necessary? CROWN should account for these)
            # for t in range(num_steps):
            #     constrs += [-1 <= ut[:, t]]
            #     constrs += [1 >= ut[:, t]]

            A_facets_i = cp.Parameter(num_states)
            obj = A_facets_i@xt[:, 0]
            prob = cp.Problem(cp.Maximize(obj), constrs)
            A_ = A_facets
            b_ = np.empty(num_facets)
            for i in range(num_facets):
                A_facets_i.value = A_facets[i, :]
                prob.solve()

                # prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
                b_[i] = prob.value

            # This cell of the backprojection set is upper-bounded by the
            # cell of the backreachable set that we used in the NN relaxation
            # ==> the polytope is the intersection (i.e., concatenation)
            # of the polytope used for relaxing the NN and the soln to the LP
            A_NN, b_NN = range_to_polytope(ranges)
            A_stack = np.vstack([A_, A_NN])
            b_stack = np.hstack([b_, b_NN])

            if isinstance(output_constraint, constraints.LpConstraint):
                b_max = b_[0:int(len(b_)/2)]
                b_min = -b_[int(len(b_)/2):int(len(b_))]

                xt_range_max = np.max((xt_range_max, b_max),axis=0)
                xt_range_min = np.min((xt_range_min, b_min),axis=0)

                tightened_constraint.range = np.array([xt_range_min,xt_range_max]).T

            elif isinstance(output_constraint, constraints.PolytopeConstraint):
                vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                if len(vertices) > 0:
                    # import pdb; pdb.set_trace()
                    # pypoman.polygon.compute_polygon_hull(A_stack, b_stack+1e-10)
                    # vertices = np.array(pypoman.duality.compute_polytope_vertices(A_stack,b_stack))
                    
                    xt_max_candidate = np.max(vertices, axis=0)
                    xt_min_candidate = np.min(vertices, axis=0)
                    xt_range_max = np.maximum(xt_range_max, xt_max_candidate)
                    xt_range_min = np.minimum(xt_range_min, xt_min_candidate)

                    
                    tightened_constraint.A.append(A_)
                    tightened_constraint.b.append(b_)
            else:
                raise NotImplementedError

        if isinstance(output_constraint, constraints.LpConstraint):
            input_constraint = deepcopy(input_constraints[-1])
            input_constraint.range = np.vstack((xt_range_min, xt_range_max)).T
        elif isinstance(output_constraint, constraints.PolytopeConstraint):
            x_overapprox = np.vstack((xt_range_min, xt_range_max)).T
            A_overapprox, b_overapprox = range_to_polytope(x_overapprox)

            # infos[-1]['tightened_constraint'] = tightened_constraint
            # infos[-1]['tightened_overapprox'] = constraints.PolytopeConstraint(A_overapprox, b_overapprox)
            
            input_constraint = deepcopy(input_constraints[-1])
            input_constraint.A = [A_overapprox]
            input_constraint.b = [b_overapprox]

        info = infos[-1]
        info['one_step_backprojection_overapprox'] = input_constraints[-1]

        return input_constraint, info

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
        prev_state_max = torch.Tensor(np.array([x_max]))
        prev_state_min = torch.Tensor(np.array([x_min]))
        nn_input_max = prev_state_max
        nn_input_min = prev_state_min
        if self.dynamics.sensor_noise is not None:
            nn_input_max += torch.Tensor(np.array([self.dynamics.sensor_noise[:, 1]]))
            nn_input_min += torch.Tensor(np.array([self.dynamics.sensor_noise[:, 0]]))

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

        output_constraint = deepcopy(output_constraints[-1])
        output_constraint.range = ranges
        return output_constraint, infos


class ClosedLoopCROWNRefinedPropagator(ClosedLoopCROWNPropagator):
    def __init__(self, input_shape=None, dynamics=None):
        ClosedLoopCROWNPropagator.__init__(
            self, input_shape=input_shape, dynamics=dynamics
        )

    '''
    Inputs:
    - upper_A, lower_A, upper_sum_b, lower_sum_b:
    - backreachable_set: constraints.LpConstraint: bounds on x_t
    - target_sets: [constraints.LpConstraint, ...]: bounds on x_t+1, ..., x_t+T

    Outputs:
    - backprojection_set: constraints.LpConstraint: subset of backreachable_set
        that leads to all timesteps of target_set (under relaxed NN)
    '''
    def get_one_step_backprojection_set_overapprox(
        self,
        backreachable_set,
        target_sets,
    ):

        # An over-approximation of the backprojection set is the set of:
        # all x_t s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        num_states, num_control_inputs = self.dynamics.bt.shape
        num_steps = len(target_sets)
        
        xt = cp.Variable((num_states, num_steps+1))
        ut = cp.Variable((num_control_inputs, num_steps))
        constrs = []

        # x_{t=0} \in this partition of 0-th backreachable set
        constrs += [backreachable_set.range[:, 0] <= xt[:, 0]]
        constrs += [xt[:, 0] <= backreachable_set.range[:, 1]]

        # Each ut must not exceed CROWN bounds
        for t in range(num_steps):
            if t == 0:
                upper_A, lower_A, upper_sum_b, lower_sum_b = backreachable_set.crown_matrices.to_numpy()
            else:
                upper_A, lower_A, upper_sum_b, lower_sum_b = target_sets[-t].crown_matrices.to_numpy()

            constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
            constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]

        # Each xt must fall in the backprojections/target set
        for t in range(1, num_steps+1):
            constrs += [target_sets[-t].range[:, 0] <= xt[:, t]]
            constrs += [xt[:, t] <= target_sets[-t].range[:, 1]]

        # x_t and x_{t+1} connected through system dynamics
        for t in range(num_steps):
            constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]

        b, status = optimize_over_all_states(xt[:, 0], constrs)

        backprojection_set = optimization_results_to_backprojection_set(
            status, b, backreachable_set
        )
        return backprojection_set
