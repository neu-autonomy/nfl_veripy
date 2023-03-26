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
from nn_closed_loop.utils.nn_bounds import BoundClosedLoopController

import nn_closed_loop.dynamics as dynamics
from typing import Optional


class ClosedLoopCROWNIBPCodebasePropagator(ClosedLoopPropagator):
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type, num_polytope_facets=num_polytope_facets,
        )
        self.params: dict[str, bool] = {}
        self.method_opt: str = "TODO"

    def torch2network(self, torch_model: torch.nn.Sequential) -> BoundClosedLoopController:
        torch_model_cl = BoundClosedLoopController.convert(
            torch_model, dynamics=self.dynamics, bound_opts=self.params
        )
        return torch_model_cl

    def forward_pass(self, input_data: np.ndarray) -> np.ndarray:
        return self.network(
            torch.Tensor(input_data), method_opt=None
        ).data.numpy()

    def get_one_step_reachable_set(self, initial_set: constraints.SingleTimestepConstraint) -> tuple[constraints.SingleTimestepConstraint, dict]:
        # initial_set: constraints.LpConstraint(range=(num_states, 2))
        # reachable_set: constraints.LpConstraint(range=(num_states, 2))

        A_inputs, b_inputs, x_max, x_min, norm = initial_set.to_reachable_input_objects()

        reachable_set = constraints.create_empty_constraint(self.boundary_type, num_facets=self.num_polytope_facets)
        A_out, num_facets = reachable_set.to_fwd_reachable_output_objects(self.dynamics.num_states)

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

            reachable_set.set_bound(i, A_out_xt1_max, A_out_xt1_min)

        # Auxiliary info
        nn_matrices = {
            'lower_A': lower_A.data.numpy()[0],
            'upper_A': upper_A.data.numpy()[0],
            'lower_sum_b': lower_sum_b.data.numpy()[0],
            'upper_sum_b': upper_sum_b.data.numpy()[0],
        }

        return reachable_set, {'nn_matrices': nn_matrices}

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
        backreachable_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
        overapprox: bool = False,
        infos: dict = {},
    ) -> tuple[Optional[constraints.SingleTimestepConstraint], dict]:

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

        return backprojection_set, infos

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
        backreachable_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.PolytopeConstraint]:
        # For our under-approximation, refer to the Access21 paper.

        if not backreachable_set.crown_matrices:
            raise ValueError('Need to set backreachable_set.crown_matrices.')
        upper_A, lower_A, upper_sum_b, lower_sum_b = backreachable_set.crown_matrices.to_numpy()

        target_set = target_sets.get_constraint_at_time_index(0)
        target_set_range = target_set.to_range()
        assert target_set_range is not None, "target_set.range is None -- make sure it's set to an array."

        # The NN matrices define three types of constraints:
        # - NN's resulting lower bnds on xt1 >= lower bnds on xt1
        # - NN's resulting upper bnds on xt1 <= upper bnds on xt1
        # - NN matrices are only valid within the partition
        A_NN, b_NN = backreachable_set.get_polytope()
        A = np.vstack([
                (self.dynamics.At+self.dynamics.bt@upper_A),
                -(self.dynamics.At+self.dynamics.bt@lower_A),
                A_NN
            ])
        b = np.hstack([
                target_set_range[:, 1] - self.dynamics.bt@upper_sum_b,
                -target_set_range[:, 0] + self.dynamics.bt@lower_sum_b,
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
        backreachable_set: constraints.SingleTimestepConstraint,
        target_sets: constraints.MultiTimestepConstraint,
    ) -> Optional[constraints.SingleTimestepConstraint]:

        num_states, num_control_inputs = self.dynamics.bt.shape

        # An over-approximation of the backprojection set is the set of:
        # all x_t s.t. there exists some u \in [pi^L(x_t), pi^U(x_t)]
        #              that leads to the target set

        xt = cp.Variable(num_states)
        ut = cp.Variable(num_control_inputs)
        constrs = []

        # Constraints to ensure that xt stays within the backreachable set
        A_backreach, b_backreach = backreachable_set.get_polytope()
        constrs += [A_backreach@xt <= b_backreach]

        # Constraints to ensure that ut satisfies the affine bounds
        assert backreachable_set.crown_matrices is not None, "Need to set backreachable_set.crown_matrices."
        constrs += [backreachable_set.crown_matrices.lower_A_numpy@xt+backreachable_set.crown_matrices.lower_sum_b_numpy <= ut]
        constrs += [ut <= backreachable_set.crown_matrices.upper_A_numpy@xt+backreachable_set.crown_matrices.upper_sum_b_numpy]

        # Constraints to ensure xt reaches the "target set" given ut
        # ... where target set = our best bounds on the next state set
        # (i.e., either the true target set or a backprojection set)
        A_target, b_target = target_sets.get_constraint_at_time_index(-1).get_polytope()
        constrs += [A_target@self.dynamics.dynamics_step(xt, ut) <= b_target]

        b, status = optimize_over_all_states(xt, constrs)

        backprojection_set = optimization_results_to_backprojection_set(
            status, b, backreachable_set
        )

        return backprojection_set

    '''
    NOT USED in CDC 2022 Paper

    Essentially combines uses N-step (ReBReach-LP) constraints directly into original (BReach-LP) algorithm

    Inputs: 
        ranges: section of backreachable set
        upper_A, lower_A, upper_sum_b, lower_sum_b: CROWN variables defining upper and lower affine bounds
        xt1max, xt1min: target set max values
        A_t: matrix describing in which directions to optimize
        input_constraint: empty constraint object
        xt_range_min, xt_range_max: min and max x values current overall BP set estimate (expanded as new partitions are analyzed)
        ut_min, ut_max: min and max u values current overall BP set estimate (expanded as new partitions are analyzed)
        collected_input_constraints: BP over-approximations from previously calculated timesteps
        infos: dict containing crown bounds from previously calculated timesteps
    Outputs: 
        input_constraint: one step BP set over-approximation
        xt_range_min, xt_range_max: min and max x values current overall BP set estimate (expanded as new partitions are analyzed)
        ut_min, ut_max: min and max u values current overall BP set estimate (expanded as new partitions are analyzed)
    '''
    def get_refined_one_step_backprojection_set_overapprox(
        self,
        ranges,
        upper_A,
        lower_A,
        upper_sum_b,
        lower_sum_b,
        xt1_max,
        xt1_min,
        A_t,
        xt_range_min,
        xt_range_max,
        ut_min,
        ut_max,
        input_constraint,
        collected_input_constraints,
        infos
    ):
        # TODO (MFE 3/2023): make this look just like get_one_step_backprojection_set_overapprox
        # but also add these constraints
        '''
        for t in range(num_steps):
            # Gather CROWN bounds and previous BP bounds
            if t > 0:
                upper_A = infos[-t]['upper_A']
                lower_A = infos[-t]['lower_A']
                upper_sum_b = infos[-t]['upper_sum_b']
                lower_sum_b = infos[-t]['lower_sum_b']

                # Each xt must fall in the original backprojection
                constrs += [collected_input_constraints[-t].range[:,0] <= xt[:, t]]
                constrs += [xt[:, t] <= collected_input_constraints[-t].range[:,1]]

            # u_t bounded by CROWN bounds
            constrs += [lower_A@xt[:, t]+lower_sum_b <= ut[:, t]]
            constrs += [ut[:, t] <= upper_A@xt[:, t]+upper_sum_b]

            # x_t and x_{t+1} connected through system dynamics
            constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]
        '''
        raise NotImplementedError

    def setup_LPs(self, nstep=False, modifier=0, infos=None, collected_input_constraints=None):
        if nstep:
            return self.setup_LPs_nstep(modifier=modifier, infos=infos, collected_input_constraints=collected_input_constraints)
        else:
            return self.setup_LPs_1step(modifier=modifier, infos=infos, collected_input_constraints=collected_input_constraints)

    def setup_LPs_1step(self, modifier=0, infos=None, collected_input_constraints=None):
        num_states = self.dynamics.At.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]

        xt = cp.Variable(num_states)
        ut = cp.Variable(num_control_inputs)

        lower_A = cp.Parameter((num_control_inputs, num_states))
        upper_A = cp.Parameter((num_control_inputs, num_states))
        lower_sum_b = cp.Parameter(num_control_inputs)
        upper_sum_b = cp.Parameter(num_control_inputs)
        if isinstance(collected_input_constraints[0], constraints.LpConstraint) or isinstance(collected_input_constraints[0], constraints.RotatedLpConstraint):
            xt_min = cp.Parameter(num_states)
            xt_max = cp.Parameter(num_states)

            xt1_min = cp.Parameter(num_states)
            xt1_max = cp.Parameter(num_states)

            A_t = cp.Parameter(num_states)

            params = {
                'lower_A': lower_A,
                'upper_A': upper_A,
                'lower_sum_b': lower_sum_b,
                'upper_sum_b': upper_sum_b,
                'xt_min': xt_min,
                'xt_max': xt_max,
                'xt1_min': xt1_min,
                'xt1_max': xt1_max,
                'A_t': A_t
            }

            constrs = []
            constrs += [lower_A@xt + lower_sum_b <= ut]
            constrs += [ut <= upper_A@xt + upper_sum_b]

            # Included state limits to reduce size of backreachable sets by eliminating states that are not physically possible (e.g., maximum velocities)
            constrs += [xt_min <= xt]
            constrs += [xt <= xt_max]


            # Dynamics must be satisfied

            constrs += [self.dynamics.dynamics_step(xt, ut) <= xt1_max]
            constrs += [self.dynamics.dynamics_step(xt, ut) >= xt1_min]
        
        elif isinstance(collected_input_constraints[0], constraints.RotatedLpConstraint):
            xt_min = cp.Parameter(num_states)
            xt_max = cp.Parameter(num_states)

            R = cp.Parameter((num_states, num_states))
            pose = cp.Parameter(num_states)
            W = cp.Parameter(num_states)

            A_t = cp.Parameter(num_states)

            params = {
                'lower_A': lower_A,
                'upper_A': upper_A,
                'lower_sum_b': lower_sum_b,
                'upper_sum_b': upper_sum_b,
                'xt_min': xt_min,
                'xt_max': xt_max,
                'R': R,
                'W': W,
                'pose': pose,
                'A_t': A_t
            }

            constrs = []
            constrs += [lower_A@xt + lower_sum_b <= ut]
            constrs += [ut <= upper_A@xt + upper_sum_b]

            # Included state limits to reduce size of backreachable sets by eliminating states that are not physically possible (e.g., maximum velocities)
            constrs += [xt_min <= xt]
            constrs += [xt <= xt_max]


            # Dynamics must be satisfied

            constrs += [R@self.dynamics.dynamics_step(xt,ut)-R@pose <= W]
            constrs += [R@self.dynamics.dynamics_step(xt,ut)-R@pose >= np.array([0, 0])]
        
        obj = A_t@xt
        min_prob = cp.Problem(cp.Minimize(obj), constrs)
        max_prob = cp.Problem(cp.Maximize(obj), constrs)

        return max_prob, min_prob, params


    def setup_LPs_nstep(self, nstep=False, modifier=0, infos=None, collected_input_constraints=None):
        num_states = self.dynamics.At.shape[0]
        num_control_inputs = self.dynamics.bt.shape[1]
        
        if isinstance(collected_input_constraints[0], constraints.LpConstraint):
            num_steps = len(collected_input_constraints)
            xt = cp.Variable((num_states, num_steps+1))
            ut = cp.Variable((num_control_inputs, num_steps))


            lower_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
            upper_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
            lower_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]
            upper_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]

            xt_min = [cp.Parameter(num_states) for i in range(num_steps+1)]
            xt_max = [cp.Parameter(num_states) for i in range(num_steps+1)]


            A_t = cp.Parameter(num_states)

            params = {
                'lower_A': lower_A_list,
                'upper_A': upper_A_list,
                'lower_sum_b': lower_sum_b_list,
                'upper_sum_b': upper_sum_b_list,
                'xt_min': xt_min,
                'xt_max': xt_max,
                'A_t': A_t
            }


            constrs = []

            for t in range(num_steps):
                # Gather CROWN bounds and previous BP bounds
                if t > 0:
                    upper_A_list[t].value = infos[-t-modifier]['upper_A']
                    lower_A_list[t].value = infos[-t-modifier]['lower_A']
                    upper_sum_b_list[t].value = infos[-t-modifier]['upper_sum_b']
                    lower_sum_b_list[t].value = infos[-t-modifier]['lower_sum_b']

                    xt_min[t].value = collected_input_constraints[-t-modifier].range[:, 0]
                    xt_max[t].value = collected_input_constraints[-t-modifier].range[:, 1]

                # u_t bounded by CROWN bounds
                constrs += [lower_A_list[t]@xt[:, t]+lower_sum_b_list[t] <= ut[:, t]]
                constrs += [ut[:, t] <= upper_A_list[t]@xt[:, t]+upper_sum_b_list[t]]

                # Each xt must fall in the original backprojection
                constrs += [xt_min[t] <= xt[:, t]]
                constrs += [xt[:, t] <= xt_max[t]]

                # x_t and x_{t+1} connected through system dynamics
                constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]

        elif isinstance(collected_input_constraints[0], constraints.RotatedLpConstraint):

            num_steps = len(collected_input_constraints)
            xt = cp.Variable((num_states, num_steps+1))
            ut = cp.Variable((num_control_inputs, num_steps))


            lower_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
            upper_A_list = [cp.Parameter((num_control_inputs, num_states)) for i in range(num_steps)]
            lower_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]
            upper_sum_b_list = [cp.Parameter(num_control_inputs) for i in range(num_steps)]

            xt_min = [cp.Parameter(num_states)]
            xt_max = [cp.Parameter(num_states)]

            R = [cp.Parameter((num_states, num_states)) for i in range(num_steps+1)]
            pose = [cp.Parameter(num_states) for i in range(num_steps+1)] 
            W = [cp.Parameter(num_states) for i in range(num_steps+1)]


            A_t = cp.Parameter(num_states)

            params = {
                'lower_A': lower_A_list,
                'upper_A': upper_A_list,
                'lower_sum_b': lower_sum_b_list,
                'upper_sum_b': upper_sum_b_list,
                'xt_min': xt_min,
                'xt_max': xt_max,
                'A_t': A_t
            }


            constrs = []

            constrs += [xt_min[0] <= xt[:, 0]]
            constrs += [xt[:, 0] <= xt_max[0]]

            for t in range(num_steps):
                # Gather CROWN bounds and previous BP bounds
                if t > 0:
                    upper_A_list[t].value = infos[-t-modifier]['upper_A']
                    lower_A_list[t].value = infos[-t-modifier]['lower_A']
                    upper_sum_b_list[t].value = infos[-t-modifier]['upper_sum_b']
                    lower_sum_b_list[t].value = infos[-t-modifier]['lower_sum_b']

                    theta = collected_input_constraints[-t-modifier].theta

                    R[t].value = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
                    W[t].value = collected_input_constraints[-t-modifier].width
                    pose[t].value = collected_input_constraints[-t-modifier].pose

                    # Each xt must fall in the previously calculated (rotated) backprojection
                    constrs += [R[t]@xt[:, t] <= W[t] + R[t]@pose[t]]
                    constrs += [R[t]@xt[:, t] >= R[t]@pose[t]]

                # u_t bounded by CROWN bounds
                constrs += [lower_A_list[t]@xt[:, t]+lower_sum_b_list[t] <= ut[:, t]]
                constrs += [ut[:, t] <= upper_A_list[t]@xt[:, t]+upper_sum_b_list[t]]

                # x_t and x_{t+1} connected through system dynamics
                constrs += [self.dynamics.dynamics_step(xt[:, t], ut[:, t]) == xt[:, t+1]]
        obj = A_t@xt[:, 0]
        min_prob = cp.Problem(cp.Minimize(obj), constrs)
        max_prob = cp.Problem(cp.Maximize(obj), constrs)

        return max_prob, min_prob, params


class ClosedLoopIBPPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type, num_polytope_facets=num_polytope_facets,
        )
        raise NotImplementedError
        # TODO: Write nn_bounds.py:BoundClosedLoopController:interval_range
        # (using bound_layers.py:BoundSequential:interval_range)
        self.method_opt = "interval_range"
        self.params = {}


class ClosedLoopCROWNPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type, num_polytope_facets=num_polytope_facets,
        )
        self.method_opt = "full_backward_range"
        # self.params = {"same-slope": False, "zero-lb": True}
        self.params = {"same-slope": False}


class ClosedLoopCROWNLPPropagator(ClosedLoopCROWNPropagator):
    # Same as ClosedLoopCROWNPropagator but don't allow the
    # use of closed-form soln to the optimization, even if it's possible
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type
        )
        self.params['try_to_use_closed_form'] = False


class ClosedLoopFastLinPropagator(ClosedLoopCROWNIBPCodebasePropagator):
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type, num_polytope_facets=num_polytope_facets,
        )
        self.method_opt = "full_backward_range"
        self.params = {"same-slope": True}


class ClosedLoopCROWNNStepPropagator(ClosedLoopCROWNPropagator):
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type, num_polytope_facets=num_polytope_facets,
        )

    def get_reachable_set(self, initial_set: constraints.SingleTimestepConstraint, t_max: int) -> tuple[constraints.MultiTimestepConstraint, dict]:

        # initial_set: constraint.LpConstraint(range=(num_states, 2))
        # reachable_sets: constraint.LpConstraint(range=(num_timesteps, num_states, 2))
        reachable_sets, infos = super().get_reachable_set(
            initial_set, t_max)

        # "undo" the packaging of reachable_sets to simply get a list of the ranges
        reachable_sets = [constraints.LpConstraint(range=r) for r in reachable_sets.range]

        # Symbolically refine all N steps
        tightened_reachable_sets = [] # type: list[constraints.SingleTimestepConstraint]
        tightened_infos = deepcopy(infos)
        N = len(reachable_sets)
        for t in range(2, N + 1):

            # N-step analysis accepts as input:
            # [1st output constraint, {2,...,t} tightened output constraints,
            #  dummy output constraint]
            reachable_set, info = self.get_N_step_reachable_set(
                initial_set, reachable_sets[:1] + tightened_reachable_sets + reachable_sets[:1], infos['per_timestep'][:t]
            )
            tightened_reachable_sets.append(reachable_set)
            tightened_infos['per_timestep'][t-1] = info

        # "redo" the packaging of reachable_sets as range=(num_timesteps, num_states, 2)
        reachable_sets = constraints.list_to_constraint(reachable_sets[:1] + tightened_reachable_sets)

        # Do N-step "symbolic" refinement
        # reachable_set, new_infos = self.get_N_step_reachable_set(
        #     input_constraint, reachable_sets, infos['per_timestep']
        # )
        # reachable_sets[-1] = reachable_set
        # tightened_infos = infos  # TODO: fix this...

        return reachable_sets, tightened_infos

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
            
            input_constraint = deepcopy(input_constraints[-1])
            input_constraint.A = [A_overapprox]
            input_constraint.b = [b_overapprox]

        info = infos[-1]
        info['one_step_backprojection_overapprox'] = input_constraints[-1]

        return input_constraint, info

    def get_N_step_reachable_set(
        self,
        initial_set,
        reachable_sets,
        infos,
    ):

        # TODO: Get this to work for Polytopes too
        # TODO: Confirm this works with partitioners
        # TODO: pull the cvxpy out of this function
        # TODO: add back in noise, observation model

        A_out = np.eye(self.dynamics.At.shape[0])
        num_facets = A_out.shape[0]
        ranges = np.zeros((num_facets, 2))
        num_steps = len(reachable_sets)

        # Because there might sensor noise, the NN could see a different set of
        # states than the system is actually in
        x_min = reachable_sets[-2].range[..., 0]
        x_max = reachable_sets[-2].range[..., 1]
        norm = reachable_sets[-2].p
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
            xs[0] <= initial_set.range[..., 1],
            xs[0] >= initial_set.range[..., 0]
        ]

        # xt+1 \in our one-step overbound on Xt+1
        for t in range(num_steps - 1):
            constraints += [
                xs[t+1] <= reachable_sets[t].range[..., 1],
                xs[t+1] >= reachable_sets[t].range[..., 0]
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

        reachable_set = deepcopy(reachable_sets[-1])
        reachable_set.range = ranges
        return reachable_set, infos


class ClosedLoopCROWNRefinedPropagator(ClosedLoopCROWNPropagator):
    def __init__(self, input_shape: Optional[np.ndarray] = None, dynamics: Optional[dynamics.Dynamics] = None, boundary_type: str = "rectangle", num_polytope_facets: Optional[int] = None):
        super().__init__(
            input_shape=input_shape, dynamics=dynamics, boundary_type=boundary_type, num_polytope_facets=num_polytope_facets,
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




