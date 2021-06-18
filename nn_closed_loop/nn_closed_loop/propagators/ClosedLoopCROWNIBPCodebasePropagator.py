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

        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
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
        elif isinstance(input_constraint, constraints.LpInputConstraint):
            x_min = input_constraint.range[..., 0]
            x_max = input_constraint.range[..., 1]
            norm = input_constraint.p
            A_inputs = None
            b_inputs = None
        else:
            raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            A_out = output_constraint.A
            num_facets = A_out.shape[0]
            bs = np.zeros((num_facets))
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
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
                output_constraint, constraints.PolytopeOutputConstraint
            ):
                bs[i] = A_out_xt1_max
            elif isinstance(output_constraint, constraints.LpOutputConstraint):
                ranges[i, 0] = A_out_xt1_min
                ranges[i, 1] = A_out_xt1_max
            else:
                raise NotImplementedError

        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
            output_constraint.b = bs
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
            output_constraint.range = ranges
        else:
            raise NotImplementedError
        return output_constraint, {}

    def get_one_step_backprojection_set(self, output_constraint, input_constraint):
        # Given an output_constraint, compute the input_constraint
        # that ensures that starting from within the input_constraint
        # will lead to a state within the output_constraint

        # Extract elementwise bounds on xt1 from the lp-ball or polytope constraint
        if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
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
        elif isinstance(output_constraint, constraints.LpOutputConstraint):
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
        num_control_inputs = 1
        xt = cp.Variable(xt1_min.shape+(2,))
        ut = cp.Variable(num_control_inputs)

        A_t = np.eye(xt1_min.shape[0])
        num_facets = A_t.shape[0]
        coords = np.empty((2*num_states, num_states))

        # For each dimension of the output constraint (facet/lp-dimension):
        # compute a bound of the NN output using the pre-computed matrices
        for i in range(num_facets):
            xt = cp.Variable(xt1_min.shape)
            ut = cp.Variable(num_control_inputs)

            constrs = []
            constrs += [u_min <= ut]
            constrs += [ut <= u_max]
            constrs += [self.dynamics.At@xt + self.dynamics.bt@ut <= xt1_max]
            constrs += [self.dynamics.At@xt + self.dynamics.bt@ut >= xt1_min]

            obj = A_t[i, :]@xt
            prob = cp.Problem(cp.Minimize(obj), constrs)
            prob.solve()
            coords[2*i, :] = xt.value
            prob = cp.Problem(cp.Maximize(obj), constrs)
            prob.solve()
            coords[2*i+1, :] = xt.value

        # min/max of each element of xt in the backreachable set
        ranges = np.vstack([coords.min(axis=0), coords.max(axis=0)]).T

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
        # TODO: Make num_partitions a parameter
        num_partitions = np.array([10, 10])
        input_range = ranges
        input_shape = input_range.shape[:-1]
        slope = np.divide(
            (input_range[..., 1] - input_range[..., 0]), num_partitions
        )

        # Set an empty InputConstraint that will get filled in
        input_constraint = constraints.PolytopeInputConstraint(A=[], b=[])

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
        return input_constraint, {}


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
