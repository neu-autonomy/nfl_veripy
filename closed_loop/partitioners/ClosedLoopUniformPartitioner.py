from .ClosedLoopPartitioner import ClosedLoopPartitioner
import closed_loop.constraints as constraints
import numpy as np
import pypoman
from itertools import product
from copy import deepcopy

class ClosedLoopUniformPartitioner(ClosedLoopPartitioner):
    def __init__(self, dynamics, num_partitions=16):
        ClosedLoopPartitioner.__init__(self, dynamics=dynamics)
        self.num_partitions = num_partitions
        self.interior_condition = "linf"
        self.show_animation = False
        self.make_animation = False

    def get_one_step_reachable_set(self, input_constraint, output_constraint, propagator, num_partitions=None):
        reachable_set, info, prob = self.get_reachable_set(input_constraint, output_constraint, propagator, t_max=1, num_partitions=num_partitions)
        return reachable_set, info, prob

    def get_reachable_set(self, input_constraint, output_constraint, propagator, t_max, num_partitions=None):

        if isinstance(input_constraint, constraints.PolytopeInputConstraint):
            A_inputs = input_constraint.A
            b_inputs = input_constraint.b

            # only used to compute slope in non-closedloop manner...
            input_polytope_verts = pypoman.duality.compute_polytope_vertices(A_inputs, b_inputs)
            input_range = np.empty((A_inputs.shape[1],2))
            input_range[:,0] = np.min(np.stack(input_polytope_verts), axis=0)
            input_range[:,1] = np.max(np.stack(input_polytope_verts), axis=0)

        elif isinstance(input_constraint, constraints.LpInputConstraint):
            input_range = input_constraint.range
        else:
            raise NotImplementedError

        info = {}
        num_propagator_calls = 0

        input_shape = input_range.shape[:-1]

        if num_partitions is None:
            num_partitions = np.ones(input_shape, dtype=int)
            if isinstance(self.num_partitions, np.ndarray) and input_shape == self.num_partitions.shape:
                num_partitions = self.num_partitions
            elif len(input_shape) > 1:
                num_partitions[0,0] = self.num_partitions
            else:
                num_partitions *= self.num_partitions
        slope = np.divide((input_range[...,1] - input_range[...,0]), num_partitions)
        
        ranges = []
        reachable_set = None

        for element in product(*[range(num) for num in num_partitions.flatten()]):
            element_ = np.array(element).reshape(input_shape)
            input_range_ = np.empty_like(input_range)
            input_range_[...,0] = input_range[...,0]+np.multiply(element_, slope)
            input_range_[...,1] = input_range[...,0]+np.multiply(element_+1, slope)

            if isinstance(input_constraint, constraints.PolytopeInputConstraint):
                # This is a disaster hack to partition polytopes
                A_rect, b_rect = init_state_range_to_polytope(input_range_)
                rectangle_verts = pypoman.polygon.compute_polygon_hull(A_rect, b_rect)
                input_polytope_verts = pypoman.polygon.compute_polygon_hull(A_inputs, b_inputs)
                partition_verts = pypoman.intersection.intersect_polygons(input_polytope_verts, rectangle_verts)
                A_inputs_, b_inputs_ = pypoman.duality.compute_polytope_halfspaces(partition_verts)
                input_constraint_ = input_constraint.__class__(A_inputs_, b_inputs_)
            elif isinstance(input_constraint, constraints.LpInputConstraint):
                input_constraint_ = input_constraint.__class__(range=input_range_, p=input_constraint.p)
            else:
                raise NotImplementedError

            output_constraint_, info= propagator.get_reachable_set(input_constraint_, deepcopy(output_constraint), t_max)
            num_propagator_calls += t_max

            if isinstance(output_constraint, constraints.PolytopeOutputConstraint):
                reachable_set_ = [o.b for o in output_constraint_]
                if output_constraint.b is None:
                    output_constraint.b = np.stack(reachable_set_)

                tmp = np.dstack([output_constraint.b, np.stack(reachable_set_)])
                output_constraint.b = np.max(tmp, axis=-1)
                
                ranges.append((input_range_, reachable_set_))
            elif isinstance(output_constraint, constraints.LpOutputConstraint):
                reachable_set_ = [o.range for o in output_constraint_]
                if output_constraint.range is None:
                    output_constraint.range = np.stack(reachable_set_)
  
                tmp = np.stack([output_constraint.range, np.stack(reachable_set_)], axis=-1)
               
                output_constraint.range[...,0] = np.min(tmp[...,0,:], axis=-1)
                output_constraint.range[...,1] = np.max(tmp[...,1,:], axis=-1)
                ranges.append((input_range_, np.stack(reachable_set_)))
            else:
                raise NotImplementedError

        info["all_partitions"] = ranges
        info["num_propagator_calls"] = num_propagator_calls
        info["num_partitions"] = np.product(num_partitions)
        prob_list =None
        return output_constraint, info,prob_list