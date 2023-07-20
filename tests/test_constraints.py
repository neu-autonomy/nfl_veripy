import os
import unittest

import numpy as np

from nfl_veripy import constraints

EPS = 1e-6
dir_path = os.path.dirname(os.path.abspath(__file__))


class TestLpConstraint(unittest.TestCase):
    def test_instantiate_lp_constraint(self):
        constraint = constraints.LpConstraint(
            range=np.array([[2.5, 3.0], [-0.25, 0.25]])
        )
        np.testing.assert_array_equal(
            constraint.range, np.array([[2.5, 3.0], [-0.25, 0.25]])
        )

    def test_instantiate_add_cells_to_lp_constraint(self):
        constraint = constraints.LpConstraint()
        constraint1 = constraints.LpConstraint(
            range=np.array([[2.5, 3.0], [-0.1, 0.5]])
        )
        constraint2 = constraints.LpConstraint(
            range=np.array([[5.5, 6.0], [-0.25, 0.25]])
        )
        constraint.add_cell(constraint1)
        constraint.add_cell(constraint2)
        constraint.update_main_constraint_with_cells(overapprox=True)
        np.testing.assert_array_equal(
            constraint.range, np.array([[2.5, 6.0], [-0.25, 0.5]])
        )


class TestMultiTimestepLpConstraint(unittest.TestCase):
    def test_instantiate_multi_timestep_lp_constraint(self):
        constraint1 = constraints.LpConstraint(
            range=np.array([[2.5, 3.0], [-0.25, 0.25]])
        )
        constraint2 = constraints.LpConstraint(
            range=np.array([[3.5, 4.0], [-0.25, 0.25]])
        )
        constraint3 = constraints.LpConstraint(
            range=np.array([[4.5, 5.0], [-0.25, 0.25]])
        )
        constraint = constraints.MultiTimestepLpConstraint()
        constraint.constraints = [constraint1, constraint2, constraint3]

        np.testing.assert_array_equal(
            constraint.range,
            np.array(
                [constraint1.range, constraint2.range, constraint3.range]
            ),
        )

    def test_instantiate_add_cells_to_multi_timestep_lp_constraint(self):
        constraint = constraints.MultiTimestepLpConstraint()
        constraint1_1 = constraints.LpConstraint(
            range=np.array([[2.5, 3.0], [-0.1, 0.5]])
        )
        constraint1_2 = constraints.LpConstraint(
            range=np.array([[5.0, 6.0], [-0.25, 0.25]])
        )
        constraint1 = constraints.MultiTimestepLpConstraint(
            constraints=[constraint1_1, constraint1_2]
        )
        constraint2_1 = constraints.LpConstraint(
            range=np.array([[2.0, 4.5], [-0.1, 0.5]])
        )
        constraint2_2 = constraints.LpConstraint(
            range=np.array([[5.5, 7.0], [-0.3, 0.15]])
        )
        constraint2 = constraints.MultiTimestepLpConstraint(
            constraints=[constraint2_1, constraint2_2]
        )
        constraint.add_cell(constraint1)
        constraint.add_cell(constraint2)
        constraint.update_main_constraint_with_cells(overapprox=True)
        np.testing.assert_array_equal(
            constraint.range,
            [
                np.array([[2.0, 4.5], [-0.1, 0.5]]),
                np.array([[5.0, 7.0], [-0.3, 0.25]]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
