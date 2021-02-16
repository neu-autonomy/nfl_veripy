import unittest
import os
from importlib import reload, import_module
import subprocess

EPS = 1e-6

class TestSum(unittest.TestCase):

    def test_example_script(self):
        output = subprocess.run([
            "python", "-m", "nn_partition.example",
                "--partitioner", "GreedySimGuided",
                "--propagator", "CROWN_LIRPA",
                "--term_type", "time_budget",
                "--term_val", "2",
                "--interior_condition", "lower_bnds",
                "--model", "random_weights",
                "--activation", "relu",
                "--show_input", "--show_output",
                "--skip_show_plot", "--save_plot",
            ])
        # Check that code runs without error
        self.assertEqual(output.returncode, 0)
        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/example/000_learning_2agents.png'
        # self.assertTrue(os.path.isfile(plot_filename))

if __name__ == '__main__':
    unittest.main()