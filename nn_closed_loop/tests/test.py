import unittest
import os
from importlib import reload, import_module
import subprocess

EPS = 1e-6

class TestSum(unittest.TestCase):

    def test_example_script(self):
        output = subprocess.run([
            "python", "-m", "nn_closed_loop.example",
                "--partitioner", "None",
                "--propagator", "CROWN",
                "--system", "double_integrator",
                "--t_max", "5",
                "--state_feedback",
                "--skip_show_plot", "--save_plot",
            ])
        # Check that code runs without error
        self.assertEqual(output.returncode, 0)
        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../results/analyzer/'
        # self.assertTrue(os.path.isfile(plot_filename))

if __name__ == '__main__':
    unittest.main()