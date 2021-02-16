import unittest
import os
from importlib import reload, import_module
import subprocess

EPS = 1e-6

class TestSum(unittest.TestCase):

    def check_if_cmd_runs(self, command):
        # Check that code runs without error
        output = subprocess.run(command.split())
        self.assertEqual(output.returncode, 0)

    def test_fig4a(self):
        command = "python -m nn_partition.example \
                    --partitioner GreedySimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition lower_bnds \
                    --model random_weights \
                    --activation relu \
                    --show_input --show_output \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig4b(self):
        command = "python -m nn_partition.example \
                    --partitioner GreedySimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition linf \
                    --model random_weights \
                    --activation relu \
                    --show_input --show_output \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig4c(self):
        command = "python -m nn_partition.example \
                    --partitioner GreedySimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model random_weights \
                    --activation relu \
                    --show_input --show_output \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig5a(self):
        command = "python -m nn_partition.example \
                    --partitioner SimGuided \
                    --propagator IBP_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model random_weights \
                    --activation relu \
                    --input_plot_labels None None \
                    --show_input --skip_show_output \
                    --input_plot_aspect equal \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig5b(self):
        command = "python -m nn_partition.example \
                    --partitioner SimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model random_weights \
                    --activation relu \
                    --input_plot_labels None None \
                    --show_input --skip_show_output \
                    --input_plot_aspect equal \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig5c(self):
        command = "python -m nn_partition.example \
                    --partitioner GreedySimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model random_weights \
                    --activation relu \
                    --input_plot_labels None None \
                    --show_input --skip_show_output \
                    --input_plot_aspect equal \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig5d(self):
        command = "python -m nn_partition.example \
                    --partitioner AdaptiveGreedySimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model random_weights \
                    --activation relu \
                    --input_plot_labels None None \
                    --show_input --skip_show_output \
                    --input_plot_aspect equal \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig6a(self):
        command = "python -m nn_partition.example \
                    --partitioner SimGuided \
                    --propagator IBP_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model robot_arm \
                    --activation tanh \
                    --output_plot_labels x y \
                    --output_plot_aspect equal \
                    --skip_show_input \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

    def test_fig6b(self):
        command = "python -m nn_partition.example \
                    --partitioner AdaptiveGreedySimGuided \
                    --propagator CROWN_LIRPA \
                    --term_type time_budget \
                    --term_val 2 \
                    --interior_condition convex_hull \
                    --model robot_arm \
                    --activation tanh \
                    --output_plot_labels x y \
                    --output_plot_aspect equal \
                    --skip_show_input \
                    --skip_show_plot --save_plot"
        self.check_if_cmd_runs(command)

        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/example/000_learning_2agents.png'
        # self.assertTrue(os.path.isfile(plot_filename))

if __name__ == '__main__':
    unittest.main()