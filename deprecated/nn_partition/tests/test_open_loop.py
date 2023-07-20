import unittest
import os
import subprocess
import shlex

EPS = 1e-6
figure_dir = "{}/lcss21_figures/".format(
    os.path.dirname(os.path.abspath(__file__))
)
this_dir = "{}/".format(
    os.path.dirname(os.path.abspath(__file__))
)


class TestSum(unittest.TestCase):
    def check_if_cmd_runs(self, file_dir, cmd_filename):
        with open(file_dir + cmd_filename, "r") as file:
            command = file.read().replace("\\", "")
        # Check that code runs without error
        output = subprocess.run(shlex.split(command, posix=True))
        self.assertEqual(output.returncode, 0)

    def test_fig4a(self):
        self.check_if_cmd_runs(figure_dir, "fig4a")

    def test_fig4b(self):
        self.check_if_cmd_runs(figure_dir, "fig4b")

    def test_fig4c(self):
        self.check_if_cmd_runs(figure_dir, "fig4c")

    def test_fig5a(self):
        self.check_if_cmd_runs(figure_dir, "fig5a")

    def test_fig5b(self):
        self.check_if_cmd_runs(figure_dir, "fig5b")

    def test_fig5c(self):
        self.check_if_cmd_runs(figure_dir, "fig5c")

    def test_fig5d(self):
        self.check_if_cmd_runs(figure_dir, "fig5d")

    def test_fig6a(self):
        self.check_if_cmd_runs(figure_dir, "fig6a")

    def test_fig6b(self):
        self.check_if_cmd_runs(figure_dir, "fig6b")

    def test_animate_gsg_crown(self):
        self.check_if_cmd_runs(this_dir, "animate_gsg_crown")

    def test_animate_sg_crown(self):
        self.check_if_cmd_runs(this_dir, "animate_sg_crown")

    def test_animate_agsg_crown(self):
        self.check_if_cmd_runs(this_dir, "animate_agsg_crown")

        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../experiments/results/example/000_learning_2agents.png'
        # self.assertTrue(os.path.isfile(plot_filename))


if __name__ == "__main__":
    unittest.main()
