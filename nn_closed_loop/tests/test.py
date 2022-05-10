import unittest
import os
import subprocess
import shlex

EPS = 1e-6
figure_dir = "{}/icra21_figures/".format(
    os.path.dirname(os.path.abspath(__file__))
)


class TestSum(unittest.TestCase):
    def check_if_cmd_runs(self, cmd_filename):
        with open(figure_dir + cmd_filename, "r") as file:
            command = file.read().replace("\\", "")
        # Check that code runs without error
        output = subprocess.run(shlex.split(command, posix=True))
        self.assertEqual(output.returncode, 0)

    def test_fig3_reach_sdp(self):
        self.check_if_cmd_runs("fig3_reach_sdp")

    def test_fig3_reach_lp(self):
        self.check_if_cmd_runs("fig3_reach_lp")

    def test_fig3_reach_lp_partition(self):
        self.check_if_cmd_runs("fig3_reach_lp_partition")

    def test_fig4b(self):
        self.check_if_cmd_runs("fig4b")

    def test_fig5a(self):
        self.check_if_cmd_runs("fig5a")

    def test_fig5b(self):
        self.check_if_cmd_runs("fig5b")

    def test_sg(self):
        self.check_if_cmd_runs("../journal_figures/simguided")

    def test_gsg(self):
        self.check_if_cmd_runs("../journal_figures/greedysimguided")

    def test_backreach(self):
        self.check_if_cmd_runs("../journal_figures/backreach")

    def test_duffing(self):
        self.check_if_cmd_runs("../journal_figures/duffing")

    def test_iss(self):
        self.check_if_cmd_runs("../journal_figures/iss")

    def test_crownnstep_double_integrator(self):
        self.check_if_cmd_runs("../newer/crownnstep_double_integrator")

    def test_crownnstep_quadrotor(self):
        self.check_if_cmd_runs("../newer/crownnstep_quadrotor")

        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../results/analyzer/'
        # self.assertTrue(os.path.isfile(plot_filename))


if __name__ == "__main__":
    unittest.main()
