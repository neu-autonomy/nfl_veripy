import unittest
import os
import subprocess
import shlex

EPS = 1e-6
figure_dir = "{}/".format(
    os.path.dirname(os.path.abspath(__file__))
)


class TestSum(unittest.TestCase):
    def check_if_cmd_runs(self, cmd_filename):
        with open(figure_dir + cmd_filename, "r") as file:
            command = file.read().replace("\\", "")
        # Check that code runs without error
        output = subprocess.run(shlex.split(command, posix=True))
        self.assertEqual(output.returncode, 0)

    # def test_fig3_reach_sdp(self):
        # self.check_if_cmd_runs("icra21_figures/fig3_reach_sdp")

    def test_fig3_reach_lp(self):
        self.check_if_cmd_runs("icra21_figures/fig3_reach_lp")

    def test_fig3_reach_lp_partition(self):
        self.check_if_cmd_runs("icra21_figures/fig3_reach_lp_partition")

    def test_fig4b(self):
        self.check_if_cmd_runs("icra21_figures/fig4b")

    def test_fig5a(self):
        self.check_if_cmd_runs("icra21_figures/fig5a")

    def test_fig5b(self):
        self.check_if_cmd_runs("icra21_figures/fig5b")

    def test_sg(self):
        self.check_if_cmd_runs("journal_figures/simguided")

    def test_gsg(self):
        self.check_if_cmd_runs("journal_figures/greedysimguided")

    def test_backreach(self):
        self.check_if_cmd_runs("journal_figures/backreach")

    def test_duffing(self):
        self.check_if_cmd_runs("journal_figures/duffing")

    def test_iss(self):
        self.check_if_cmd_runs("journal_figures/iss")

    def test_crownnstep_double_integrator(self):
        self.check_if_cmd_runs("newer/crownnstep_double_integrator")

    def test_crownnstep_quadrotor(self):
        self.check_if_cmd_runs("newer/crownnstep_quadrotor")

    def test_cdc_fig3a(self):
        self.check_if_cmd_runs("cdc22_figures/fig3a")

    def test_cdc_fig4a(self):
        self.check_if_cmd_runs("cdc22_figures/fig4a")
    
    def test_cdc_fig4b(self):
        self.check_if_cmd_runs("cdc22_figures/fig4b")

    def test_cdc_fig4c(self):
        self.check_if_cmd_runs("cdc22_figures/fig4c")

    def test_cdc_fig5(self):
        self.check_if_cmd_runs("cdc22_figures/fig5")

    def test_jax_bwd_di_lp(self):
        self.check_if_cmd_runs("jax/jax_bwd_di_lp")

    def test_jax_bwd_di_rect(self):
        self.check_if_cmd_runs("jax/jax_bwd_di_rect")

    def test_jax_fwd_double_integrator(self):
        self.check_if_cmd_runs("jax/jax_fwd_double_integrator")

    def test_jax_fwd_quadrotor(self):
        self.check_if_cmd_runs("jax/jax_fwd_quadrotor")

    def test_breach_double_integrator(self):
        self.check_if_cmd_runs("ojcsys23/di_breach")

    def test_hybreach_double_integrator(self):
        self.check_if_cmd_runs("ojcsys23/di_hybreach")

        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../results/analyzer/'
        # self.assertTrue(os.path.isfile(plot_filename))


if __name__ == "__main__":
    unittest.main()
