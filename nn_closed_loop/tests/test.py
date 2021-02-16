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

    def test_fig3_reach_sdp(self):
        command = "python -m nn_closed_loop.example \
                    --partitioner None \
                    --propagator SDP \
                    --system double_integrator \
                    --state_feedback \
                    --t_max 5 \
                    --save_plot --skip_show_plot"
        self.check_if_cmd_runs(command)

    def test_fig3_reach_lp(self):
        command = "python -m nn_closed_loop.example \
                    --partitioner None \
                    --propagator CROWN \
                    --system double_integrator \
                    --state_feedback \
                    --t_max 5 \
                    --save_plot --skip_show_plot"
        self.check_if_cmd_runs(command)

    def test_fig3_reach_lp_partition(self):
        command = "python -m nn_closed_loop.example \
                    --partitioner Uniform \
                    --propagator CROWN \
                    --system double_integrator \
                    --state_feedback \
                    --t_max 5 \
                    --save_plot --skip_show_plot"
        self.check_if_cmd_runs(command)

    def test_fig4b(self):
        command = "python -m nn_closed_loop.example \
                    --partitioner None \
                    --propagator CROWN \
                    --system double_integrator \
                    --state_feedback \
                    --t_max 4 \
                    --save_plot --skip_show_plot \
                    --boundaries polytope \
                    --num_polytope_facets 8 \
                    --init_state_range '[[-2., -1.5], [0.4, 0.8]]' \
                    --save_plot --skip_show_plot"
        self.check_if_cmd_runs(command)

    def test_fig5a(self):
        command = "python -m nn_closed_loop.example \
                    --partitioner None \
                    --propagator CROWN \
                    --system quadrotor \
                    --state_feedback \
                    --t_max 1.2 \
                    --save_plot --skip_show_plot \
                    --boundaries lp \
                    --plot_aspect equal \
                    --save_plot --skip_show_plot"
        self.check_if_cmd_runs(command)

    def test_fig5b(self):
        command = "python -m nn_closed_loop.example \
                    --partitioner None \
                    --propagator CROWN \
                    --system quadrotor \
                    --output_feedback \
                    --t_max 1.2 \
                    --save_plot --skip_show_plot \
                    --boundaries lp \
                    --plot_aspect equal \
                    --save_plot --skip_show_plot"
        self.check_if_cmd_runs(command)

        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../results/analyzer/'
        # self.assertTrue(os.path.isfile(plot_filename))

if __name__ == '__main__':
    unittest.main()