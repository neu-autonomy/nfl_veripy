import unittest
import os
import subprocess
import shlex
from parameterized import parameterized
from pathlib import Path

EPS = 1e-6
dir_path = os.path.dirname(os.path.abspath(__file__))

def enumerate_tests():
    # Make a list of all the test configs (which end in .yaml)
    # so we can check that we can run each of them
    tests = []
    for path in Path(dir_path).rglob('*.yaml'):
        config = str(path).split(dir_path)[1].split('.')[0]
        tests.append(config)
    tests_to_skip = [
        "/param_version/icra21/fig3_reach_sdp",
        "/param_version/icra21/fig3_reach_sdp_partition",
    ]
    for test in tests_to_skip:
        tests.remove(test)
    tests = [[test] for test in tests]

tests = enumerate_tests()

print('---------------------------------')
print("Going to run the following tests:")
print('---------------------------------')
for test in tests:
    print(test[0])
print('---------------------------------')

class TestSum(unittest.TestCase):
    def check_if_cmd_runs(self, config):
        base_cmd = "python -m nn_closed_loop.example_cfg --config {}/{}"
        command = base_cmd.format(dir_path, config)

        # Check that code runs without error
        output = subprocess.run(shlex.split(command, posix=True))
        self.assertEqual(output.returncode, 0)

    # Run through each test in tests and check that the command runs
    @parameterized.expand(tests)
    def test_commands_run(self, config):
        self.check_if_cmd_runs(config)

        # # Check that plot was generated
        # plot_filename = os.path.dirname(os.path.realpath(__file__)) + '/../results/analyzer/'
        # self.assertTrue(os.path.isfile(plot_filename))


if __name__ == "__main__":
    unittest.main()
