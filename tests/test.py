import os
import shlex
import subprocess
import unittest
from pathlib import Path

from parameterized import parameterized

EPS = 1e-6
dir_path = os.path.dirname(os.path.abspath(__file__))


def enumerate_tests():
    # Make a list of all the test configs (which end in .yaml)
    # so we can check that we can run each of them
    tests = []
    path_to_configs = f"{dir_path}/../src/nfl_veripy/_static/example_configs"
    print(path_to_configs)
    for path in Path(path_to_configs).rglob("*.yaml"):
        config = str(path).split(path_to_configs)[1]
        tests.append(config)
    tests_to_skip = [
        "/icra21/fig3_reach_sdp.yaml",
        "/icra21/fig3_reach_sdp_partition.yaml",
        "/ojcsys23/quadrotor_hybreach.yaml",
    ]
    print(tests)
    for test in tests_to_skip:
        tests.remove(test)
    tests = [[f"example_configs{test}"] for test in tests]
    return tests


tests = enumerate_tests()

print("---------------------------------")
print("Going to run the following tests:")
print("---------------------------------")
for test in tests:
    print(test[0])
print("---------------------------------")


class TestSum(unittest.TestCase):
    def check_if_cmd_runs(self, config):
        base_cmd = "python -m nfl_veripy.example --config {}"
        command = base_cmd.format(config)

        # Check that code runs without error
        output = subprocess.run(shlex.split(command, posix=True))
        self.assertEqual(output.returncode, 0)

    # Run through each test in tests and check that the command runs
    @parameterized.expand(tests)
    def test_commands_run(self, config):
        self.check_if_cmd_runs(config)


if __name__ == "__main__":
    unittest.main()
