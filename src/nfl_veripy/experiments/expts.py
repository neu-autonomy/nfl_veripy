from tabulate import tabulate

import nfl_veripy.examples.example as ex


def compare_lp_vs_cf(system):
    rows = []
    rows.append(["", "1", "4", "16"])

    propagator_names = {"CROWNLP": "L.P.", "CROWN": "C.F."}
    t_max = {"quadrotor": "2", "double_integrator": "2"}
    partitions = {
        "quadrotor": ["[1,1,1,1,1,1]", "[2,2,1,1,1,1]", "[2,2,2,2,1,1]"],
        "double_integrator": ["[1,1]", "[2,2]", "[4,4]"],
    }

    parser = ex.setup_parser()

    for propagator in ["CROWNLP", "CROWN"]:
        row = [propagator_names[propagator]]
        for num_partitions in partitions[system]:
            args = parser.parse_args(
                [
                    "--partitioner",
                    "Uniform",
                    "--propagator",
                    propagator,
                    "--system",
                    system,
                    "--state_feedback",
                    "--t_max",
                    t_max[system],
                    "--num_partitions",
                    num_partitions,
                    # "--show_plot",
                    "--estimate_runtime",
                ]
            )
            stats = ex.main(args)

            mean_runtime = stats["runtimes"].mean()
            std_runtime = stats["runtimes"].std()
            runtime_str = "${:.3f} \pm {:.3f}$".format(
                mean_runtime, std_runtime
            )
            row.append(runtime_str)
        rows.append(row)

    print(tabulate(rows, headers="firstrow"))
    print()
    print(tabulate(rows, headers="firstrow", tablefmt="latex_raw"))


if __name__ == "__main__":
    compare_lp_vs_cf("double_integrator")
    compare_lp_vs_cf("quadrotor")
