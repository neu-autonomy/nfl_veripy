from nn_closed_loop.example import setup_parser
from nn_closed_loop.example import main as run_experiment


def simple():
    parser = setup_parser()
    args = parser.parse_args()

    args.save_plot = False
    args.system = "double_integrator"
    args.t_max = 5

    expts = [
        # {
        #     'partitioner': 'SimGuided',
        #     'propagator': 'CROWN',
        # },
        {
            'partitioner': 'None',
            'propagator': 'CROWN',
        }
    ]

    for expt in expts:
        for key, value in expt.items():
            setattr(args, key, value)
        stats, info = run_experiment(args)
        print('--')
        # print(stats, info)


if __name__ == '__main__':
    simple()