import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                    help='path to config file;',
                    type=str,
                    # required=True,
                    default='default.cfg')

    parser.add_argument('--root-dir',
                    help='path to root directory;',
                    type=str,
                    # required=True,
                    default='..')

    #Ahmed - add new arguments
    parser.add_argument('--sample',
                        help='sampling method of the dataset',
                        type=str,
                        required=False,
                        default='')

    parser.add_argument('--resume',
                        help='wether to resume from suspended run',
                        type=bool,
                        required=False,
                        default=False)

    parser.add_argument('--metrics-name',
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir',
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    # num epochs will be determined by config file
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')

    return parser.parse_args()
