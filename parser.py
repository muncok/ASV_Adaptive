import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q_size', type=int,
                        help='queue size of enrollments',
                        default='7')

    parser.add_argument('-n_enrs', type=int,
                        help='number of enrollments',
                        default='1')

    parser.add_argument('-out_dir',
                        type=str,
                        help='output_dir base',
                        required=True
                        )

    parser.add_argument('-trial_in',
                        type=str,
                        help='trial directory',
                        required=True
                        )

    parser.add_argument('-n_process',
                        type=int,
                        help='number of processes',
                        default=40)

    parser.add_argument('-sv_mode',
                        nargs='+',
                        help='type of sv_system',
                        required=True
                        )

    parser.add_argument('-trial_sort',
                        type=str,
                        help='trial sorting method',
                        choices=['random', 'sortedPos'],
                        default='no_sort'
                        )

    return parser.parse_args()
