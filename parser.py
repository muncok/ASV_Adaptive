import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n_enr', type=str,
                        help='number of enrollments',
                        default='7')

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
                        type=str,
                        help='type of sv_system',
                        choices=['base', 'inc', 'inc_neg', 'inc_update'],
                        required=True
                        )

    parser.add_argument('-trial_sort',
                        type=str,
                        help='trial sorting method',
                        choices=['random', 'sortedPos'],
                        default='no_sort'
                        )

    parser.add_argument('-ths_update',
                        help='use of thresh update',
                        action='store_true')

    parser.add_argument('-incl_init',
                        help='include the init enrollment',
                        action='store_true')

    return parser.parse_args()
