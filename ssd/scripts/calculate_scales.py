import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n',
                        '--num_feature_maps',
                        type=int,
                        required=True,
                        choices=[6, 7],
                        metavar='',
                        help='Number of feature maps')

    parser.add_argument('--s_first',
                        type=float,
                        required=True,
                        metavar='',
                        help='Scale on first feature map')

    parser.add_argument('--smin',
                        type=float,
                        required=True,
                        metavar='',
                        help='Scale on second feature map')

    parser.add_argument('--smax',
                        type=float,
                        required=True,
                        metavar='',
                        help='Scale on last feature map')
    args = parser.parse_args()
    return args


def calculate_scales(s_first, smin, smax, num_feature_maps):
    smin = smin * 100
    smax = smax * 100
    scales = [s_first]
    m = num_feature_maps
    
    for k in range(1, m+2):
        sl = smin + (smax - smin)//(m - 1) * (k - 1)
        scales.append(sl/100)

    return scales


if __name__ == '__main__':
    args = _parse_args()
    s_first = args.s_first
    smin = args.smin
    smax = args.smax
    m = args.num_feature_maps - 1
    print(calculate_scales(s_first, smin, smax, m))