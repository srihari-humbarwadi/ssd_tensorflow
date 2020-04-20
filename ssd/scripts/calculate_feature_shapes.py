import numpy as np
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_height',
                        type=int,
                        help='input height of the model')

    parser.add_argument('--image_width',
                        type=int,
                        help='input width of the model')

    parser.add_argument('--num_feature_maps',
                        type=int,
                        required=True,
                        help='Number of feature maps')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()

    input_height = args.image_height
    input_width = args.image_width
    num_feature_maps = args.num_feature_maps

    fh = input_height
    fw = input_width
    feature_shapes = []

    for _ in range(num_feature_maps):
        fh = int(np.ceil(fh / 2))
        fw = int(np.ceil(fw / 2))
        feature_shapes.append([fh, fw])

    for _ in range(2):
        fh = fh - 3 + 1
        fw = fw - 3 + 1
        feature_shapes.append([fh, fw])

    print(feature_shapes[-num_feature_maps:])

