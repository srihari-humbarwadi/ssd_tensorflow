import argparse
import os

from ssd.evaluators.coco_evaluator import CocoEvaluator
from ssd.common.config import load_config


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=True,
                        metavar='',
                        help='Path to the config')

    parser.add_argument('-d',
                        '--coco_dir',
                        type=str,
                        required=True,
                        metavar='',
                        help='Location of the unzipped coco dataset files')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()

    config = load_config(args.config)
    coco_dir = args.coco_dir
    results_path = os.path.join(coco_dir, 'results.json')

    evaluator = CocoEvaluator(coco_dir, config, results_path)
    evaluator.load_model()
    evaluator.run_inference()
    evaluator.compute_statistics()
