import argparse
import os

import numpy as np
import tensorflow as tf

from ssd.data.parsers.coco_parser import CocoParser
from ssd.data.tfrecord_writer import TFrecordWriter



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--download_path',
                        type=str,
                        required=True,
                        metavar='',
                        help='Location of the unzipped coco dataset files')

    parser.add_argument('-n',
                        '--num_shards',
                        type=int,
                        required=False,
                        default=8,
                        metavar='',
                        help='Number of output tfrecords required')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=True,
                        metavar='',
                        help='Output directory for tfrecords')

    args = parser.parse_args()
    return args


def write_tfrecords(data, num_shards, data_dir, output_dir, split_name):
    tfrecord_writer = TFrecordWriter(n_samples=len(data),
                                     n_shards=num_shards,
                                     output_dir=output_dir,
                                     prefix=split_name)
    bad_samples = 0
    for sample in data:
        image_path = os.path.join(data_dir, sample['image'])

        try:
            with tf.io.gfile.GFile(image_path, 'rb') as fp:
                image = fp.read()
                tf.image.decode_image(image)
        except Exception:
            bad_samples += 1
            continue

        tfrecord_writer.push(image,
                             np.array(sample['label']['boxes'], dtype=np.float32),
                             np.array(sample['label']['classes'], dtype=np.int32))
    tfrecord_writer.flush_last()
    print('Skipped {} corrupted samples from {} data'.format(bad_samples, split_name))


if __name__ == '__main__':
    args = _parse_args()
    download_path = args.download_path
    num_shards = args.num_shards
    output_dir = args.output_dir

    coco_parser = CocoParser(download_path)
    train_data = coco_parser.dataset['train']
    val_data = coco_parser.dataset['val']

    train_data_dir = os.path.join(download_path, 'train2017')
    val_data_dir = os.path.join(download_path, 'val2017')

    write_tfrecords(train_data, num_shards, train_data_dir, output_dir, 'train')
    write_tfrecords(val_data, num_shards, val_data_dir, output_dir, 'val')
