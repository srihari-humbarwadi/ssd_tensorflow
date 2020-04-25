import logging

import tensorflow as tf

from ssd.common.config import load_config
from ssd.data.dataset_builder import DatasetBuilder
from ssd.losses.multibox_loss import MultiBoxLoss
from ssd.models.ssd_model import SSDModel

print('TensorFlow:', tf.__version__)
logger = tf.get_logger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    config = load_config('ssd/cfg/shapes_dataset.yaml')
    logger.info('\n\nconfig: {}\n\n'.format(config))

    train_dataset = DatasetBuilder('train', config)
    val_dataset = DatasetBuilder('val', config)
    loss = MultiBoxLoss(config)

    model = SSDModel(config)
