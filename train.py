import logging
import os
from pprint import pprint
import shutil
import sys

import tensorflow as tf

from ssd.common.callbacks import CallbackBuilder
from ssd.common.distribute import get_strategy
from ssd.common.config import load_config
from ssd.data.dataset_builder import DatasetBuilder
from ssd.losses.multibox_loss import MultiBoxLoss
from ssd.models.ssd_model import SSDModel

logger = tf.get_logger()
logger.setLevel(logging.INFO)

logger.info('version: {}'.format(tf.__version__))


config = load_config(sys.argv[1])

strategy = get_strategy(config)

epochs = config['epochs']

lr = config['base_lr']
lr = config['base_lr']
lr = lr if not config['scale_lr'] else lr * strategy.num_replicas_in_sync

batch_size = config['batch_size']
batch_size = batch_size if not config['scale_batch_size'] else batch_size * strategy.num_replicas_in_sync
config['batch_size'] = batch_size

train_steps = config['train_images'] // config['batch_size']
val_steps = config['val_images'] // config['batch_size']

print('\n')
pprint(config, width=120, compact=True)


if config['clear_previous_runs']:
    if config['use_tpu']:
        logger.warning('Skipping GCS Bucket')
    else:
        try:
            shutil.rmtree(os.path.join(config['model_dir']))
            logger.info('Cleared existing model files\n')
        except FileNotFoundError:
            logger.warning('mode_dir not found!')
            os.mkdir(config['model_dir'])

with strategy.scope():
    train_dataset = DatasetBuilder('train', config)
    val_dataset = DatasetBuilder('val', config)

    loss_fn = MultiBoxLoss(config)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    callbacks_list = CallbackBuilder('480', config).get_callbacks()

    model = SSDModel(config)
    model.compile(loss_fn=loss_fn, optimizer=optimizer)
    if config['resume_training']:
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(config['model_dir'] , 'checkpoints'))
        if latest_checkpoint is not None:
            logger.info('Loading weights from {}'.format(latest_checkpoint))
            model.load_weights(latest_checkpoint)
        else:
            logger.warning('No weights found, training from scratch')

model.fit(train_dataset.dataset,
          epochs=epochs,
          steps_per_epoch=train_steps,
          validation_data=val_dataset.dataset,
          validation_steps=val_steps,
          callbacks=callbacks_list)

with strategy.scope():
    save_path = os.path.join(config['model_dir'], 'final_weights', 'ssd_weights')
    logger.info('Saving final weights at in {}'.format(save_path))
    model.save_weights(save_path)
