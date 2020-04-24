import logging

import tensorflow as tf


logger = tf.get_logger()
logger.setLevel(logging.INFO)


def get_strategy(config):
    if (not config['use_gpu']) and (not config['use_tpu']):
        strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
        logger.info('Running on CPU')

    elif config['use_gpu']:
        if not tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
            logger.warning('No GPU found! Running on CPU')
        elif config['multi_gpu']:
            strategy = tf.distribute.MirroredStrategy()
            logger.info('Running with MirroredStrategy on {} GPU\'s '.format(strategy.num_replicas_in_sync))
        else:
            strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
            logger.info('Running on GPU')

    elif config['use_tpu']:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(config['tpu_name'])
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            logger.info('Running with TPUStrategy on TPU {} with {} cores '
                        .format(tpu.cluster_spec().as_dict()['worker'],
                                strategy.num_replicas_in_sync))
        except Exception:
            raise ValueError
            strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
            logger.warning('Failed initializing TPU! Running on CPU')
    return strategy
