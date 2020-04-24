import logging
import pprint

import tensorflow as tf
from tqdm import tqdm

from ssd.common.box_utils import convert_to_corners
from ssd.common.viz_utils import draw_boxes_cv2, imshow_multiple
from ssd.data.dataset_builder import DatasetBuilder
from ssd.layers.decode_predictions import DecodePredictions
from ssd.common.config import load_config

print('TensorFlow:', tf.__version__)
logger = tf.get_logger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    config = load_config('ssd/cfg/sku110k.yaml')
    logger.info('\n\nconfig: {}\n'.format(config))

    train_dataset = DatasetBuilder('train', config)
    train_ds = train_dataset.dataset.take(1)

    for images, label in train_ds:
        matched_boxes = []
        for i in tqdm(range(images.shape[0])):
            image = images[i]
            save_dir = 'assets/matched_default_boxes/'
            image_name = save_dir + '{}.png'.format(i+1)
            image = image * 127.5 + 127.5
            decoded_boxes, decoded_cls_ids, _ = DecodePredictions(config)(label[i:i+1])
            positive_mask = tf.cast(tf.argmax(label[i, :, 4:], axis=-1) != 0, dtype=tf.float32)
            matched_default_boxes = tf.gather(train_dataset._label_encoder._default_boxes,
                                              tf.where(positive_mask != 0)[:, 0])
            matched_default_boxes_xywh = convert_to_corners(matched_default_boxes)

            default_box_viz = draw_boxes_cv2(image, matched_default_boxes_xywh,
                                             range(len(matched_default_boxes)), show_labels=False)
            gt_box_viz = draw_boxes_cv2(image, decoded_boxes, [
                                        'circle' if x == 0 else 'rectangle' for x in decoded_cls_ids], show_labels=True)
            imshow_multiple([gt_box_viz, default_box_viz, ], ['GT_labels', 'Matched_default_boxes'], save_path=image_name)
            matched_boxes.append(tf.reduce_sum(positive_mask).numpy())
        logger.info('No. of matched default boxes: {}\n\n'.format(matched_boxes))
