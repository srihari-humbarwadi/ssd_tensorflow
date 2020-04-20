import logging
import pprint

import tensorflow as tf
import yaml

from ssd.common.box_utils import convert_to_corners
from ssd.common.viz_utils import draw_boxes_cv2, imshow_multiple
from ssd.data.dataset_builder import DatasetBuilder
from ssd.models.decode_predictions import DecodePredictions

print('TensorFlow:', tf.__version__)
logger = tf.get_logger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    with open('ssd/cfg/shapes_dataset.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    logger.info('\n\nconfig: {}\n\n'.format(config))

    train_dataset = DatasetBuilder('train', config)
    train_ds = train_dataset.dataset.take(1)

    for image, label in train_ds:
        image = image[0]
        image = image * 127.5 + 127.5
        decoded_boxes, decoded_cls_ids, _ = DecodePredictions(config)(label)

        positive_mask = tf.cast(tf.argmax(label[0, :, 4:], axis=-1) != 0, dtype=tf.float32)
        matched_default_boxes = tf.gather(train_dataset._label_encoder._default_boxes,
                                          tf.where(positive_mask != 0)[:, 0])
        matched_default_boxes_xywh = convert_to_corners(matched_default_boxes)

        default_box_viz = draw_boxes_cv2(image, matched_default_boxes_xywh,
                                         range(len(matched_default_boxes)), show_labels=False)
        gt_box_viz = draw_boxes_cv2(image, decoded_boxes, [
                                    'circle' if x == 0 else 'rectangle' for x in decoded_cls_ids], show_labels=True)
        imshow_multiple([gt_box_viz, default_box_viz, ], ['GT_labels', 'Matched_default_boxes'], save_path='default_boxes.png')
        logger.info('\n\nNo. of matched default boxes: {}\n\n'.format(tf.reduce_sum(positive_mask).numpy()))
