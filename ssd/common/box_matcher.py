import tensorflow as tf

from ssd.common.box_utils import convert_to_corners


class BoxMatcher:

    def __init__(self, iou_threshold=0.5):
        self._iou_threshold = iou_threshold

    def _compute_iou(self, boxes1, boxes2):
        boxes1 = tf.cast(boxes1, dtype=tf.float32)
        boxes2 = tf.cast(boxes2, dtype=tf.float32)

        boxes1_t = convert_to_corners(boxes1)
        boxes2_t = convert_to_corners(boxes2)

        lu = tf.maximum(boxes1_t[:, None, :2], boxes2_t[:, :2])
        rd = tf.minimum(boxes1_t[:, None, 2:], boxes2_t[:, 2:])

        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, 0] * intersection[:, :, 1]

        square1 = boxes1[:, 2] * boxes1[:, 3]
        square2 = boxes2[:, 2] * boxes2[:, 3]

        union_square = tf.maximum(square1[:, None] + square2 - inter_square,
                                  1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def __call__(self, default_boxes, gt_boxes):
        iou_matrix = self._compute_iou(default_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.cast(tf.greater_equal(max_iou, self._iou_threshold),
                                dtype=tf.float32)
        return matched_gt_idx, positive_mask
