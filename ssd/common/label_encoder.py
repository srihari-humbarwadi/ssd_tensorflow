import tensorflow as tf

from ssd.common.box_matcher import BoxMatcher
from ssd.common.default_boxes import DefaultBoxes

_policy = tf.keras.mixed_precision.experimental.global_policy()


class LabelEncoder:

    def __init__(self, config):
        self._default_boxes = DefaultBoxes(config).boxes
        self._loc_variance = tf.constant(config['loc_variance'],
                                         dtype=_policy.compute_dtype)
        self._num_classes = config['num_classes']
        self._box_matcher = BoxMatcher(config['match_iou_threshold'])

    def _compute_loc_target(self, matched_gt_boxes):
        loc_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - self._default_boxes[:, :2]) / self._default_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / self._default_boxes[:, 2:])
            ], axis=-1)
        loc_target = loc_target / self._loc_variance
        return loc_target

    def encode_sample(self, gt_boxes, cls_ids):
        cls_ids = tf.cast(cls_ids + 1, dtype=_policy.compute_dtype)  # add background class with cls_id = 0

        matched_gt_idx, positive_mask = self._box_matcher(
            self._default_boxes, gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)

        loc_target = self._compute_loc_target(matched_gt_boxes)
        cls_target = matched_gt_cls_ids * positive_mask
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([loc_target, cls_target], axis=-1)
        return label
