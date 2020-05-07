import tensorflow as tf

from ssd.common.box_utils import convert_to_corners
from ssd.common.default_boxes import DefaultBoxes


class DecodePredictions(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self._default_boxes = DefaultBoxes(config).boxes
        self._loc_variance = config['loc_variance']
        self._nms_iou_threshold = config['nms_iou_threshold']
        self._nms_score_threshold = config['score_threshold']
        self._input_height = config['image_height']
        self._input_width = config['image_width']
        self._max_detections = config['max_detections']

    def _decode_loc_predictions(self, loc_predictions):
        boxes = loc_predictions * self._loc_variance
        boxes = tf.concat([
            boxes[:, :2] * self._default_boxes[:, 2:] + self._default_boxes[:, :2],
            tf.math.exp(boxes[:, 2:]) * self._default_boxes[:, 2:]
        ],
            axis=-1)
        boxes_transformed = convert_to_corners(boxes)
        boxes_transformed = tf.stack([
            tf.clip_by_value(boxes_transformed[:, 0], 0, self._input_width),
            tf.clip_by_value(boxes_transformed[:, 1], 0, self._input_height),
            tf.clip_by_value(boxes_transformed[:, 2], 0, self._input_width),
            tf.clip_by_value(boxes_transformed[:, 3], 0, self._input_height),
        ], axis=-1)
        return boxes_transformed

    def _decode_cls_predictions(self, cls_predictions):
        cls_ids = tf.argmax(cls_predictions, axis=-1)
        cls_scores = tf.reduce_max(cls_predictions, axis=-1)
        return cls_ids, cls_scores

    def _filter_background_predictions(self, boxes, cls_ids, cls_scores):
        foreground_idx = tf.where(cls_ids != 0)[:, 0]

        filtered_boxes = tf.gather(boxes, foreground_idx)
        filtered_cls_ids = tf.gather(cls_ids, foreground_idx)
        filtered_cls_scores = tf.gather(cls_scores, foreground_idx)
        return filtered_boxes, filtered_cls_ids, filtered_cls_scores

    def call(self, predictions):
        predictions = predictions[0]

        loc_predictions = predictions[:, :4]
        cls_predictions = tf.nn.softmax(predictions[:, 4:])

        boxes = self._decode_loc_predictions(loc_predictions)
        cls_ids, cls_scores = self._decode_cls_predictions(cls_predictions)
        boxes, cls_ids, cls_scores = self._filter_background_predictions(boxes,
                                                                         cls_ids,
                                                                         cls_scores)
        cls_ids = cls_ids - 1  # background is encoded cls_id 0
        nms_idx = tf.image.non_max_suppression(boxes,
                                               cls_scores,
                                               max_output_size=self._max_detections,
                                               iou_threshold=self._nms_iou_threshold,
                                               score_threshold=self._nms_score_threshold)

        decoded_boxes = tf.gather(boxes, nms_idx)
        decoded_cls_ids = tf.gather(cls_ids, nms_idx)
        decoded_cls_scores = tf.gather(cls_scores, nms_idx)

        return decoded_boxes, decoded_cls_ids, decoded_cls_scores

    def compute_output_shape(self, input_shape):
        return ([None, 4], [None], [None])
