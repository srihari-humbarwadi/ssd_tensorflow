import tensorflow as tf
import numpy as np


class DefaultBoxes:

    def __init__(self, config):
        self._input_height = config['image_height']
        self._input_width = config['image_width']

        self._scales = config['scales']
        self._feature_sizes = config['feature_sizes']
        self._strides = tf.constant(
            [[np.ceil(config['image_height'] / x[0]),
              np.ceil(config['image_width'] / x[1])]
             for x in config['feature_sizes']], dtype=tf.float32)
        self._aspect_ratios = config['aspect_ratios']
        self.clip_default_boxes = config['clip_default_boxes']
        self._default_boxes = []
        self._build_default_boxes()

    def _get_centers(self, feature_size, stride):
        rx = tf.range(feature_size[1], dtype=tf.float32)
        ry = tf.range(feature_size[0], dtype=tf.float32)
        centers = tf.stack(tf.meshgrid((0.5 + rx) * stride[1],
                                       (0.5 + ry) * stride[0]), axis=-1)
        return tf.cast(centers, dtype=tf.float32)

    def _get_dims(self, scale, ratio):
        h = self._input_height * scale / np.sqrt(ratio)
        w = self._input_width * scale * np.sqrt(ratio)
        wh = tf.constant([w, h], dtype=tf.float32, shape=[1, 1, 2])
        return wh

    def _build_default_boxes(self):
        default_boxes = []
        for i in range(len(self._feature_sizes)):
            feature_size = self._feature_sizes[i]
            aspect_ratios = self._aspect_ratios[i]
            stride = self._strides[i]
            sl = self._scales[i]
            sl_next = self._scales[i + 1]
            centers = self._get_centers(feature_size, stride)

            default_box = []
            for ratio in aspect_ratios:
                wh = self._get_dims(sl, ratio)
                wh = tf.tile(wh,
                             multiples=[feature_size[0], feature_size[1], 1])
                box = tf.concat([centers, wh], axis=-1)
                box = tf.expand_dims(box, axis=2)
                default_box.append(box)

            extra_wh = tf.constant([
                self._input_height * np.sqrt(sl * sl_next),
                self._input_width * np.sqrt(sl * sl_next)
            ],
                dtype=tf.float32,
                shape=[1, 1, 2])
            extra_wh = tf.tile(extra_wh,
                               multiples=[feature_size[0], feature_size[1], 1])
            extra_box = tf.concat([centers, extra_wh], axis=-1)
            extra_box = tf.expand_dims(extra_box, axis=2)
            default_box.append(extra_box)

            default_box = tf.concat(default_box, axis=2)
            default_box = tf.reshape(default_box, shape=[-1, 4])
            default_boxes.append(default_box)
        self._default_boxes = tf.concat(default_boxes, axis=0)

        if self.clip_default_boxes:
            raise NotImplementedError('Clipping default boxes is yet not supported')

    @property
    def boxes(self):
        return self._default_boxes

