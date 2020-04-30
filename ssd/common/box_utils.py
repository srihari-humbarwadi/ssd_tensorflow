import tensorflow as tf

_policy = tf.keras.mixed_precision.experimental.global_policy()


def convert_to_xywh(boxes):
    boxes = tf.cast(boxes, dtype=_policy.compute_dtype)
    return tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2.0,
        boxes[..., 2:] - boxes[..., :2]
    ], axis=-1)


def convert_to_corners(boxes):
    boxes = tf.cast(boxes, dtype=_policy.compute_dtype)
    return tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2.0,
        boxes[..., :2] + boxes[..., 2:] / 2.0
    ], axis=-1)


def rescale_boxes(boxes, original_dims, new_dims):
    boxes = tf.cast(boxes, dtype=_policy.compute_dtype)
    original_dims = tf.cast(original_dims, dtype=_policy.compute_dtype)
    new_dims = tf.cast(new_dims, dtype=_policy.compute_dtype)
    scale = new_dims / original_dims
    return tf.stack([
        boxes[..., 0] * scale[1],
        boxes[..., 1] * scale[0],
        boxes[..., 2] * scale[1],
        boxes[..., 3] * scale[0]
    ], axis=-1)


def relative_to_absolute(boxes, image_dims):
    boxes = tf.cast(boxes, dtype=_policy.compute_dtype)
    image_dims = tf.cast(image_dims, dtype=_policy.compute_dtype)
    return tf.stack([
        boxes[..., 0] * image_dims[1],
        boxes[..., 1] * image_dims[0],
        boxes[..., 2] * image_dims[1],
        boxes[..., 3] * image_dims[0]
    ], axis=-1)


def absolute_to_relative(boxes, image_dims):
    boxes = tf.cast(boxes, dtype=_policy.compute_dtype)
    image_dims = tf.cast(image_dims, dtype=_policy.compute_dtype)
    return tf.stack([
        boxes[..., 0] / image_dims[1],
        boxes[..., 1] / image_dims[0],
        boxes[..., 2] / image_dims[1],
        boxes[..., 3] / image_dims[0]
    ], axis=-1)


def swap_xy(boxes):
    boxes = tf.cast(boxes, dtype=_policy.compute_dtype)
    return tf.stack([
        boxes[:, 1],
        boxes[:, 0],
        boxes[:, 3],
        boxes[:, 2],
    ], axis=-1)
