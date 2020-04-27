import tensorflow as tf

from ssd.common.box_utils import convert_to_xywh, rescale_boxes
from ssd.common.label_encoder import LabelEncoder


class DatasetBuilder:

    def __init__(self, split, config):
        self._dataset = None
        self._split = split
        self._backbone = config['backbone']
        self._label_encoder = LabelEncoder(config)
        self._input_height = config['image_height']
        self._input_width = config['image_width']
        self._batch_size = config['batch_size']
        self._tfrecords = tf.data.Dataset.list_files(config['tfrecords_' + split])
        self._augment_val_dataset = config['augment_val_dataset']
        self._random_brightness = config['random_brightness']
        self._random_contrast = config['random_contrast']
        self._random_saturation = config['random_saturation']
        self._random_flip_horizonal = config['random_flip_horizonal']
        self._random_patch = config['random_patch']
        self._brightness_max_delta = config['brightness_max_delta']
        self._contrast_lower = config['contrast_lower']
        self._contrast_upper = config['contrast_upper']
        self._saturation_lower = config['saturation_lower']
        self._saturation_upper = config['saturation_upper']
        self._cache_dataset_in_memory = config['cache_dataset_in_memory']
        self._build_tfrecord_dataset()

    def _random_flip_horizontal_fn(self, image, boxes):
        w = tf.cast(tf.shape(image)[1], dtype=tf.float32)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            boxes = tf.stack(
                [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]],
                axis=-1)
        return image, boxes

    def _random_brightness_fn(self, image):
        image = tf.image.random_brightness(image, self._brightness_max_delta)
        return tf.clip_by_value(image, 0.0, 1)

    def _random_contrast_fn(self, image):
        image = tf.image.random_contrast(image, self._contrast_lower, self._contrast_upper)
        return tf.clip_by_value(image, 0.0, 1)

    def _random_saturation_fn(self, image):
        image = tf.image.random_contrast(image, self._saturation_lower, self._saturation_upper)
        return tf.clip_by_value(image, 0.0, 1)

    def _random_patch_fn(self, image, boxes):
        pass


    def _augment_data(self, image, boxes):
        if self._split == 'val' and not self._augment_val_dataset:
            return image, boxes
        image = image / 255.0
        if self._random_flip_horizonal:
            image, boxes = self._random_flip_horizontal_fn(image, boxes)
        if self._random_brightness:
            image = self._random_brightness_fn(image)
        if self._random_contrast:
            image = self._random_contrast_fn(image)
        if self._random_saturation:
            image = self._random_saturation_fn(image)
        image = image * 255.0
        return image, boxes

    def _parse_example(self, example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'xmins': tf.io.VarLenFeature(tf.float32),
            'ymins': tf.io.VarLenFeature(tf.float32),
            'xmaxs': tf.io.VarLenFeature(tf.float32),
            'ymaxs': tf.io.VarLenFeature(tf.float32),
            'classes': tf.io.VarLenFeature(tf.int64),
        }

        parsed_example = tf.io.parse_single_example(example_proto,
                                                    feature_description)
        classes = tf.sparse.to_dense(parsed_example['classes'])

        image = tf.io.decode_image(parsed_example['image'], channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image.set_shape([None, None, 3])
        original_dims = tf.shape(image)
        image = tf.image.resize(image,
                                size=[self._input_height, self._input_width])

        boxes = tf.stack([
            tf.sparse.to_dense(parsed_example['xmins']),
            tf.sparse.to_dense(parsed_example['ymins']),
            tf.sparse.to_dense(parsed_example['xmaxs']),
            tf.sparse.to_dense(parsed_example['ymaxs']),
        ], axis=-1)
        boxes = rescale_boxes(boxes,
                              [original_dims[0], original_dims[1]],
                              [self._input_height, self._input_width])
        return image, boxes, classes

    def _parse_and_create_label(self, example_proto):
        image, boxes, classes = self._parse_example(example_proto)
        image, boxes = self._augment_data(image, boxes)
        
        if 'resnet' in self._backbone:
            image = (image - 127.5) / 127.5
        
        boxes_xywh = convert_to_xywh(boxes)
        label = self._label_encoder.encode_sample(boxes_xywh, classes)
        return image, label

    def _build_tfrecord_dataset(self):
        dataset = self._tfrecords.interleave(
            tf.data.TFRecordDataset,
            cycle_length=8,
            block_length=32,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self._cache_dataset_in_memory:
            dataset = dataset.cache()
        dataset = dataset.shuffle(512)
        dataset = dataset.map(self._parse_and_create_label,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset
