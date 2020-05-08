import json
import os

import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm

from pycocotools.cocoeval import COCOeval
from ssd.common.box_utils import rescale_boxes
from ssd.data.parsers.coco_parser import CocoParser
from ssd.models.ssd_model import SSDModel

logger = tf.get_logger()


class CocoEvaluator(CocoParser):
    def __init__(self, download_path, config, output_json_path):
        super(CocoEvaluator, self).__init__(download_path, only_val=True)
        self._image_dir = download_path + '/val2017/'
        self._input_height = config['image_height']
        self._input_width = config['image_width']
        self._model_dir = config['model_dir']
        self._mean_pixel = config['mean_pixel']
        self._output_json_path = output_json_path
        self._config = config

    def load_model(self, latest_checkpoint=None):
        model = SSDModel(self._config)
        logger.info('Building model')
        if latest_checkpoint is None:
            weights_dir = os.path.join(self._model_dir, 'best_weights')
            latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
        model.load_weights(latest_checkpoint)
        self.model = model
        logger.info('Initialized model weights from {}'.format(latest_checkpoint))

    def _convert_to_coco_format(self, image_id, detections):
        coco_results = []
        coco_eval_dict = {
            'image_id': None,
            'category_id': None,
            'bbox': [],
            'score': None
        }
        boxes, train_ids, scores = [detections[x].numpy() for x in ['boxes', 'cls_ids', 'scores']]
        boxes = np.int32(boxes)
        boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        for box, train_id, score in zip(boxes, train_ids, scores):
            temp_dict = coco_eval_dict.copy()
            temp_dict['image_id'] = int(image_id)
            temp_dict['category_id'] = self.get_class_id(train_id=train_id)
            temp_dict['bbox'] = box.tolist()
            temp_dict['score'] = float(score)
            coco_results += [temp_dict]
        return coco_results

    def _pad_to_square_fn(self, image):
        dims = tf.shape(image)
        image_height = dims[0]
        image_width = dims[1]

        side = tf.maximum(image_height, image_width)
        offset_x = 0
        offset_y = 0
        padded_image = tf.image.pad_to_bounding_box(image, offset_y, offset_x, side, side)
        return padded_image, side

    def get_detections(self, sample):
        image = tf.image.decode_image(tf.io.read_file(self._image_dir + sample['image']), channels=3)
        image.set_shape([None, None, 3])
        image = tf.cast(image, dtype=tf.float32)
        padded_image, side = self._pad_to_square_fn(image)
        input_image = tf.image.resize(padded_image, [self._input_height, self._input_width])
        input_image = input_image[:, :, ::-1] - tf.constant(self._mean_pixel)
        input_image = tf.expand_dims(input_image, axis=0)
        detections = self.model.get_detections(input_image)
        detections['boxes'] = rescale_boxes(detections['boxes'],
                                            [self._input_height, self._input_width],
                                            [side, side])
        coco_result = self._convert_to_coco_format(sample['image_id'], detections)
        return image, detections, coco_result

    def run_inference(self):
        results = []
        for sample in tqdm(self.dataset['val']):
            image, detections, result = self.get_detections(sample)
            results += result
        self.results = results
        self.dump_results()

    def dump_results(self):
        with open(self._output_json_path, 'w') as f:
            json.dump(self.results, f, indent=4)

    def compute_statistics(self, predictions_path=None):
        if predictions_path is None:
            predictions_path = self._output_json_path
        coco_val_obj = self._ann['val']
        predictions = coco_val_obj.loadRes(predictions_path)
        annotation_type = 'bbox'
        cocoEval = COCOeval(coco_val_obj, predictions, annotation_type)

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
