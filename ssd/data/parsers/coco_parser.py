import os

import numpy as np
from pycocotools.coco import COCO

from ssd.data.parsers.base_parser import Parser


class CocoParser(Parser):
    def __init__(self, download_path, only_mappings=False, only_val=False):
        super(CocoParser, self).__init__(download_path)
        self._only_mappings = only_mappings
        self._only_val = only_val
        if not only_val:
            self._train_annotations_path = os.path.join(download_path, 'annotations/instances_train2017.json')
        self._val_annotations_path = os.path.join(download_path, 'annotations/instances_val2017.json')
        self._ann = {}
        self._build_dataset()

    def _build_dataset(self):
        def _convert_box_format(boxes):
            boxes = np.array(boxes)
            return np.concatenate([
                boxes[:, :2],
                boxes[:, :2] + boxes[:, 2:]
            ], axis=-1)

        def _build(annptations_path, split_name):
            print('Building {} data'.format(split_name))
            coco = COCO(annptations_path)
            if self._class_id_to_class_name == {}:
                self._class_id_to_class_name = {cat_dict['id']: cat_dict['name'] for _, cat_dict in coco.cats.items()}
            if self._class_name_to_class_id == {}:
                self._class_name_to_class_id = {cat_dict['name']: cat_dict['id'] for _, cat_dict in coco.cats.items()}

            self._classes = sorted(self._class_name_to_class_id.keys())

            if self._class_id_to_train_id == {}:
                self._class_id_to_train_id = {
                    self._class_name_to_class_id[class_name]: idx for idx, class_name in enumerate(self._classes)}
            if self._train_id_to_class_id == {}:
                self._train_id_to_class_id = {
                    idx: self._class_name_to_class_id[class_name] for idx, class_name in enumerate(self._classes)}

            if self._only_mappings:
                return
            
            for image_id, annotation in coco.imgToAnns.items():
                image_path = coco.imgs[image_id]['file_name']
                boxes = []
                classes = []
                for obj in annotation:

                    boxes.append(obj['bbox'])
                    classes.append(self.get_train_id(obj['category_id']))

                sample = {
                    'image': image_path,
                    'image_id': image_id,
                    'label': {
                        'boxes': _convert_box_format(boxes),
                        'classes': classes
                    }
                }
                self._data[split_name].append(sample)
                self._ann[split_name] = coco
        if not self._only_val:
            _build(self._train_annotations_path, 'train')
        _build(self._val_annotations_path, 'val')
