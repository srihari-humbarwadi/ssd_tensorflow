from abc import ABC, abstractmethod


class Parser(ABC):
    def __init__(self, download_path):
        self._download_path = download_path
        self._data = {
            'train': [],
            'val': [],
            'test': []
        }
        self._classes = []
        self._class_name_to_class_id = {}
        self._class_id_to_class_name = {}
        self._class_id_to_train_id = {}
        self._train_id_to_class_id = {}

    def get_class_id(self, train_id=None, class_name=None):
        if train_id is not None:
            return self._train_id_to_class_id[train_id]
        return self._class_name_to_class_id[class_name]

    def get_train_id(self, class_id):
        return self._class_id_to_train_id[class_id]

    def get_class_name(self, train_id=None, class_id=None):
        if train_id is not None:
            class_id = self._train_id_to_class_id[train_id]
        return self._class_id_to_class_name[class_id]

    @abstractmethod
    def _build_dataset(self):
        pass

    @property
    def dataset(self):
        return self._data

    @property
    def classes(self):
        return self._classes
