from tensorflow.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50, ResNet101, ResNet152


class FeatureExtractors:
    def __init__(self):
        self.model_fns = {
            'resnet_152': FeatureExtractors._resnet_152,
            'resnet_101': FeatureExtractors._resnet_101,
            'resnet_50': FeatureExtractors._resnet_50,
        }

    @staticmethod
    def _resnet_50(input_shape):
        image = Input(shape=input_shape, name='image')
        _layer_names = [
            'conv3_block4_out',
            'conv4_block6_out',
            'conv5_block3_out'
        ]

        base_model = ResNet50(include_top=False, input_tensor=image)
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps

    @staticmethod
    def _resnet_101(input_shape):
        image = Input(shape=input_shape, name='image')
        _layer_names = [
            'conv3_block4_out',
            'conv4_block23_out',
            'conv5_block3_out'
        ]
        base_model = ResNet101(include_top=False, input_tensor=image)
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps

    @staticmethod
    def _resnet_152(input_shape):
        image = Input(shape=input_shape, name='image')
        _layer_names = [
            'conv3_block8_out',
            'conv4_block36_out',
            'conv5_block3_out'
        ]
        base_model = ResNet152(include_top=False, input_tensor=image)
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps
