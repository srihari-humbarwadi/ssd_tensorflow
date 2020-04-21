from tensorflow.python.keras.applications.efficientnet import (EfficientNetB0,
                                                               EfficientNetB3,
                                                               EfficientNetB7)

from tensorflow.python.keras.applications.resnet_v2 import (ResNet50V2,
                                                            ResNet101V2)


class FeatureExtractors:
    def __init__(self):
        self.model_fns = {
            'efficient_net_b0': FeatureExtractors._efficient_net_b0,
            'efficient_net_b3': FeatureExtractors._efficient_net_b3,
            'efficient_net_b7': FeatureExtractors._efficient_net_b7,
            'resnet_101_v2': FeatureExtractors._resnet_101_v2,
            'resnet_50_v2': FeatureExtractors._resnet_50_v2,
        }

    @staticmethod
    def _efficient_net_b0():
        _layer_names = [
            'block4a_expand_activation',
            'block6a_expand_activation',
            'top_activation'
        ]
        base_model = EfficientNetB0(include_top=False, input_shape=[None, None, 3])
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps

    @staticmethod
    def _efficient_net_b3():
        _layer_names = [
            'block4a_expand_activation',
            'block6a_expand_activation',
            'top_activation'
        ]
        base_model = EfficientNetB3(include_top=False, input_shape=[None, None, 3])
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps

    @staticmethod
    def _efficient_net_b7():
        _layer_names = [
            'block4a_expand_activation',
            'block6a_expand_activation',
            'top_activation'
        ]
        base_model = EfficientNetB7(include_top=False, input_shape=[None, None, 3])
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps

    @staticmethod
    def _resnet_50_v2():
        _layer_names = [
            'conv3_block4_1_relu',
            'conv4_block6_1_relu',
            'post_relu',
        ]

        base_model = ResNet50V2(include_top=False, input_shape=[None, None, 3])
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps

    @staticmethod
    def _resnet_101_v2():
        _layer_names = [
            'conv3_block4_1_relu',
            'conv4_block23_1_relu',
            'post_relu'
        ]
        base_model = ResNet101V2(include_top=False, input_shape=[None, None, 3])
        feature_maps = [base_model.get_layer(layer_name).output for layer_name in _layer_names]
        return base_model, feature_maps
