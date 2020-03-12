from importlib import import_module
from urllib.request import urlopen

import cv2
import numpy as np
from classification_models.tfkeras import Classifiers as QubvelClassifiers
from keras_applications import imagenet_utils
from tensorflow.python.keras.applications import keras_modules_injection


class KerasApplicationsWrapper(object):
    """ https://github.com/keras-team/keras-applications """

    def __init__(self):
        self._script_name_to_model_name_list_dict = {
            "vgg16": ["VGG16"],
            "vgg19": ["VGG19"],
            "inception_v3": ["InceptionV3"],
            "inception_resnet_v2": ["InceptionResNetV2"],
            "xception": ["Xception"],
            "mobilenet": ["MobileNet"],
            "mobilenet_v2": ["MobileNetV2"],
            "densenet": ["DenseNet121", "DenseNet169", "DenseNet201"],
            "nasnet": ["NASNetMobile", "NASNetLarge"],
            "resnet": ["ResNet50", "ResNet101", "ResNet152"],
            "resnet_v2": ["ResNet50V2", "ResNet101V2", "ResNet152V2"],
            "resnext": ["ResNeXt50", "ResNeXt101"],
            "efficientnet": [
                "EfficientNetB{}".format(index) for index in range(8)
            ]
        }

        self._model_name_to_script_name_dict = {}
        for script_name, model_name_list in self._script_name_to_model_name_list_dict.items(
        ):
            for model_name in model_name_list:
                self._model_name_to_script_name_dict[model_name] = script_name

    def get_model_name_list(self):
        return sorted(self._model_name_to_script_name_dict.keys())

    def query_by_model_name(self, model_name):
        script_name = self._model_name_to_script_name_dict.get(model_name, None)
        if script_name is None:
            return None
        module_name = "keras_applications.{}".format(script_name)
        print("Importing {} from {} ...".format(model_name, module_name))
        module = import_module(module_name)
        model_function = keras_modules_injection(getattr(module, model_name))
        preprocess_function = keras_modules_injection(
            getattr(module, "preprocess_input"))
        decode_function = keras_modules_injection(
            getattr(imagenet_utils, "decode_predictions"))
        return model_function, preprocess_function, decode_function


class LarqZooWrapper(object):
    """
        https://larq.dev/zoo/
        https://github.com/larq/zoo
    """

    def __init__(self):
        self._script_name_to_model_name_list_dict = {
            "binarynet": ["BinaryAlexNet"],
            "birealnet": ["BiRealNet"],
            "xnornet": ["XNORNet"],
            "resnet_e": ["BinaryResNetE18"],
            "densenet": [
                "BinaryDenseNet28", "BinaryDenseNet37",
                "BinaryDenseNet37Dilated", "BinaryDenseNet45"
            ],
            "dorefanet": ["DoReFaNet"]
        }

        self._model_name_to_script_name_dict = {}
        for script_name, model_name_list in self._script_name_to_model_name_list_dict.items(
        ):
            for model_name in model_name_list:
                self._model_name_to_script_name_dict[model_name] = script_name

    def get_model_name_list(self):
        return sorted(self._model_name_to_script_name_dict.keys())

    def query_by_model_name(self, model_name):
        script_name = self._model_name_to_script_name_dict.get(model_name, None)
        if script_name is None:
            return None
        module_name = "larq_zoo.literature.{}".format(script_name)
        print("Importing {} from {} ...".format(model_name, module_name))
        module = import_module(module_name)
        model_function = getattr(module, model_name)
        general_preprocess_function = keras_modules_injection(
            getattr(imagenet_utils, "preprocess_input"))
        preprocess_function = lambda x: general_preprocess_function(
            x, mode="torch")
        decode_function = getattr(import_module("larq_zoo.utils"),
                                  "decode_predictions")
        return model_function, preprocess_function, decode_function


class QubvelWrapper(object):
    """ https://github.com/qubvel/classification_models """

    def __init__(self):
        self._prefix = "qubvel"
        self._qubvel_classifiers = QubvelClassifiers

    def get_model_name_list(self):
        return [
            "{}_{}".format(self._prefix, model_name)
            for model_name in sorted(self._qubvel_classifiers.models_names())
        ]

    def query_by_model_name(self, model_name):
        if model_name not in self.get_model_name_list():
            return None
        model_function, preprocess_function = self._qubvel_classifiers.get(
            model_name[len(self._prefix) + 1:])
        decode_function = keras_modules_injection(
            getattr(imagenet_utils, "decode_predictions"))
        return model_function, preprocess_function, decode_function


class ExtraModelWrapper(object):

    def __init__(self):
        self._script_name_to_model_name_list_dict = {
            "customized_resnet": [
                "CustomizedResNet50", "CustomizedResNet101",
                "CustomizedResNet152", "CustomizedResNeXt50",
                "CustomizedResNeXt101"
            ]
        }

        self._model_name_to_script_name_dict = {}
        for script_name, model_name_list in self._script_name_to_model_name_list_dict.items(
        ):
            for model_name in model_name_list:
                self._model_name_to_script_name_dict[model_name] = script_name

    def get_model_name_list(self):
        return sorted(self._model_name_to_script_name_dict.keys())

    def query_by_model_name(self, model_name):
        script_name = self._model_name_to_script_name_dict.get(model_name, None)
        if script_name is None:
            return None
        if __name__ == "__main__":
            module_name = script_name
            package = None
        else:
            module_name = "..{}".format(script_name)
            package = __name__
        print("Importing {} from {} ...".format(model_name, module_name))
        module = import_module(module_name, package=package)
        model_function = keras_modules_injection(getattr(module, model_name))
        preprocess_function = keras_modules_injection(
            getattr(module, "preprocess_input"))
        decode_function = keras_modules_injection(
            getattr(module, "decode_predictions"))
        return model_function, preprocess_function, decode_function


class BackboneWrapper(object):

    def __init__(self):
        self._wrapper_class_list = [
            KerasApplicationsWrapper, LarqZooWrapper, QubvelWrapper,
            ExtraModelWrapper
        ]

        self._wrapper_instance_list = []
        for wrapper_class in self._wrapper_class_list:
            wrapper_instance = wrapper_class()
            self._wrapper_instance_list.append(wrapper_instance)

        self._preprocess_function = lambda x: x / 255.0

    def get_model_name_list(self):
        model_name_list = []
        for wrapper_instance in self._wrapper_instance_list:
            model_name_list += wrapper_instance.get_model_name_list()
        return sorted(model_name_list)

    def query_by_model_name(self, model_name):
        for wrapper_instance in self._wrapper_instance_list:
            query_result = wrapper_instance.query_by_model_name(model_name)
            if query_result is not None:
                query_result = list(query_result)
                query_result[1] = self._preprocess_function
                return query_result
        return None


def example(backbone_model_name="CustomizedResNet50"):
    print("Initiating the model ...")
    query_result = BackboneWrapper().query_by_model_name(backbone_model_name)
    assert query_result is not None, "Backbone {} is not supported.".format(
        backbone_model_name)
    model_instantiation, preprocess_input, decode_predictions = query_result
    model = model_instantiation(input_shape=(224, 224, 3),
                                weights="imagenet",
                                include_top=True)

    print("Loading the image content ...")
    raw_data = urlopen(
        url="https://avatars3.githubusercontent.com/u/15064790").read()
    raw_data = np.frombuffer(raw_data, np.uint8)
    image_content = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    image_content = cv2.resize(image_content, model.input_shape[1:3][::-1])
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    print("Generating the batch ...")
    image_content_list = [image_content]
    image_content_array = np.array(image_content_list, dtype=np.float32)
    image_content_array = preprocess_input(image_content_array)

    print("Generating predictions ...")
    predictions = model.predict(image_content_array)
    print("Predictions: {}".format(decode_predictions(predictions, top=3)[0]))

    print("All done!")


if __name__ == "__main__":
    example()
