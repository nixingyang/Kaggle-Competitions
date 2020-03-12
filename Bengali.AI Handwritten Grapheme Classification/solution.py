import os
import pickle
import shutil
import sys
from collections import OrderedDict
from datetime import datetime

import cv2
import larq as lq
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import (Callback, LearningRateScheduler,
                                               ModelCheckpoint)
from tensorflow.python.keras.layers import (BatchNormalization, Dense, Dropout,
                                            GlobalAveragePooling2D)
from tensorflow.python.keras.models import Model, model_from_json
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils import Sequence, plot_model

from backbone.backbone_wrapper import BackboneWrapper
from data_generator.load_dataset import load_Bengali
from image_augmentation import image_augmentors_wrapper
from image_augmentation.cutmix_and_mixup import perform_cutmix, perform_mixup

# Specify the backend of matplotlib
matplotlib.use("Agg")

# https://github.com/tensorflow/tensorflow/issues/29161
# https://github.com/keras-team/keras/issues/10340
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())
K.set_session(session)

flags = tf.compat.v1.app.flags
flags.DEFINE_string("dataset_name", "Bengali", "Name of the dataset.")
flags.DEFINE_string("backbone_model_name", "DenseNet121",
                    "Name of the backbone model."
                   )  # ["qubvel_seresnext50", "DenseNet121", "EfficientNetB0"]
flags.DEFINE_bool("freeze_backbone_model", False,
                  "Freeze layers in the backbone model.")
flags.DEFINE_integer("image_width", 240, "Width of the images.")
flags.DEFINE_integer("image_height", 144, "Height of the images.")
flags.DEFINE_bool("use_manual_manipulation", False, "Use manual manipulation.")
flags.DEFINE_bool("use_batchnormalization", False, "Use BatchNormalization.")
flags.DEFINE_float("dropout_rate", 0.2,
                   "Dropout rate before final classifier layer.")
flags.DEFINE_float("kernel_regularization_factor", 1e-5,
                   "Regularization factor of kernel.")
flags.DEFINE_float("bias_regularization_factor", 0,
                   "Regularization factor of bias.")
flags.DEFINE_float("gamma_regularization_factor", 0,
                   "Regularization factor of gamma.")
flags.DEFINE_float("beta_regularization_factor", 0,
                   "Regularization factor of beta.")
flags.DEFINE_integer("fold_num", 1, "Number of folds.")
flags.DEFINE_integer("fold_index", 0, "Index of fold.")
flags.DEFINE_integer(
    "evaluate_validation_every_N_epochs", 10,
    "Evaluate the performance on validation samples every N epochs.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_string("learning_rate_mode", "default",
                    "Mode of the learning rate scheduler."
                   )  # ["constant", "linear", "cosine", "warmup", "default"]
flags.DEFINE_float("learning_rate_start", 5e-4, "Starting learning rate.")
flags.DEFINE_float("learning_rate_end", 5e-4, "Ending learning rate.")
flags.DEFINE_float("learning_rate_base", 5e-4, "Base learning rate.")
flags.DEFINE_integer("learning_rate_warmup_epochs", 10,
                     "Number of epochs to warmup the learning rate.")
flags.DEFINE_integer("learning_rate_steady_epochs", 5,
                     "Number of epochs to keep the learning rate steady.")
flags.DEFINE_float("learning_rate_drop_factor", 0,
                   "Factor to decrease the learning rate.")
flags.DEFINE_float("learning_rate_lower_bound", 1e-5,
                   "Lower bound of the learning rate.")
flags.DEFINE_integer("steps_per_epoch", 5000, "Number of steps per epoch.")
flags.DEFINE_integer("epoch_num", 100, "Number of epochs.")
flags.DEFINE_integer("workers", 5,
                     "Number of processes to spin up for data generator.")
flags.DEFINE_float("cutmix_probability", 0.8, "Probability of using cutmix.")
flags.DEFINE_float("mixup_probability", 0, "Probability of using mixup.")
flags.DEFINE_string("image_augmentor_name", "GridMaskImageAugmentor",
                    "Name of image augmentor.")
# ["BaseImageAugmentor", "AugMixImageAugmentor", "AutoAugmentImageAugmentor",
# "GridMaskImageAugmentor", "RandomErasingImageAugmentor"]
flags.DEFINE_bool("use_data_augmentation_in_training", True,
                  "Use data augmentation in training.")
flags.DEFINE_bool("use_data_augmentation_in_evaluation", False,
                  "Use data augmentation in evaluation.")
flags.DEFINE_bool("use_label_smoothing_in_training", True,
                  "Use label smoothing in training.")
flags.DEFINE_bool("use_label_smoothing_in_evaluation", False,
                  "Use label smoothing in evaluation.")
flags.DEFINE_bool("evaluation_only", False, "Only perform evaluation.")
flags.DEFINE_string("pretrained_model_file_path", "",
                    "File path of the pretrained model.")
flags.DEFINE_string(
    "output_folder_path",
    os.path.abspath(
        os.path.join(__file__, "../output_{}".format(
            datetime.now().strftime("%Y_%m_%d")))),
    "Path to directory to output files.")
FLAGS = flags.FLAGS


def apply_cross_validation(y, fold_num, fold_index, random_state=0):
    cross_validator = StratifiedKFold(n_splits=fold_num,
                                      shuffle=False,
                                      random_state=random_state)
    for index, result in enumerate(
            cross_validator.split(X=np.zeros(len(y)), y=y)):
        if index == fold_index:
            return result

    assert False, "Invalid arguments: fold_num {}, fold_index {}.".format(
        fold_num, fold_index)


def init_model(backbone_model_name, freeze_backbone_model, input_shape,
               attribute_name_to_label_encoder_dict, use_batchnormalization,
               dropout_rate, kernel_regularization_factor,
               bias_regularization_factor, gamma_regularization_factor,
               beta_regularization_factor, evaluation_only,
               pretrained_model_file_path):

    def _add_regularizers(model, kernel_regularization_factor,
                          bias_regularization_factor,
                          gamma_regularization_factor,
                          beta_regularization_factor):
        update_model_required = False
        for layer in model.layers:
            if not np.isclose(kernel_regularization_factor, 0.0) and hasattr(
                    layer, "kernel_regularizer"):
                update_model_required = True
                layer.kernel_regularizer = l2(kernel_regularization_factor)
            if not np.isclose(bias_regularization_factor, 0.0) and hasattr(
                    layer, "bias_regularizer"):
                if layer.use_bias:
                    update_model_required = True
                    layer.bias_regularizer = l2(bias_regularization_factor)
            if not np.isclose(gamma_regularization_factor, 0.0) and hasattr(
                    layer, "gamma_regularizer"):
                if layer.scale:
                    update_model_required = True
                    layer.gamma_regularizer = l2(gamma_regularization_factor)
            if not np.isclose(beta_regularization_factor, 0.0) and hasattr(
                    layer, "beta_regularizer"):
                if layer.center:
                    update_model_required = True
                    layer.beta_regularizer = l2(beta_regularization_factor)
        if update_model_required:
            print("Adding regularizers ...")
            # https://github.com/keras-team/keras/issues/2717#issuecomment-447570737
            vanilla_weights = model.get_weights()
            model = model_from_json(model.to_json(),
                                    custom_objects={
                                        "tf": tf,
                                        "swish": tf.nn.swish
                                    })
            model.set_weights(vanilla_weights)
        return model

    # Initiate the backbone model
    query_result = BackboneWrapper().query_by_model_name(backbone_model_name)
    assert query_result is not None, "Backbone {} is not supported.".format(
        backbone_model_name)
    model_instantiation, preprocess_input, _ = query_result
    backbone_model_weights = None if len(
        pretrained_model_file_path) > 0 else "imagenet"
    backbone_model = model_instantiation(input_shape=input_shape,
                                         weights=backbone_model_weights,
                                         include_top=False)
    if freeze_backbone_model:
        for layer in backbone_model.layers:
            layer.trainable = False

    # Add GlobalAveragePooling2D
    global_average_pooling_tensor = GlobalAveragePooling2D()(
        backbone_model.output)

    # https://arxiv.org/pdf/1801.07698v1.pdf Section 3.2.2 Output setting
    # https://arxiv.org/pdf/1807.11042.pdf
    classification_embedding_tensor = global_average_pooling_tensor
    if use_batchnormalization:
        classification_embedding_tensor = BatchNormalization()(
            classification_embedding_tensor)
    if dropout_rate > 0:
        classification_embedding_tensor = Dropout(
            rate=dropout_rate)(classification_embedding_tensor)

    # Add categorical crossentropy loss
    classification_output_tensor_list = []
    for attribute_name, label_encoder in attribute_name_to_label_encoder_dict.items(
    ):
        classification_output_tensor = Dense(
            units=len(label_encoder.classes_),
            activation="softmax",
            name="{}_classification_output".format(attribute_name))(
                classification_embedding_tensor)
        classification_output_tensor_list.append(classification_output_tensor)
    classification_loss_function_list = [
        "categorical_crossentropy"
    ] * len(classification_output_tensor_list)

    # Define the model
    model = Model(inputs=[backbone_model.input],
                  outputs=classification_output_tensor_list)
    model = _add_regularizers(model, kernel_regularization_factor,
                              bias_regularization_factor,
                              gamma_regularization_factor,
                              beta_regularization_factor)
    if evaluation_only:
        print("Freezing the whole model in the evaluation_only mode ...")
        model.trainable = False

    # Compile the model
    extra_attributes_num = len(attribute_name_to_label_encoder_dict) - 1
    loss_weights = [1.0] + (np.ones(extra_attributes_num) /
                            extra_attributes_num).tolist()
    model.compile(optimizer=Adam(),
                  loss=classification_loss_function_list,
                  loss_weights=loss_weights,
                  metrics={
                      "grapheme_classification_output": ["accuracy"],
                      "consonant_diacritic_classification_output": ["accuracy"],
                      "grapheme_root_classification_output": ["accuracy"],
                      "vowel_diacritic_classification_output": ["accuracy"]
                  })

    # Print the summary of the model
    print("Summary of model:")
    model.summary()
    lq.models.summary(model)

    # Load weights from the pretrained model
    if len(pretrained_model_file_path) > 0:
        assert os.path.isfile(pretrained_model_file_path)
        print("Loading weights from {} ...".format(pretrained_model_file_path))
        model.load_weights(pretrained_model_file_path)

    return model, preprocess_input


def process_image_content(image_content,
                          input_shape,
                          use_manual_manipulation,
                          intensity_threshold_percentage=0.2,
                          edge_threshold=5):
    if use_manual_manipulation:
        # Cropping
        intensity_threshold = np.uint8(
            np.max(image_content) * intensity_threshold_percentage)
        width_mask = np.sum(
            image_content[edge_threshold:-edge_threshold, edge_threshold:
                          -edge_threshold] > intensity_threshold,
            axis=0) > 1
        height_mask = np.sum(
            image_content[edge_threshold:-edge_threshold, edge_threshold:
                          -edge_threshold] > intensity_threshold,
            axis=1) > 1
        width_start, width_end = np.where(width_mask)[0][[0, -1]]
        width_start, width_end = max(
            0, width_start - edge_threshold * 2), width_end + edge_threshold * 2
        height_start, height_end = np.where(height_mask)[0][[0, -1]]
        height_start, height_end = max(
            0,
            height_start - edge_threshold * 2), height_end + edge_threshold * 2
        image_content = image_content[height_start:height_end, width_start:
                                      width_end]

        # Apply zero padding to make it square
        height, width = image_content.shape
        max_length = np.max(image_content.shape)
        height_pad = (max_length - height) // 2
        width_pad = (max_length - width) // 2
        image_content = np.pad(image_content, ((height_pad,), (width_pad,)),
                               mode="constant",
                               constant_values=0)

    # Resize the image
    image_content = cv2.resize(image_content, input_shape[:2][::-1])

    # Normalization
    min_intensity, max_intensity = np.min(image_content), np.max(image_content)
    image_content = ((image_content.astype(np.float32) - min_intensity) /
                     (max_intensity - min_intensity) * 255).astype(np.uint8)

    # Expand and repeat
    image_content = np.expand_dims(image_content, axis=-1)
    image_content = np.repeat(image_content, repeats=3, axis=2)

    return image_content


def read_image_file(image_file_path, input_shape, use_manual_manipulation):
    # Read image file
    image_content = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)

    # Process image content
    image_content = process_image_content(image_content, input_shape,
                                          use_manual_manipulation)

    return image_content


def apply_label_smoothing(y_true, epsilon=0.1):
    # https://github.com/keras-team/keras/pull/4723
    # https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks/blob/master/tools/loss.py#L6
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf
    y_true = (1 - epsilon) * y_true + epsilon / y_true.shape[1]
    return y_true


class VanillaDataSequence(Sequence):

    def __init__(self, accumulated_info_dataframe,
                 attribute_name_to_label_encoder_dict, input_shape,
                 use_manual_manipulation, batch_size, steps_per_epoch):
        super(VanillaDataSequence, self).__init__()

        # Save as variables
        self.accumulated_info_dataframe, self.attribute_name_to_label_encoder_dict = accumulated_info_dataframe, attribute_name_to_label_encoder_dict
        self.input_shape, self.use_manual_manipulation = input_shape, use_manual_manipulation
        self.batch_size, self.steps_per_epoch = batch_size, steps_per_epoch

        # Initiation
        image_num_per_epoch = batch_size * steps_per_epoch
        self.sample_index_list_generator = self._get_sample_index_list_generator(
            sample_num=len(self.accumulated_info_dataframe),
            batch_size=batch_size,
            image_num_per_epoch=image_num_per_epoch)
        self.sample_index_list = next(self.sample_index_list_generator)

    def _get_sample_index_in_batches_generator(self, sample_num, batch_size):
        sample_index_in_batches = []
        sample_index_array = np.arange(sample_num)
        while True:
            np.random.shuffle(sample_index_array)
            for sample_index in sample_index_array:
                sample_index_in_batches.append(sample_index)
                if len(sample_index_in_batches) == batch_size:
                    yield sample_index_in_batches
                    sample_index_in_batches = []

    def _get_sample_index_list_generator(self, sample_num, batch_size,
                                         image_num_per_epoch):
        sample_index_list = []
        sample_index_in_batches_generator = self._get_sample_index_in_batches_generator(
            sample_num, batch_size)
        for sample_index_in_batches in sample_index_in_batches_generator:
            sample_index_list += sample_index_in_batches
            if len(sample_index_list) == image_num_per_epoch:
                yield sample_index_list
                sample_index_list = []

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        image_content_list, attribute_name_to_one_hot_encoding_list_dict = [], OrderedDict(
            {})
        sample_index_list = self.sample_index_list[index *
                                                   self.batch_size:(index + 1) *
                                                   self.batch_size]
        for sample_index in sample_index_list:
            # Get current sample from accumulated_info_dataframe
            accumulated_info = self.accumulated_info_dataframe.iloc[
                sample_index]

            # Read image
            image_file_path = accumulated_info["image_file_path"]
            image_content = read_image_file(image_file_path, self.input_shape,
                                            self.use_manual_manipulation)
            image_content_list.append(image_content)

            # Load annotations
            for attribute_name, label_encoder in self.attribute_name_to_label_encoder_dict.items(
            ):
                # Get the one hot encoding vector
                attribute_value = accumulated_info[attribute_name]
                one_hot_encoding = np.zeros(len(label_encoder.classes_))
                one_hot_encoding[label_encoder.transform([attribute_value
                                                         ])[0]] = 1

                # Append one_hot_encoding
                if attribute_name not in attribute_name_to_one_hot_encoding_list_dict:
                    attribute_name_to_one_hot_encoding_list_dict[
                        attribute_name] = []
                attribute_name_to_one_hot_encoding_list_dict[
                    attribute_name].append(one_hot_encoding)
        assert len(image_content_list) == self.batch_size

        # Construct image_content_array
        image_content_array = np.array(image_content_list, dtype=np.float32)

        # Construct one_hot_encoding_array_list
        one_hot_encoding_array_list = []
        for one_hot_encoding_list in attribute_name_to_one_hot_encoding_list_dict.values(
        ):
            one_hot_encoding_array = np.array(one_hot_encoding_list)
            one_hot_encoding_array_list.append(one_hot_encoding_array)

        return image_content_array, one_hot_encoding_array_list

    def on_epoch_end(self):
        self.sample_index_list = next(self.sample_index_list_generator)


class CutMixAndMixUpDataSequence(Sequence):

    def __init__(self, datasequence_instance_alpha, datasequence_instance_beta,
                 cutmix_probability, mixup_probability):
        super(CutMixAndMixUpDataSequence, self).__init__()

        # Sanity check
        assert len(datasequence_instance_alpha) == len(
            datasequence_instance_beta)
        assert cutmix_probability + mixup_probability <= 1

        # Save as variables
        self.datasequence_instance_alpha = datasequence_instance_alpha
        self.datasequence_instance_beta = datasequence_instance_beta
        self.cutmix_probability, self.mixup_probability = cutmix_probability, mixup_probability

    def __len__(self):
        return len(self.datasequence_instance_alpha)

    def __getitem__(self, index):
        image_content_array_alpha, one_hot_encoding_array_list_alpha = self.datasequence_instance_alpha[
            index]
        image_content_array_beta, one_hot_encoding_array_list_beta = self.datasequence_instance_beta[
            index]
        probability_array = np.random.uniform(
            size=len(image_content_array_alpha))

        image_content_list, one_hot_encoding_tuple_list = [], []
        for image_content_alpha, one_hot_encoding_tuple_alpha, image_content_beta, one_hot_encoding_tuple_beta, probability in zip(
                image_content_array_alpha,
                zip(*one_hot_encoding_array_list_alpha),
                image_content_array_beta,
                zip(*one_hot_encoding_array_list_beta), probability_array):

            image_content, one_hot_encoding_tuple = image_content_alpha, one_hot_encoding_tuple_alpha
            if probability < self.cutmix_probability:
                image_content, one_hot_encoding_tuple = perform_cutmix(
                    image_content_alpha, one_hot_encoding_tuple_alpha,
                    image_content_beta, one_hot_encoding_tuple_beta)
            elif probability < self.cutmix_probability + self.mixup_probability:
                image_content, one_hot_encoding_tuple = perform_mixup(
                    image_content_alpha, one_hot_encoding_tuple_alpha,
                    image_content_beta, one_hot_encoding_tuple_beta)

            image_content_list.append(image_content)
            one_hot_encoding_tuple_list.append(one_hot_encoding_tuple)

        # Construct image_content_array
        image_content_array = np.array(image_content_list, dtype=np.float32)

        # Construct one_hot_encoding_array_list
        one_hot_encoding_array_list = [
            np.array(item) for item in zip(*one_hot_encoding_tuple_list)
        ]

        return image_content_array, one_hot_encoding_array_list

    def on_epoch_end(self):
        self.datasequence_instance_alpha.on_epoch_end()
        self.datasequence_instance_beta.on_epoch_end()


class PreprocessingDataSequence(Sequence):

    def __init__(self, datasequence_instance, preprocess_input, image_augmentor,
                 use_data_augmentation, use_label_smoothing):
        super(PreprocessingDataSequence, self).__init__()

        # Save as variables
        self.datasequence_instance = datasequence_instance
        self.preprocess_input = preprocess_input
        self.image_augmentor, self.use_data_augmentation = image_augmentor, use_data_augmentation
        self.use_label_smoothing = use_label_smoothing

    def __len__(self):
        return len(self.datasequence_instance)

    def __getitem__(self, index):
        image_content_array, one_hot_encoding_array_list = self.datasequence_instance[
            index]

        if self.use_data_augmentation:
            # Apply data augmentation
            image_content_array = self.image_augmentor.apply_augmentation(
                image_content_array)

        # Apply preprocess_input function
        image_content_array = self.preprocess_input(image_content_array)

        if self.use_label_smoothing:
            # Apply label smoothing
            one_hot_encoding_array_list = [
                apply_label_smoothing(one_hot_encoding_array)
                for one_hot_encoding_array in one_hot_encoding_array_list
            ]

        return image_content_array, one_hot_encoding_array_list

    def on_epoch_end(self):
        self.datasequence_instance.on_epoch_end()


def learning_rate_scheduler(epoch_index, epoch_num, learning_rate_mode,
                            learning_rate_start, learning_rate_end,
                            learning_rate_base, learning_rate_warmup_epochs,
                            learning_rate_steady_epochs,
                            learning_rate_drop_factor,
                            learning_rate_lower_bound):
    learning_rate = None
    if learning_rate_mode == "constant":
        assert learning_rate_start == learning_rate_end, "starting and ending learning rates should be equal!"
        learning_rate = learning_rate_start
    elif learning_rate_mode == "linear":
        learning_rate = (learning_rate_end - learning_rate_start) / (
            epoch_num - 1) * epoch_index + learning_rate_start
    elif learning_rate_mode == "cosine":
        assert learning_rate_start > learning_rate_end, "starting learning rate should be higher than ending learning rate!"
        learning_rate = (learning_rate_start - learning_rate_end) / 2 * np.cos(
            np.pi * epoch_index /
            (epoch_num - 1)) + (learning_rate_start + learning_rate_end) / 2
    elif learning_rate_mode == "warmup":
        learning_rate = (learning_rate_end - learning_rate_start) / (
            learning_rate_warmup_epochs - 1) * epoch_index + learning_rate_start
        learning_rate = np.min((learning_rate, learning_rate_end))
    elif learning_rate_mode == "default":
        if epoch_index == 0:
            learning_rate = learning_rate_lower_bound
        elif epoch_index < learning_rate_warmup_epochs:
            learning_rate = learning_rate_base * (
                epoch_index + 1) / learning_rate_warmup_epochs
        else:
            if learning_rate_drop_factor == 0:
                learning_rate_drop_factor = np.exp(
                    learning_rate_steady_epochs /
                    (epoch_num - learning_rate_warmup_epochs * 2) *
                    np.log(learning_rate_base / learning_rate_lower_bound))
            learning_rate = learning_rate_base / np.power(
                learning_rate_drop_factor,
                int((epoch_index - learning_rate_warmup_epochs) /
                    learning_rate_steady_epochs))
    else:
        assert False, "{} is an invalid argument!".format(learning_rate_mode)
    learning_rate = np.max((learning_rate, learning_rate_lower_bound))
    return learning_rate


class HistoryLogger(Callback):

    def __init__(self, output_folder_path):
        super(HistoryLogger, self).__init__()

        self.accumulated_logs_dict = {}
        self.output_folder_path = output_folder_path

    def visualize(self, loss_name):
        # Unpack the values
        epoch_to_loss_value_dict = self.accumulated_logs_dict[loss_name]
        epoch_list = sorted(epoch_to_loss_value_dict.keys())
        loss_value_list = [
            epoch_to_loss_value_dict[epoch] for epoch in epoch_list
        ]
        epoch_list = (np.array(epoch_list) + 1).tolist()

        # Save the figure to disk
        figure = plt.figure()
        if isinstance(loss_value_list[0], dict):
            for metric_name in loss_value_list[0].keys():
                metric_value_list = [
                    loss_value[metric_name] for loss_value in loss_value_list
                ]
                print("{} {} {:.6f}".format(loss_name, metric_name,
                                            metric_value_list[-1]))
                plt.plot(epoch_list,
                         metric_value_list,
                         label="{} {:.6f}".format(metric_name,
                                                  metric_value_list[-1]))
        else:
            print("{} {:.6f}".format(loss_name, loss_value_list[-1]))
            plt.plot(epoch_list,
                     loss_value_list,
                     label="{} {:.6f}".format(loss_name, loss_value_list[-1]))
            plt.ylabel(loss_name)
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(
            os.path.join(self.output_folder_path, "{}.png".format(loss_name)))
        plt.close(figure)

    def on_epoch_end(self, epoch, logs=None):
        # Visualize each figure
        for loss_name, loss_value in logs.items():
            if loss_name not in self.accumulated_logs_dict:
                self.accumulated_logs_dict[loss_name] = {}
            self.accumulated_logs_dict[loss_name][epoch] = loss_value
            self.visualize(loss_name)

        # Save the accumulated_logs_dict to disk
        with open(
                os.path.join(self.output_folder_path,
                             "accumulated_logs_dict.pkl"), "wb") as file_object:
            pickle.dump(self.accumulated_logs_dict, file_object,
                        pickle.HIGHEST_PROTOCOL)


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    dataset_name = FLAGS.dataset_name
    backbone_model_name, freeze_backbone_model = FLAGS.backbone_model_name, FLAGS.freeze_backbone_model
    image_height, image_width = FLAGS.image_height, FLAGS.image_width
    input_shape = (image_height, image_width, 3)
    use_manual_manipulation = FLAGS.use_manual_manipulation
    use_batchnormalization, dropout_rate = FLAGS.use_batchnormalization, FLAGS.dropout_rate
    kernel_regularization_factor = FLAGS.kernel_regularization_factor
    bias_regularization_factor = FLAGS.bias_regularization_factor
    gamma_regularization_factor = FLAGS.gamma_regularization_factor
    beta_regularization_factor = FLAGS.beta_regularization_factor
    fold_num, fold_index = FLAGS.fold_num, FLAGS.fold_index
    use_validation = fold_num >= 2
    evaluate_validation_every_N_epochs = FLAGS.evaluate_validation_every_N_epochs
    batch_size = FLAGS.batch_size
    learning_rate_mode, learning_rate_start, learning_rate_end = FLAGS.learning_rate_mode, FLAGS.learning_rate_start, FLAGS.learning_rate_end
    learning_rate_base, learning_rate_warmup_epochs, learning_rate_steady_epochs = FLAGS.learning_rate_base, FLAGS.learning_rate_warmup_epochs, FLAGS.learning_rate_steady_epochs
    learning_rate_drop_factor, learning_rate_lower_bound = FLAGS.learning_rate_drop_factor, FLAGS.learning_rate_lower_bound
    steps_per_epoch = FLAGS.steps_per_epoch
    epoch_num = FLAGS.epoch_num
    workers = FLAGS.workers
    use_multiprocessing = workers > 1
    cutmix_probability, mixup_probability = FLAGS.cutmix_probability, FLAGS.mixup_probability
    image_augmentor_name = FLAGS.image_augmentor_name
    use_data_augmentation_in_training, use_data_augmentation_in_evaluation = FLAGS.use_data_augmentation_in_training, FLAGS.use_data_augmentation_in_evaluation
    use_label_smoothing_in_training, use_label_smoothing_in_evaluation = FLAGS.use_label_smoothing_in_training, FLAGS.use_label_smoothing_in_evaluation
    evaluation_only = FLAGS.evaluation_only
    pretrained_model_file_path = FLAGS.pretrained_model_file_path

    output_folder_path = os.path.join(
        FLAGS.output_folder_path,
        "{}_{}x{}_{}_{}".format(backbone_model_name, input_shape[0],
                                input_shape[1], fold_num, fold_index))
    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Loading the annotations of the {} dataset ...".format(dataset_name))
    train_and_valid_accumulated_info_dataframe, train_and_valid_attribute_name_to_label_encoder_dict = load_Bengali(
    )

    if use_validation:
        print("Using customized cross validation splits ...")
        train_and_valid_grapheme_array = train_and_valid_accumulated_info_dataframe[
            "grapheme"].values
        train_indexes, valid_indexes = apply_cross_validation(
            y=train_and_valid_grapheme_array,
            fold_num=fold_num,
            fold_index=fold_index)
        train_accumulated_info_dataframe = train_and_valid_accumulated_info_dataframe.iloc[
            train_indexes]
        valid_accumulated_info_dataframe = train_and_valid_accumulated_info_dataframe.iloc[
            valid_indexes]
    else:
        train_accumulated_info_dataframe = train_and_valid_accumulated_info_dataframe
        valid_accumulated_info_dataframe = None

    print("Initiating the model ...")
    model, preprocess_input = init_model(
        backbone_model_name, freeze_backbone_model, input_shape,
        train_and_valid_attribute_name_to_label_encoder_dict,
        use_batchnormalization, dropout_rate, kernel_regularization_factor,
        bias_regularization_factor, gamma_regularization_factor,
        beta_regularization_factor, evaluation_only, pretrained_model_file_path)
    try:
        plot_model(model,
                   show_shapes=True,
                   show_layer_names=True,
                   to_file=os.path.join(output_folder_path, "model.png"))
    except Exception as exception:  # pylint: disable=broad-except
        print(exception)

    print("Initiating the image augmentor {} ...".format(image_augmentor_name))
    image_augmentor = getattr(image_augmentors_wrapper, image_augmentor_name)()
    image_augmentor.compose_transforms()

    print("Perform training ...")
    train_generator_alpha = VanillaDataSequence(
        train_accumulated_info_dataframe,
        train_and_valid_attribute_name_to_label_encoder_dict, input_shape,
        use_manual_manipulation, batch_size, steps_per_epoch)
    train_generator_beta = VanillaDataSequence(
        train_accumulated_info_dataframe,
        train_and_valid_attribute_name_to_label_encoder_dict, input_shape,
        use_manual_manipulation, batch_size, steps_per_epoch)
    train_generator = CutMixAndMixUpDataSequence(
        datasequence_instance_alpha=train_generator_alpha,
        datasequence_instance_beta=train_generator_beta,
        cutmix_probability=cutmix_probability,
        mixup_probability=mixup_probability)
    train_generator = PreprocessingDataSequence(
        train_generator,
        preprocess_input,
        image_augmentor,
        use_data_augmentation_in_training,
        use_label_smoothing_in_training,
    )
    optimal_model_file_path = os.path.join(output_folder_path, "model.h5")
    valid_generator = None
    if use_validation:
        valid_generator = VanillaDataSequence(
            valid_accumulated_info_dataframe,
            train_and_valid_attribute_name_to_label_encoder_dict, input_shape,
            use_manual_manipulation, batch_size,
            len(valid_accumulated_info_dataframe) // batch_size)
        valid_generator = PreprocessingDataSequence(
            valid_generator,
            preprocess_input,
            image_augmentor,
            use_data_augmentation_in_evaluation,
            use_label_smoothing_in_evaluation,
        )
    modelcheckpoint_callback = ModelCheckpoint(
        filepath=optimal_model_file_path,
        save_best_only=False,
        save_weights_only=False,
        period=evaluate_validation_every_N_epochs,
        verbose=1)
    learningratescheduler_callback = LearningRateScheduler(
        schedule=lambda epoch_index: learning_rate_scheduler(
            epoch_index, epoch_num, learning_rate_mode, learning_rate_start,
            learning_rate_end, learning_rate_base, learning_rate_warmup_epochs,
            learning_rate_steady_epochs, learning_rate_drop_factor,
            learning_rate_lower_bound),
        verbose=1)
    historylogger_callback = HistoryLogger(output_folder_path)
    if evaluation_only:
        model.fit(x=train_generator,
                  steps_per_epoch=1,
                  validation_data=valid_generator,
                  validation_freq=evaluate_validation_every_N_epochs,
                  callbacks=[historylogger_callback],
                  epochs=1,
                  workers=workers,
                  use_multiprocessing=use_multiprocessing,
                  verbose=2)
    else:
        model.fit(x=train_generator,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=valid_generator,
                  validation_freq=evaluate_validation_every_N_epochs,
                  callbacks=[
                      modelcheckpoint_callback, learningratescheduler_callback,
                      historylogger_callback
                  ],
                  epochs=epoch_num,
                  workers=workers,
                  use_multiprocessing=use_multiprocessing,
                  verbose=2)

    print("All done!")


if __name__ == "__main__":
    tf.compat.v1.app.run()
