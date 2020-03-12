import gc
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model, load_model

# https://github.com/tensorflow/tensorflow/issues/29161
# https://github.com/keras-team/keras/issues/10340
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())
K.set_session(session)

# https://www.tensorflow.org/api_docs/python/tf/compat/v1/disable_eager_execution
tf.compat.v1.disable_eager_execution()


def init_model(model_file_path):
    backbone_model = load_model(model_file_path,
                                custom_objects={
                                    "tf": tf,
                                    "swish": tf.nn.swish
                                },
                                compile=False)
    input_tensor = Input(shape=list(backbone_model.input_shape[1:-1]) + [1])
    output_tensor = Lambda(lambda x: K.repeat_elements(x, rep=3, axis=3),
                           name="repeat_elements")(input_tensor)
    preprocess_input_wrapper = lambda x: x / 255.0
    output_tensor = Lambda(preprocess_input_wrapper,
                           name="preprocess_input")(output_tensor)
    output_tensor_list = backbone_model(output_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor_list)
    return model


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

    # Add dummy dimensions
    image_content = np.expand_dims(image_content, axis=-1)
    image_content = np.expand_dims(image_content, axis=0)

    return image_content


def perform_inference(model_file_path_pattern="inference.h5",
                      use_manual_manipulation=False,
                      batch_size=64,
                      ensembling_option=np.mean):
    inference_start = time.time()

    # Initiation
    height = 137
    width = 236
    attribute_name_list = [
        "consonant_diacritic", "grapheme_root", "vowel_diacritic"
    ]

    # Paths of folders
    root_folder_path_list = [
        os.path.expanduser("~/Documents/Local Storage/Dataset"), "/kaggle/input"
    ]
    root_folder_path_mask = [
        os.path.isdir(path) for path in root_folder_path_list
    ]
    root_folder_path = root_folder_path_list[root_folder_path_mask.index(True)]
    dataset_folder_name = "bengaliai-cv19"
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)

    # Paths of files
    test_parquet_file_path_list = sorted(
        glob.glob(os.path.join(dataset_folder_path,
                               "test_image_data_*.parquet")))

    # Load models
    model_list = []
    model_file_path_list = sorted(glob.glob(model_file_path_pattern))
    assert len(model_file_path_list) > 0
    for model_file_path in model_file_path_list:
        print("Loading the model from {} ...".format(model_file_path))
        model = init_model(model_file_path)
        model_list.append(model)
    input_shape = model_list[0].input_shape[1:]

    # Process the test split
    concatenated_image_id_list, probability_array_dict = [], {}
    process_image_content_wrapper = lambda image_content: process_image_content(
        image_content, input_shape, use_manual_manipulation)
    for parquet_file_path in test_parquet_file_path_list:
        print("Processing {} ...".format(parquet_file_path))
        data_frame = pd.read_parquet(parquet_file_path)
        concatenated_image_id = data_frame.iloc[:, 0].values
        concatenated_image_data = 255 - data_frame.iloc[:, 1:].values.reshape(
            -1, height, width)
        del data_frame
        gc.collect()

        # Apply process_image_content
        concatenated_image_data = np.vstack((item for item in map(
            process_image_content_wrapper, concatenated_image_data)))

        # Generate predictions
        concatenated_image_id_list += concatenated_image_id.tolist()
        for index, model in enumerate(model_list):
            probability_array_list = model.predict(concatenated_image_data,
                                                   batch_size=batch_size)
            if index not in probability_array_dict:
                probability_array_dict[index] = []
            probability_array_dict[index].append(probability_array_list)
        del concatenated_image_data
        gc.collect()

    # Ensembling
    aggregation_function = lambda input_list: [
        np.vstack(input_tuple) for input_tuple in zip(*input_list)
    ]
    ensembling_function = lambda input_list: [
        ensembling_option(input_tuple, axis=0)
        for input_tuple in zip(*input_list)
    ]
    probability_array_list = ensembling_function([
        aggregation_function(value)
        for value in probability_array_dict.values()
    ])

    # Save probability arrays to disk
    print("Saving the submission file ...")
    row_id_list, target_list = [], []
    prediction_array_list = [
        np.argmax(probability_array, axis=1)
        for probability_array in probability_array_list
    ]
    for image_index, image_id in enumerate(concatenated_image_id_list):
        for attribute_name, prediction_array in zip(attribute_name_list,
                                                    prediction_array_list):
            row_id_list.append("_".join((image_id, attribute_name)))
            target_list.append(prediction_array[image_index])
    submission_data_frame = pd.DataFrame(
        {
            "row_id": row_id_list,
            "target": target_list
        },
        columns=["row_id", "target"])
    submission_data_frame.to_csv("submission.csv", index=False)

    inference_end = time.time()
    print("Inference took {:.2f} seconds.".format(inference_end -
                                                  inference_start))

    print("All done!")
