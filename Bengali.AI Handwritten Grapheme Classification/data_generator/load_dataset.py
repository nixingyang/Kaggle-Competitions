import gc
import glob
import os
from collections import OrderedDict

import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _convert_split(converted_folder_path,
                   parquet_file_path_list,
                   height=137,
                   width=236):
    if os.path.isdir(converted_folder_path):
        return

    os.makedirs(converted_folder_path)
    for parquet_file_path in parquet_file_path_list:
        print("Converting {} ...".format(parquet_file_path))
        data_frame = pd.read_parquet(parquet_file_path)
        concatenated_image_id = data_frame.iloc[:, 0]
        concatenated_image_data = 255 - data_frame.iloc[:, 1:].values.reshape(
            -1, height, width)
        del data_frame
        gc.collect()

        for image_id, image_data in zip(concatenated_image_id,
                                        concatenated_image_data):
            cv2.imwrite(
                os.path.join(converted_folder_path, "{}.png".format(image_id)),
                image_data)


def _get_attribute_name_to_label_encoder_dict(accumulated_info_dataframe):
    attribute_name_to_label_encoder_dict = OrderedDict({})
    accumulated_info_dataframe = accumulated_info_dataframe.drop(
        columns=["image_file_path"])
    for attribute_name in accumulated_info_dataframe.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(accumulated_info_dataframe[attribute_name].values)
        attribute_name_to_label_encoder_dict[attribute_name] = label_encoder
    return attribute_name_to_label_encoder_dict


def load_Bengali():
    # Paths of folders
    root_folder_path_list = [
        os.path.expanduser("~/Documents/Local Storage/Dataset"),
        "/sgn-data/MLG/nixingyang/Dataset"
    ]
    root_folder_path_mask = [
        os.path.isdir(path) for path in root_folder_path_list
    ]
    root_folder_path = root_folder_path_list[root_folder_path_mask.index(True)]
    dataset_folder_name = "bengaliai-cv19"
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)

    # Paths of files
    train_parquet_file_path_list = sorted(
        glob.glob(
            os.path.join(dataset_folder_path, "train_image_data_*.parquet")))
    test_parquet_file_path_list = sorted(  # pylint: disable=unused-variable
        glob.glob(os.path.join(dataset_folder_path,
                               "test_image_data_*.parquet")))
    train_annotation_file_path = os.path.join(dataset_folder_path, "train.csv")

    # Convert the training split
    converted_train_folder_path = os.path.join(dataset_folder_path,
                                               "converted_train")
    _convert_split(converted_folder_path=converted_train_folder_path,
                   parquet_file_path_list=train_parquet_file_path_list)

    # Load annotations of the training split
    train_annotation_data_frame = pd.read_csv(train_annotation_file_path)
    train_annotation_data_frame["image_id"] = train_annotation_data_frame.apply(
        lambda row, converted_train_folder_path=
        converted_train_folder_path: os.path.join(
            converted_train_folder_path, "{}.png".format(row["image_id"])),
        axis=1)
    train_annotation_data_frame = train_annotation_data_frame.rename(
        columns={"image_id": "image_file_path"})

    train_and_valid_accumulated_info_dataframe = train_annotation_data_frame[[
        "image_file_path", "grapheme", "consonant_diacritic", "grapheme_root",
        "vowel_diacritic"
    ]]
    assert not train_and_valid_accumulated_info_dataframe.isnull().values.any(
    )  # All fields contain value
    train_and_valid_attribute_name_to_label_encoder_dict = _get_attribute_name_to_label_encoder_dict(
        train_and_valid_accumulated_info_dataframe)

    return train_and_valid_accumulated_info_dataframe, train_and_valid_attribute_name_to_label_encoder_dict


if __name__ == "__main__":
    load_Bengali()
