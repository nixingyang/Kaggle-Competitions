import matplotlib
matplotlib.use("Agg")

import os
import glob
import shutil
import pylab
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from scipy.misc import imread, imresize
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit

# Dataset
DATASET_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset/The Nature Conservancy Fisheries Monitoring")
CROPPED_TRAIN_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "cropped_train")
CROPPED_TEST_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "cropped_test_stg1")
CLUSTERING_RESULT_FILE_PATH = os.path.join(DATASET_FOLDER_PATH, "clustering_result.npy")

# Workspace
WORKSPACE_FOLDER_PATH = os.path.join("/tmp", os.path.basename(DATASET_FOLDER_PATH))
CLUSTERING_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "clustering")
ACTUAL_DATASET_FOLDER_PATH = os.path.join(WORKSPACE_FOLDER_PATH, "actual_dataset")
ACTUAL_CROPPED_TRAIN_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "cropped_train")
ACTUAL_CROPPED_VALID_FOLDER_PATH = os.path.join(ACTUAL_DATASET_FOLDER_PATH, "cropped_valid")

# Output
OUTPUT_FOLDER_PATH = os.path.join(DATASET_FOLDER_PATH, "{}_output".format(os.path.basename(__file__).split(".")[0]))
OPTIMAL_WEIGHTS_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "Optimal Weights")
OPTIMAL_WEIGHTS_FILE_RULE = os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "epoch_{epoch:03d}-loss_{loss:.5f}-val_loss_{val_loss:.5f}.h5")
SUBMISSION_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "submission")
TRIAL_NUM = 10

# Image processing
IMAGE_ROW_SIZE = 256
IMAGE_COLUMN_SIZE = 256

# Training and Testing procedure
PERFORM_TRAINING = True
WEIGHTS_FILE_PATH = None
MAXIMUM_EPOCH_NUM = 1000
PATIENCE = 100
BATCH_SIZE = 32
SEED = 0

def perform_CV(image_path_list, resized_image_row_size=64, resized_image_column_size=64):
    if os.path.isfile(CLUSTERING_RESULT_FILE_PATH):
        print("Loading clustering result ...")
        image_name_to_cluster_ID_array = np.load(CLUSTERING_RESULT_FILE_PATH)
        image_name_to_cluster_ID_dict = dict(image_name_to_cluster_ID_array)
        cluster_ID_array = np.array([image_name_to_cluster_ID_dict[os.path.basename(image_path)] for image_path in image_path_list], dtype=np.int)
    else:
        print("Reading image content ...")
        image_content_array = np.array([imresize(imread(image_path), (resized_image_row_size, resized_image_column_size)) for image_path in image_path_list])
        image_content_array = np.reshape(image_content_array, (len(image_content_array), -1))
        image_content_array = np.array([(image_content - image_content.mean()) / image_content.std() for image_content in image_content_array], dtype=np.float32)

        print("Apply clustering ...")
        cluster_ID_array = DBSCAN(eps=1.5 * resized_image_row_size * resized_image_column_size, min_samples=20, metric="l1", n_jobs=-1).fit_predict(image_content_array)

        print("Saving clustering result ...")
        image_name_to_cluster_ID_array = np.transpose(np.vstack(([os.path.basename(image_path) for image_path in image_path_list], cluster_ID_array)))
        np.save(CLUSTERING_RESULT_FILE_PATH, image_name_to_cluster_ID_array)

    print("The ID value and count are as follows:")
    cluster_ID_values, cluster_ID_counts = np.unique(cluster_ID_array, return_counts=True)
    for cluster_ID_value, cluster_ID_count in zip(cluster_ID_values, cluster_ID_counts):
        print("{}\t{}".format(cluster_ID_value, cluster_ID_count))

    print("Visualizing clustering result ...")
    shutil.rmtree(CLUSTERING_FOLDER_PATH, ignore_errors=True)
    for image_path, cluster_ID in zip(image_path_list, cluster_ID_array):
        sub_clustering_folder_path = os.path.join(CLUSTERING_FOLDER_PATH, str(cluster_ID))
        if not os.path.isdir(sub_clustering_folder_path):
            os.makedirs(sub_clustering_folder_path)
        os.symlink(image_path, os.path.join(sub_clustering_folder_path, os.path.basename(image_path)))

    cv_object = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    for cv_index, (train_index_array, valid_index_array) in enumerate(cv_object.split(X=np.zeros((len(cluster_ID_array), 1)), groups=cluster_ID_array), start=1):
        print("Checking cv {} ...".format(cv_index))
        valid_sample_ratio = len(valid_index_array) / (len(train_index_array) + len(valid_index_array))
        if -1 in np.unique(cluster_ID_array[train_index_array]) and valid_sample_ratio > 0.15 and valid_sample_ratio < 0.25:
            train_unique_label, train_unique_counts = np.unique([image_path.split("/")[-2] for image_path in np.array(image_path_list)[train_index_array]], return_counts=True)
            valid_unique_label, valid_unique_counts = np.unique([image_path.split("/")[-2] for image_path in np.array(image_path_list)[valid_index_array]], return_counts=True)
            if np.array_equal(train_unique_label, valid_unique_label):
                train_unique_ratio = train_unique_counts / np.sum(train_unique_counts)
                valid_unique_ratio = valid_unique_counts / np.sum(valid_unique_counts)
                print("Using {:.2f}% original training samples as validation samples ...".format(valid_sample_ratio * 100))
                print("For training samples: {}".format(train_unique_ratio))
                print("For validation samples: {}".format(valid_unique_ratio))
                return train_index_array, valid_index_array

    assert False

def reorganize_dataset():
    # Get list of files
    original_image_path_list = sorted(glob.glob(os.path.join(CROPPED_TRAIN_FOLDER_PATH, "*/*")))

    # Perform Cross Validation
    train_index_array, valid_index_array = perform_CV(original_image_path_list)

    # Create symbolic links
    shutil.rmtree(ACTUAL_DATASET_FOLDER_PATH, ignore_errors=True)
    for folder_path, index_array in zip((ACTUAL_CROPPED_TRAIN_FOLDER_PATH, ACTUAL_CROPPED_VALID_FOLDER_PATH), (train_index_array, valid_index_array)):
        for index_value in index_array:
            original_image_path = original_image_path_list[index_value]
            path_suffix = original_image_path[len(CROPPED_TRAIN_FOLDER_PATH):]
            actual_original_image_path = folder_path + path_suffix
            os.makedirs(os.path.abspath(os.path.join(actual_original_image_path, os.pardir)), exist_ok=True)
            os.symlink(original_image_path, actual_original_image_path)

    return len(glob.glob(os.path.join(ACTUAL_CROPPED_TRAIN_FOLDER_PATH, "*/*"))), len(glob.glob(os.path.join(ACTUAL_CROPPED_VALID_FOLDER_PATH, "*/*")))

def init_model(target_num, additional_block_num=3, additional_filter_num=128, learning_rate=0.0001, freeze_pretrained_model=True):
    # Get the input tensor
    input_tensor = Input(shape=(3, IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE))

    # Convolutional blocks
    pretrained_model = VGG16(include_top=False, weights="imagenet", input_shape=input_tensor._keras_shape[1:])
    if freeze_pretrained_model:
        for layer in pretrained_model.layers:
            layer.trainable = False
    output_tensor = pretrained_model(input_tensor)

    # Additional convolutional blocks
    output_tensor = BatchNormalization(mode=0, axis=1)(output_tensor)
    for _ in range(additional_block_num):
        output_tensor = Convolution2D(additional_filter_num, 3, 3, subsample=(1, 1), activation="relu", border_mode="same")(output_tensor)
        output_tensor = BatchNormalization(mode=0, axis=1)(output_tensor)
    output_tensor = Convolution2D(target_num, 3, 3, subsample=(1, 1), activation="linear", border_mode="same")(output_tensor)
    output_tensor = GlobalAveragePooling2D()(output_tensor)
    output_tensor = Activation("softmax")(output_tensor)

    # Define and compile the model
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    plot(model, to_file=os.path.join(OPTIMAL_WEIGHTS_FOLDER_PATH, "model.png"), show_shapes=True, show_layer_names=True)

    # Load weights if applicable
    if WEIGHTS_FILE_PATH is not None:
        assert os.path.isfile(WEIGHTS_FILE_PATH), "Could not find file {}!".format(WEIGHTS_FILE_PATH)
        print("Loading weights from {} ...".format(WEIGHTS_FILE_PATH))
        model.load_weights(WEIGHTS_FILE_PATH)

    return model

def load_dataset(folder_path, classes=None, class_mode=None, batch_size=BATCH_SIZE, shuffle=True, seed=None):
    # Get the generator of the dataset
    data_generator_object = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1.0 / 255)
    data_generator = data_generator_object.flow_from_directory(
        directory=folder_path,
        target_size=(IMAGE_ROW_SIZE, IMAGE_COLUMN_SIZE),
        color_mode="rgb",
        classes=classes,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)

    return data_generator

class InspectLoss(Callback):
    def __init__(self):
        super(InspectLoss, self).__init__()

        self.train_loss_list = []
        self.valid_loss_list = []

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss")
        valid_loss = logs.get("val_loss")
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
        epoch_index_array = np.arange(len(self.train_loss_list)) + 1

        pylab.figure()
        pylab.plot(epoch_index_array, self.train_loss_list, "yellowgreen", label="train_loss")
        pylab.plot(epoch_index_array, self.valid_loss_list, "lightskyblue", label="valid_loss")
        pylab.grid()
        pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, mode="expand", borderaxespad=0.)
        pylab.savefig(os.path.join(OUTPUT_FOLDER_PATH, "Loss Curve.png"))
        pylab.close()

def ensemble_predictions(submission_folder_path):
    def _ensemble_predictions(ensemble_func, ensemble_submission_file_name):
        ensemble_proba = ensemble_func(proba_array, axis=0)
        ensemble_proba = ensemble_proba / np.sum(ensemble_proba, axis=1)[:, np.newaxis]
        ensemble_submission_file_content.loc[:, proba_columns] = ensemble_proba
        ensemble_submission_file_content.to_csv(os.path.join(submission_folder_path, ensemble_submission_file_name), index=False)

    # Read predictions
    submission_file_path_list = glob.glob(os.path.join(submission_folder_path, "Trial_*.csv"))
    print("There are {} submissions in total.".format(len(submission_file_path_list)))
    submission_file_content_list = [pd.read_csv(submission_file_path) for submission_file_path in submission_file_path_list]
    ensemble_submission_file_content = submission_file_content_list[0]

    # Concatenate predictions
    proba_columns = ensemble_submission_file_content.columns[1:]
    proba_list = [np.expand_dims(submission_file_content.as_matrix(proba_columns), axis=0)
                  for submission_file_content in submission_file_content_list]
    proba_array = np.vstack(proba_list)

    # Ensemble predictions
    for ensemble_func, ensemble_submission_file_name in \
        zip([np.max, np.min, np.mean, np.median], ["max.csv", "min.csv", "mean.csv", "median.csv"]):
        _ensemble_predictions(ensemble_func, ensemble_submission_file_name)

def run():
    print("Creating folders ...")
    os.makedirs(OPTIMAL_WEIGHTS_FOLDER_PATH, exist_ok=True)
    os.makedirs(SUBMISSION_FOLDER_PATH, exist_ok=True)

    print("Reorganizing dataset ...")
    train_sample_num, valid_sample_num = reorganize_dataset()

    print("Getting the labels ...")
    unique_label_list = sorted([folder_name for folder_name in os.listdir(CROPPED_TRAIN_FOLDER_PATH) if os.path.isdir(os.path.join(CROPPED_TRAIN_FOLDER_PATH, folder_name))])

    print("Initializing model ...")
    model = init_model(target_num=len(unique_label_list))

    if PERFORM_TRAINING:
        print("Performing the training procedure ...")
        train_generator = load_dataset(ACTUAL_CROPPED_TRAIN_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
        valid_generator = load_dataset(ACTUAL_CROPPED_VALID_FOLDER_PATH, classes=unique_label_list, class_mode="categorical", shuffle=True, seed=SEED)
        earlystopping_callback = EarlyStopping(monitor="val_loss", patience=PATIENCE)
        modelcheckpoint_callback = ModelCheckpoint(OPTIMAL_WEIGHTS_FILE_RULE, monitor="val_loss", save_best_only=True, save_weights_only=True)
        inspectloss_callback = InspectLoss()
        model.fit_generator(generator=train_generator,
                            samples_per_epoch=train_sample_num,
                            validation_data=valid_generator,
                            nb_val_samples=valid_sample_num,
                            callbacks=[earlystopping_callback, modelcheckpoint_callback, inspectloss_callback],
                            nb_epoch=MAXIMUM_EPOCH_NUM, verbose=2)
    else:
        assert WEIGHTS_FILE_PATH is not None

        print("Performing the testing procedure ...")
        for trial_index in np.arange(TRIAL_NUM) + 1:
            print("Working on trial {}/{} ...".format(trial_index, TRIAL_NUM))
            submission_file_path = os.path.join(SUBMISSION_FOLDER_PATH, "Trial_{}.csv".format(trial_index))
            if not os.path.isfile(submission_file_path):
                print("Performing the testing procedure ...")
                test_generator = load_dataset(CROPPED_TEST_FOLDER_PATH, shuffle=False, seed=trial_index)
                prediction_array = model.predict_generator(generator=test_generator, val_samples=len(test_generator.filenames))
                image_name_array = np.expand_dims([os.path.basename(image_path) for image_path in test_generator.filenames], axis=-1)
                index_array_for_sorting = np.argsort(image_name_array, axis=0)
                submission_file_content = pd.DataFrame(np.hstack((image_name_array, prediction_array))[index_array_for_sorting.flat])
                submission_file_content.to_csv(submission_file_path, header=["image"] + unique_label_list, index=False)

        print("Performing ensembling ...")
        ensemble_predictions(SUBMISSION_FOLDER_PATH)

    print("All done!")

if __name__ == "__main__":
    run()