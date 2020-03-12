import os
import sys

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, load_model

# https://github.com/tensorflow/tensorflow/issues/29161
# https://github.com/keras-team/keras/issues/10340
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())
K.set_session(session)

flags = tf.compat.v1.app.flags
flags.DEFINE_string("model_file_path", "model.h5",
                    "File path of the trained model.")
flags.DEFINE_string("suffix", "unspecified", "Suffix in the file name.")
FLAGS = flags.FLAGS


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    model_file_path = FLAGS.model_file_path
    suffix = FLAGS.suffix

    print("Loading the model from training ...")
    model = load_model(model_file_path,
                       custom_objects={
                           "tf": tf,
                           "swish": tf.nn.swish
                       },
                       compile=False)

    inference_model_file_path = os.path.abspath(
        os.path.join(model_file_path, "../inference_{}.h5".format(suffix)))
    print("Saving the model for inference to {} ...".format(
        inference_model_file_path))
    inference_model = Model(inputs=[model.input], outputs=model.output[1:])
    if os.path.isfile(inference_model_file_path):
        os.remove(inference_model_file_path)
    inference_model.save(inference_model_file_path)

    print("All done!")


if __name__ == "__main__":
    tf.compat.v1.app.run()
