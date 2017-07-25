import tensorflow as tf
# from tensorflow.python.platform import tf_logging as logging
import image_preprocessing

from nets import inception

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
slim = tf.contrib.slim



# ============== Dataset Setup ============== #



# State the number of classes to predict
num_classes = 2

# State the lables file and read it
labels_file = './images/labels.txt'
labels = open(labels_file, 'r')

# Create a dictionary to refer to each label to its string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] # Get rid of the newline
    labels_to_name[int(label)] = string_name


# create the file pattern for each tf record
file_pattern = 'seatbelt_%s_*.tfrecord'

# Dictionary to keep track of variables, for dataset comprehention
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured image of a seatbelt in some capacity.',
    'label': 'A label that is of this format -- 0:seatbelt, 1:no_seatbelt, etc.'
}

# A GPU with 3.5GB memory could only do a max of 10 examples per batch
# So I should try like.... 2? 3?



# ============== Training Info ============== #



# State the number of epochs to train
num_epochs = 70

# State the batch size
batch_size = 3

# Learning rate information
# These are Hyperparameters, so they'll need tweaking as we train
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2


# Function for getting a split of our data set- either training or validation - from our dataset_dir
# Read in files following the file pattern
def get_split(split_name, dataset_dir, file_pattern=file_pattern):

    """
    Inputs:
        split_name (str) : 'train' or 'validation' for whichever split you want
        dataset_dir (str) : directory your TFRecords are located in
        file_pattern (str) : string that'll pattern match to the TFRecord filenames
    Outputs:
        dataset (Dataset) : a Dataset class object where we can read its different components for easy batch creation
    """

    # Argument validation
    if split_name not in ['train', 'validation']:
        raise ValueError("The split name %s is not recognized. Enter either 'training' or 'validation'." % (split_name))


    # Create the full paht for a general file pattern to locate the tfrecord files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # Count the total number of examples in all of these shards
    # We can hardcode this step if we just 'know' the format of our TFRecord files
    num_samples = 0
    file_pattern_for_counting = 'seatbelt_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)\
                                                            if file.startswith(file_pattern_for_counting)]

    # Iterate through all the encoded images in the TFRecord file and increment the counter for each
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1



    # Create a file reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded' : tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format' : tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label' : tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    # Create the items_to_handlers dictionary for the decoder
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label')
    }

    # Create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the dataset object
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_names = labels_to_name,
        items_to_descriptions = items_to_descriptions
    )

    return dataset



def load_batch(dataset, batch_size, height, width, is_training=True):

    """
        Input:
            dataset (Dataset) : a Dataset class object we create in out get_split funciton
            batch_size (int) : self-explanatory
            height (int) : height of the image to resize to during preprocessing
            width (int) : width of the image to resize to during preprocessing
            is_training (bool) : determine whether to perform training or validation pre-processing
        Output:
            images (Tensor) : a Tensor of the shape [batch_size, height, width, channels] corresponding to one batch of images
            labels (Tensor) : a Tensor of the shape (batch_size,) that contains the corresponding labels to the images
    """

    # Create the data_provider object, where much of the format encoding went into the dataset object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24
    )

    # Obtain our tensors from the dataprovider
    raw_image, label = data_provider.get(['image', 'label'])


    # Perform the correct preprocessing for the image
    image = image_preprocessing.preprocess_image(raw_image, height, width)

    # Preprocess the raw image for display purposes.
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_images(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    # Batch up the image by enqueuing the tensors internally into a FIFO queue
    # eventually dequeue many individual elements in tf.train.batch
    images, raw_image, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True
    )

    return images, raw_image, labels