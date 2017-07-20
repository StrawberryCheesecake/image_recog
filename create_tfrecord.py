import random
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset

flags = tf.app.flags

# State the dataset directory 
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# Proportion of the dataset used for validation
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset used for validation')

# The number of shards to split the dataset into 
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files into')

# Seed for repeatability 
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability')

# Ouput filename for naming the TFRecord file 
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def main() :

    # Check to make sure a TFRecord Filename was provided
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty, please provide an argument.')
    
    # Check to make sure a dataset directory was provided
    
    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument')
    
    # Check to see if there TFRecords file is already in the directory- exit if thats the case
    if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print 'Dataset files already exist. Exiting without re-creating them.'
        return None  

    # Get photo filenames and class names
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)
    
    #--- fix for Mac - remove .DS_Store from photo_names
    photo_filenames = filter(lambda k: '.DS_Store' not in k, photo_filenames)
    
    # Refer the class names to a specific integer in a dict for reference later 
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Find the number of validation examples needed 
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide the training dataset into training and test 
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)

    # Set training and validation filenames 
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # Convert the training and validation sets into TFReader Formats 
    _convert_dataset('train', training_filenames, class_names_to_ids, 
                    dataset_dir = FLAGS.dataset_dir, 
                    tfrecord_filename = FLAGS.tfrecord_filename, 
                    _NUM_SHARDS = FLAGS.num_shards)

    _convert_dataset('validation', validation_filenames, class_names_to_ids, 
                    dataset_dir = FLAGS.dataset_dir,
                    tfrecord_filename = FLAGS.tfrecord_filename, 
                    _NUM_SHARDS = FLAGS.num_shards)

    # Write the labels file
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    print '\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename)

if __name__ == "__main__":
    main()



