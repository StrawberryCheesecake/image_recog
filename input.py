import os
import random 
import tensorpack

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# *** We gotta the mind comment below as we spin up our architecture. The Conv Net expects an input of certain size 
# our models and our batch sizes ***

# # Process images of this size. Note that this differs from the original CIFAR
# # image size of 32 x 32. If one alters this number, then the entire model
# # architecture will change and any model would need to be retrained.

IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_data(filename_queue):
  return 1

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # Filenames for LABELS not for GETTING THE DATA 

  # This is how we get our filenames, you can safely copy this into your boilerplate 
  # And reconfigure it to your filepaths
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_datafilename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)


  # Change this to the dynamic setting 
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  random_dim = random.randint(256, 480)

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly reshape the image 
  distorted_image = tf.image.resize_images(reshaped_image, [random_dim, random_dim]) if random.random() >= 0.3 else distorted_image

  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image) if random.random() >= 0.5 else distorted_image

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image =  tf.image.random_brightness(distorted_image, 
                                                max_delta=63) if random.random() >= 0.4 else distorted_image

# Here we'd like to insert a PCA analysis of all the image's RGB pixels
# and calculate some sort of offset to adjust contrast

  distorted_image = tf.image.random_contrast(distorted_image, 
                                             lower=0.2, upper=1.8) if random.random() >= 0.4 else distorted_image

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(distorted_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d images before starting to train.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


