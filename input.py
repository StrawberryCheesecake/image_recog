import os
import random
import tensorpack

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

sess = tf.InteractiveSession()

# *** We gotta the mind comment below as we spin up our architecture. The Conv Net expects an input of certain size
# our models and our batch sizes ***

# # Process images of this size. Note that this differs from the original CIFAR
# # image size of 32 x 32. If one alters this number, then the entire model
# # architecture will change and any model would need to be retrained.

IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

def read_data(filename_queue):

# Whole file reader required for jpeg decoding
  image_reader = tf.WholeFileReader()

# We don't care about the filename, so we ignore the first tuple
  _, image_file = image_reader.read(filename_queue)

# Decode the jpeg images and set them to a universal size
# so we don't run into "out of bounds" issues down the road
  image_orig = tf.image.decode_jpeg(image_file, channels=3)

  image = tf.image.resize_images(image_orig, [224, 224])

  print(image)

  return image

def _gen_image_and_label_batch(image, label, batch_size, min_queue_examples, shuffle):

  num_preprocess_threads = 16

  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size = batch_size,
      num_threads = num_preprocess_threads,
      capacity = min_queue_examples + 3 * batch_size,
      min_after_dequeue = min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
      [image, label],
      batch_size = batch_size,
      num_threads = num_preprocess_threads,
      capacity = min_queue_examples + 3 * batch_size)

  # For testing purposes
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(batch_size):

  print("Executing distorted inputs script.")

  filenames = []
  for i in range(1000):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           "./images/seatbelt%d.jpg" % i)
    if not tf.gfile.Exists(filename):
      # print("Filename %s does not exist" % filename)
      continue
    else:
      filenames.append(filename)

# Create a string queue out of all filenames found in local 'images' directory
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue
  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input, tf.float32)

  # Dimensions of our tensors. Eventually we'd like to be able to dynamically resize our images
  # as another form of data augmentation, but that'll have to wait until we figure out how this
  # plays into the model's architecture
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  random_dim = random.randint(256, 480)

  print("Reached data augmentation.")

  # Data augmentation- we apply several kinds of distortions
  # some at a given probability, to create more permutations and variance in our data set

  # Randomly crop a [height, width] section of the image
  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

 # Randomly flip the image horizontally
  distorted_image = tf.image.random_flip_left_right(distorted_image)

 # Randomly brighten the image at a given probability
  distorted_image = tf.image.random_brightness(distorted_image, 
                                              max_delta = 63) if random.random() >= 0.4 else distorted_image

# Randomly saturate the photo's contrast

    # Ideally we'd do a PCA analysis of in all 3 of the images's pixel depths
    # and adjust the contrast based on some kind of calculated offset.

    # This can be added in when we refactor for segmentation and object detection
  distorted_image = tf.image.random_contrast(distorted_image,
                                          lower = 0.2, upper = 1.8) if random.random() >= 0.4 else distorted_image


  print("Reached data pre-processing.")

# Data Pre-processing- subtract off the mean and divide by the variance of the pixels
# to center the data on the origin and then normalize it
  float_image = tf.image.per_image_standardization(distorted_image)
  label = [1]

  # Set the shapes of the tensors
  float_image.set_shape([height, width, 3])

  # Ensure the random shuffling has good mixing properties
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                        min_fraction_of_examples_in_queue)

  print ('Filling the queue with training images before passing them to the model.')

  # Generate and return a batch
  return _gen_image_and_label_batch(float_image, label,
                                  min_queue_examples, batch_size,
                                  shuffle = True)


# Uncomment this to run the script
# distorted_inputs(2)
