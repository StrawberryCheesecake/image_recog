import tensorflow as tf
import random

from six.moves import xrange  # pylint: disable=redefined-builtin



def preprocess_image(image, height, width):

  """
  Pre-process an image tensor for training or evaluation

  Input:
    image: 3D Tensor [height, width, channels] contains the image
    height (int) : image expected width
    width (int) : image expected height
    is_training (boolean) : whether its for training or not, different preprocessing for evaluation

  Output:
    Also a 3D tensor that does shit and shit

  """
  image = tf.image.resize_images(image,[height,width])
  reshaped_image = tf.cast(image, tf.float32)
  reshaped_image = tf.reshape(reshaped_image, [height, width, 3])


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

  # Set the shapes of the tensors
  float_image.set_shape([height, width, 3])

  return float_image
