from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

import matplotlib.pyplot as plt
import math


import time

from datasets import dataset_utils

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

import input_loader as ipl

image_size = inception.inception_v1.default_image_size
checkpoints_dir = '/tmp/checkpoints'
batch_size = 3
train_dir = '/tmp/inception_finetuned/'
with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = ipl.get_split('train','./images')
    images, images_raw, labels = ipl.load_batch(dataset,3,height=image_size, width=image_size)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

    probabilities = tf.nn.softmax(logits)

    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(
      checkpoint_path,
      slim.get_variables_to_restore())

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])

            for i in range(batch_size):
                image = np_images_raw[i, :, :, :]
                true_label = np_labels[i]
                predicted_label = np.argmax(np_probabilities[i, :])
                predicted_name = dataset.labels_to_names[predicted_label]
                true_name = dataset.labels_to_names[true_label]

                plt.figure()
                plt.imshow(image.astype(np.uint8))
                plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                plt.axis('off')
                plt.show()
