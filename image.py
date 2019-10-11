import os
import sys

import functools
import numpy as np
import matplotlib.image as img
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import matplotlib as mpl


PATH_IMAGES = './images/'
PATH_STYLES = './styles/'
PATH_OUTPUTS = './output/'
MAX_IMAGE_DIM = 1024

CONTENT_IMAGE = "pam1.jpg"
STYLE_IMAGE = "abo2.jpg"

def load_image(path_to_img):

    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = MAX_IMAGE_DIM / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


if __name__ == '__main__':

    count = len(sys.argv)
    if count != 3:
        print('usage: image.py image style')
        exit()

    name_original = sys.argv[1]
    name_style = sys.argv[2]
    print(f'converting: {name_original} {name_style}')

    print("TF Version: ", tf.__version__)
    print("TF-Hub version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.test.is_gpu_available())

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1'
    hub_module = hub.load(hub_handle)

    image_original = load_image(PATH_IMAGES + name_original);
    image_style = load_image(PATH_STYLES + name_style);

    results = hub_module(tf.constant(image_original), tf.constant(image_style))

    #bitmap = results[0]
    image = tf.squeeze(results[0], axis=0)

    output_name = os.path.splitext(name_original)[0] + '.' + os.path.splitext(name_style)[0] + '.jpg'
    mpl.image.imsave(PATH_OUTPUTS + output_name, image)

    print(f'result: {PATH_OUTPUTS + output_name}')

