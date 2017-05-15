#!/usr/bin/env python3
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imsave, imread
import skimage.color
import numpy as np
import random

imp = './build/parts_example.jpg'

im = imread(imp)
im = im.transpose((2, 0, 1))[np.newaxis, ...]


def preprocess_image(im):
    im /= 255
    hsv_im = skimage.color.rgb2hsv(im.transpose(1, 2, 0))
    power_s = random.uniform(0.25, 4)
    power_v = random.uniform(0.25, 4)
    factor_s = random.uniform(0.7, 1.4)
    factor_v = random.uniform(0.7, 1.4)
    value_s = random.uniform(-0.1, 0.1)
    value_v = random.uniform(-0.1, 0.1)
    hsv_im[:, :, 1] = np.power(hsv_im[:, :, 1], power_s) * factor_s + value_s
    hsv_im[:, :, 2] = np.power(hsv_im[:, :, 2], power_v) * factor_v + value_v
    im = skimage.color.hsv2rgb(hsv_im).transpose(2, 0, 1)
    im *= 255
    return im


flow = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    data_format='channels_first',
    preprocessing_function=preprocess_image
    ).flow(im, np.array([1]), batch_size=20)

for i in range(20):
    sample, labels = next(flow)
    imsave('./build/parts_example_{}.jpg'.format(i),
           sample[0].transpose((1, 2, 0)))
