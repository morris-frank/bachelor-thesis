#!/usr/bin/env python3
from skimage.data import imread
from scipy.misc import imresize
import scipy.ndimage
import numpy as np
import skimage

import ba.plt
import matplotlib.pyplot as plt

base = 'data/results/person_hair_FCN_50samples/train_iter_500/'

im = imread(base + 'heatmaps_overlays/2010_003944.png')
hm = imread(base + 'heatmaps/2010_003944.png')

# Scale heatmap to image size
hm = imresize(hm, im.shape[:-1])
hm = skimage.img_as_float(hm)
fig = ba.plt.plt_hm(hm)
plt.show()
ba.plt.savefig('distance_transform_hm')

# Threshould heatmap to get the near zero values
hmnull = hm < 0.1
ba.plt.plt_hm(hmnull)
ba.plt.savefig('distance_transform_hm_thres')

# Compute the negative distance transform inside the near zero values
hmedt = scipy.ndimage.distance_transform_cdt(hmnull).astype(float)
hmedt /= np.sum(hmedt)
ba.plt.plt_hm(hmedt)
ba.plt.savefig('distance_transform_hm_negative')
