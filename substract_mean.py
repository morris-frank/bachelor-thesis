#!/usr/bin/env python3
import os
from glob import glob
from scipy.misc import imread, imresize
import numpy as np
from tqdm import tqdm

img_path = 'data/datasets/voc2010/JPEGImages/*jpg'
res_path = 'data/tmp/mean_substracted_voc/'
mean_path = 'data/models/resnet/ResNet_mean.npy'

mean = np.load(mean_path)

for ip in tqdm(glob(img_path)):
	bn = os.path.basename(ip)
	sp = res_path + bn[:-3] + 'npy'
	data = np.array(imread(ip), dtype=np.float32)
	data = data[:, :, ::-1]
	_mean = imresize(mean, data.shape)
	data -= np.array(_mean)
	data = data.transpose((2, 0, 1))
	np.save(sp, data)
