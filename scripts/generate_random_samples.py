#!/usr/bin/env python3
from glob import glob
import os.path
from scipy.misc import imread
import random
from scipy.misc import imresize
from scipy.misc import imsave
from tqdm import tqdm


PS = (224, 224)
Folder = '/home/morris/var/hci-storage01/arthistoric_images/imageFiles_8'

d = os.path.normpath(Folder) + '/'
td = os.path.normpath(Folder) + '_patches/'
for imf in tqdm(glob(d + '*png')):
    im = imread(imf)
    bn = os.path.basename(imf)
    for _ in range(9):
        h, w = im.shape[:2]
        _h = int(h * random.uniform(.01, .5))
        _w = int(w * random.uniform(.01, .5))
        _y = random.randint(0, h - _h - 1)
        _x = random.randint(0, w - _w - 1)
        if im.ndim == 3:
            patch = im[_y:_y + _h, _x:_x + _w, :]
            patch = imresize(patch, (224, 224, 3))
        else:
            patch = im[_y:_y + _h, _x:_x + _w]
            patch = imresize(patch, (224, 224))
        imsave('{}{}_{}.png'.format(td, bn, _), patch)
