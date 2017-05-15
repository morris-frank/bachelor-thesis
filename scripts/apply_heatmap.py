#!/usr/bin/env python3
import ba.utils
from glob import glob
import os.path as path
import sys
from scipy.misc import imread
import skimage.transform as tf
from tqdm import tqdm


def doDir(images, heatmaps, target):
    images = path.normpath(images) + '/'
    heatmaps = path.normpath(heatmaps) + '/'
    target = path.normpath(target) + '/'
    _images = glob(images + '*jpg')
    _heatmaps = glob(heatmaps + '*png')

    for heatmap in tqdm(_heatmaps):
        bn = path.splitext(path.basename(heatmap))[0]
        imf = images + bn + '.jpg'
        if imf not in _images:
            continue
        outf = target + bn + '.png'
        im = imread(imf)
        hm = imread(heatmap)
        hm = tf.resize(hm, (im.shape[0], im.shape[1]), mode='reflect')
        ba.utils.apply_overlay(im, hm, outf)


def main(args):
    images = 'data/datasets/voc2010/JPEGImages/'
    heatmaps = args[0] + '/heatmaps/'
    target = ba.utils.touch(args[0] + '/heatmaps_overlays/')
    doDir(images, heatmaps, target)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No arguments given')
        sys.exit()
    main(sys.argv[1:])
