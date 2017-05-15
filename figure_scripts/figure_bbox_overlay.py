#!/usr/bin/env python3
from skimage.data import imread
import ba.plt
import numpy as np
import ba.utils
import seaborn as sns
from glob import glob
import os
from tqdm import tqdm

N = 5
cmap = sns.cubehelix_palette(N, start=2.1, rot=-0.2, gamma=0.6)
idxs = ['2008_002067', '2009_004323', '2009_004784', '2009_005222',
        '2009_002715', '2010_005967']


n = '50'
tag = 'person_hair_FCN_{}samples'.format(n)

for path in tqdm(glob('data/results/{}/*yaml'.format(tag))):
    y = ba.utils.load(path)
    bn = os.path.splitext(os.path.basename(path))[0].split('.')[0]
    for idx in idxs:
        im = imread('data/datasets/voc2010/JPEGImages/' + idx + '.jpg')
        rects = np.array(y[idx]['region'])
        bbscores = np.array(y[idx]['score'])
        bbscores -= bbscores.min()
        bbscores /= bbscores.max()
        picks = bbscores > 0.5
        bbscores = bbscores[picks]
        rects = rects[picks]

        sort_idx = np.argsort(bbscores)
        bbscores = bbscores[sort_idx]
        rects = rects[sort_idx]

        colors = [cmap[int(s * (N - 1))] for s in bbscores]
        fig, ax = ba.plt.apply_rect(im, rects.tolist(), colors=colors,
                                    path='build/{}_bbox_{}s_{}'.format(
                                        bn, n, idx))
