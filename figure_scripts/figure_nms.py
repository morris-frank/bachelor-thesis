#!/usr/bin/env python3
from scipy.misc import imread
import ba.plt
import numpy as np
import seaborn as sns

N = 5
cmap = sns.cubehelix_palette(N, start=2.1, rot=-0.2, gamma=0.6)

im = imread('./2010_002274.jpg')
im = im[0:300, 0:270, :]
dr = np.load('./2010_002274_rects.npy').tolist()
bbscores = np.array(dr['bbscores'])
bbscores -= bbscores.min()
bbscores /= bbscores.max()
rects = np.array([(s[0], s[1], e[0], e[1])
                  for s, e in zip(dr['starts'], dr['ends'])])
picks = dr['picks']
picked_bbscores = bbscores[picks]
picked_rects = rects[picks]

sort_idx = np.argsort(bbscores)
bbscores = bbscores[sort_idx]
rects = rects[sort_idx]

sort_idx = np.argsort(picked_bbscores)
picked_bbscores = picked_bbscores[sort_idx]
picked_rects = picked_rects[sort_idx]

colors = [cmap[int(s * (N - 1))] for s in bbscores]
picked_colors = [cmap[int(s * (N - 1))] for s in picked_bbscores]
# npicks = [r[0] < 300 for r in rects]
# npicks_picks = [r[0] < 300 for r in rects[picks]]

fig, ax = ba.plt.apply_rect(im, rects.tolist(), colors=colors)
ba.plt.savefig('./nms_before')
fig, ax = ba.plt.apply_rect(im, picked_rects.tolist(), colors=picked_colors)
ba.plt.savefig('./nms_after')
