#!/usr/bin/env python3
import ba.utils
import ba.plt
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import precision_recall_curve

obj = 'Person_Head'
obj = 'Person_Torso'

ITER = 500

ROOT = './data/results/' + obj + '_FCN_*/'
RESULTS = '*iter_' + ITER + '*results.mp'

with sns.color_palette("Set2", 10):
    fig, ax = ba.plt.newfig()
    for path in tqdm(glob(ROOT)):
        sp = path[len('./data/results/' + obj + '_FCN_'):-1]

        hitted_labels = []
        pred_labels = []
        for rmppath in glob(path + RESULTS):
            rmp = ba.utils.load(rmppath)
            hitted_labels.extend(rmp[b'hitted_labels'])
            pred_labels.extedn(rmp[b'pred_labels'])

        pr, rc, th = precision_recall_curve(hitted_labels, pred_labels,
                                            pos_label=1)
        ax.plot(rc, pr, label=sp)
    plt.xlabel('recall')
    plt.ylabel('precision')
    ax.legend()
    ba.plt.savefig('./' + obj + '_precs_recs')
