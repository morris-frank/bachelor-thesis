#!/usr/bin/env python3
import ba.utils
import ba.plt
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
import sklearn.metrics


obj = 'NDT_Person_Head'
# obj = 'Person_Torso'

ITER = '500'

ROOT = './data/results/' + obj + '_FCN_*/'
RESULTS = '*iter_' + ITER + '*results.mp'

AUC = 0
PR = 1
ROC = 2

mode = AUC

with sns.color_palette("Set2", 10):
    fig, ax = ba.plt.newfig()
    for path in tqdm(glob(ROOT), desc=obj):
        sp = path[len('./data/results/' + obj + '_FCN_'):-1]

        hitted_labels = []
        pred_labels = []
        for rmppath in tqdm(glob(path + RESULTS), desc=sp):
            rmp = ba.utils.load(rmppath)
            hitted_labels.extend(rmp[b'hitted_labels'])
            pred_labels.extend(rmp[b'pred_labels'])

        if mode == AUC:
            auc = sklearn.metrics.roc_auc_score(hitted_labels, pred_labels)
            tqdm.write('{}; {}'.format(sp, auc))
        elif mode == PR:
            pr, rc, th = sklearn.metrics.precision_recall_curve(
                hitted_labels, pred_labels, pos_label=1)
            ax.plot(rc, pr, label=sp)
        elif mode == ROC:
            fpr, tpr, th = sklearn.metrics.roc_curve(
                hitted_labels, pred_labels, pos_label=1)
            ax.plot(fpr, tpr, label=sp)
    if mode == PR:
        plt.xlabel('recall')
        plt.ylabel('precision')
        ax.legend()
        ba.plt.savefig('./' + obj + '_precs_recs')
    elif mode == ROC:
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        ax.legend()
        ba.plt.savefig('./' + obj + '_roc')
