#!/usr/bin/env python3
import ba.utils
import ba.plt
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import seaborn as sns

obj = 'Person_Head'
obj = 'Person_Torso'

ROOT = './data/results/' + obj + '_FCN_*/classifier_train_iter_500.prec_rec.mp'

with sns.color_palette("Set2", 10):
    fig, ax = ba.plt.newfig()
    for path in tqdm(glob(ROOT)):
        # it = str(os.path.basename(path))
        # it = it[len('classifier_train_iter_'):-len('.prec_rec.mp')]
        sp = os.path.dirname(path)
        sp = sp[len('./data/results/' + obj + '_FCN_'):]
        content = ba.utils.load(path)
        ax.plot(content[b'recall'], content[b'precision'], label=sp)
    plt.xlabel('recall')
    plt.ylabel('precision')
    ax.legend()
    ba.plt.savefig('./' + obj + '_precs_recs')
