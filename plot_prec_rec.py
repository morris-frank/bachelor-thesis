#!/usr/bin/env python3
import ba.utils
import ba.plt
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import seaborn as sns

ROOT = './data/results/Person_Head_FCN_*/classifier_train_iter_500.prec_rec.mp'

with sns.color_palette("Set2", 10):
    fig, ax = ba.plt.newfig()
    for path in tqdm(glob(ROOT)):
        # it = str(os.path.basename(path))
        # it = it[len('classifier_train_iter_'):-len('.prec_rec.mp')]
        sp = os.path.dirname(path)
        sp = sp[len('./data/results/Person_Head_FCN_'):]
        content = ba.utils.load(path)
        ax.plot(content[b'recall'], content[b'precision'], label=sp)
    plt.xlabel('recall')
    plt.xlabel('precision')
    ax.legend()
    ba.plt.savefig('./precs_recs')
