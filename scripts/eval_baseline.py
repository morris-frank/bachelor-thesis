#!/usr/bin/env python3
from glob import glob
import os
import ba.eval
from tqdm import tqdm
import ba.plt
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_palette("Set2", 5)

MODE = 'PR'

ns = [1, 10, 25, 50, 100]
ns = [100]

tags = ['cow_sheep_lhorn_rhorn', 'person_neck', 'bird_head',
        'bottle_body', 'pottedplant_plant', 'person_hair']

for tag in tqdm(tags):
    slicefile = 'data/tmp/pascpart/patches/{}/seg.yaml'.format(tag)
    fig, ax = ba.plt.newfig()
    for n in tqdm(ns):
        predfs = 'data/tmp/baseline/{}_{}samples*.yaml'.format(tag, n)
        mpfs = 'data/tmp/baseline/{}_{}samples*.mp'.format(tag, n)
        for predf in tqdm(glob(predfs)):
            mpf = predf[:-11] + 'results.mp'
            if not os.path.isfile(mpf):
                ba.eval.evalDect(predf, slicefile)
        if MODE == 'PR':
            ba.plt._plt_results(tag, 'PR', mpfs, str(n) + 'samples', ax=ax)
        elif MODE == 'ROC':
            ba.plt._plt_results(tag, 'ROC', mpfs, str(n) + 'samples', ax=ax)

    if MODE == 'PR':
        plt.xlabel('recall')
        plt.ylabel('precision')
        ax.legend()
        ba.plt.savefig('./data/tmp/baseline/' + tag + '_precs_recs')
    elif MODE == 'ROC':
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        ax.legend()
        ba.plt.savefig('./data/tmp/baseline/' + tag + '_roc')
