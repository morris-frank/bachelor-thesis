import os
from tqdm import tqdm
import numpy as np
import warnings
from scipy.misc import imread
from random import shuffle
from src.utils import touch


class SetList(object):
    """docstring for SetList."""
    list = []
    mean = []

    def __init__(self, source=''):
        self.source = source
        self.target = source
        if source != '':
            self.load()

    def __len__(self):
        return len(self.list)

    def load(self):
        touch(self.source)
        with open(self.source) as f:
            self.list = [l[:-1] for l in f.readlines() if l.strip()]

    def save(self):
        with open(self.target, 'w') as f:
            for row in self.list:
                f.write("{}\n".format(row))
        print('List {} written...'.format(self.target))

    def shuffle(self):
        self.list = shuffle(self.list)

    def addPreSuffix(self, prefix, suffix):
        self.list = [prefix + x + suffix for x in self.list]

    def rmPreSuffix(self, prefix, suffix):
        self.list = [x[len(prefix):-len(suffix)] for x in self.list]

    def calculate_mean(self):
        self.mean = [[],[],[]]
        print('Calculating mean pixel...')
        for row in tqdm(self.list):
            im = imread(row)
            self.mean[0].append(np.mean(im[...,0]))
            self.mean[1].append(np.mean(im[...,1]))
            self.mean[2].append(np.mean(im[...,2]))
        self.mean = np.mean(self.mean, axis=1)
        return self.mean

    def each(self, callback):
        if not callable(callback):
            warnings.warn('Not callable object')
            return
        print('Each of {}'.format(self.source))
        for row in tqdm(self.list):
            callback(row)
