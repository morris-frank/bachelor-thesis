import os
from tqdm import tqdm
import numpy as np
import warnings
from scipy.misc import imread
import random
from . import utils


class SetList(object):
    """docstring for SetList."""

    def __init__(self, source=''):
        self.source = source
        self.target = source
        self.list = []
        self.mean = []
        if source != '':
            self.load()

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return '{}[{}] → {}'.format(self.source, len(self.list), self.target)

    def load(self):
        '''Loads the contents of self.source into the list.
        It does replace the whole content and does not append to it.
        '''
        utils.touch(self.source)
        with open(self.source) as f:
            self.list = [l[:-1] for l in f.readlines() if l.strip()]

    def save(self):
        '''Saves the list to the path set in self.target.
        This is normally set to self.source
        '''
        with open(self.target, 'w') as f:
            for row in self.list:
                f.write("{}\n".format(row))
        print('List {} written...'.format(self.target))

    def shuffle(self):
        '''Shuffles the list
        '''
        random.shuffle(self.list)

    def addPreSuffix(self, prefix='', suffix=''):
        '''Adds a prefix and a suffix to every element of the list.

        Args:
            prefix (str,optional): The prefix to prepend
            suffix (str,optional): The prefix to append
        '''
        self.list = [prefix + x + suffix for x in self.list]

    def rmPreSuffix(self, prefix='', suffix=''):
        '''Removes a prefix and a suffix from every element of the list.

        Args:
        prefix (str,optional): The prefix to remove
        suffix (str,optional): The prefix to remove
        '''
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
        '''Applies a callable to every element of the list

        Args:
            callback (func): The callback function to use

        Returns:
            True if successfull and False if not
        '''
        if not callable(callback):
            warnings.warn('Not callable object')
            return False
        print('Each of {}'.format(self.source))
        for row in tqdm(self.list):
            callback(row)
        return True
