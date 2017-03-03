from ba import utils
from scipy.misc import imread
from tqdm import tqdm
import numpy as np
import os
import random
import warnings


class SetList(object):
    '''A class to hold lists of inputs for a network'''

    def __init__(self, source='', target=None):
        '''Constructs a new SetList.

        Args:
            source (str): The path to the list file
        '''
        self.source = source
        if target is None:
            self.target = source
        else:
            self.target = target
        self.list = []
        self.mean = []
        if source != '':
            self.load()

    def __len__(self):
        '''Returns the length of this Set'''
        return len(self.list)

    def __str__(self):
        '''Returns a str-description of this Set'''
        return '{}[{}] → {}'.format(self.source, len(self.list), self.target)

    def __iter__(self):
        '''Returns the iterator for the contained list'''
        return iter(self.list)

    def load(self):
        '''Loads the contents of self.source into the list. If source is a dir
        it will list all files in it without extensions. It does replace the
        whole content and does not append to it.'''
        utils.touch(self.source)
        if os.path.isdir(self.source):
            self.loadDir(self.source)
            self.source = ''
            self.target = ''
        else:
            with open(self.source) as f:
                self.list = [l[:-1] for l in f.readlines() if l.strip()]

    def loadDir(self, dir):
        '''Loads the contents of a dirctory into the list

        Args:
            dir (str): The path to the dir
        '''
        self.list = [os.path.splitext(f)[0] for f in next(os.walk(dir))[2]]

    def write(self):
        '''Saves the list to the path set in self.target. This is normally set
        to self.source'''
        with open(self.target, 'w') as f:
            for row in self:
                f.write("{}\n".format(row))
        print('List {} written...'.format(self.target))

    def shuffle(self):
        '''Shuffles the list'''
        random.shuffle(self.list)

    def addPreSuffix(self, prefix='', suffix=''):
        '''Adds a prefix and a suffix to every element of the list.

        Args:
            prefix (str,optional): The prefix to prepend
            suffix (str,optional): The prefix to append
        '''
        self.list = [prefix + x + suffix for x in self]

    def rmPreSuffix(self, prefix='', suffix=''):
        '''Removes a prefix and a suffix from every element of the list.

        Args:
            prefix (str,optional): The prefix to remove
            suffix (str,optional): The prefix to remove
        '''
        self.list = [x[len(prefix):-len(suffix)] for x in self]

    def calculate_mean(self):
        '''Calculates the mean pixel for this set. The list has to contain full
        paths obviously so you probably have to append Prefixes and suffixes
        before running this.

        Returns:
            The mean pixel. As BGR!
        '''
        self.mean = [[],[],[]]
        print('Calculating mean pixel...')
        for row in tqdm(self):
            im = imread(row)
            self.mean[0].append(np.mean(im[...,0]))
            self.mean[1].append(np.mean(im[...,1]))
            self.mean[2].append(np.mean(im[...,2]))
        self.mean = np.mean(self.mean, axis=1)
        if self.mean.shape == (3,):
            return self.mean
        else:
            return self.mean[:,:,::-1]

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
        for row in tqdm(self):
            callback(row)
        return True
