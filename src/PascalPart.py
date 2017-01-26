import numpy as np
import os.path
import scipy.io as sio
from scipy.misc import imsave
from tqdm import tqdm

class PascalPart(object):
    """docstring for PascalPart."""
    def __init__(self, source=''):
        super(PascalPart, self).__init__()
        self.source = source
        if source != '':
            self.load()

    def load(self):
        mat = sio.loadmat(self.source)
        try:
            mat = mat['anno'][0][0][1][0][0]
        except IndexError:
            print('PascalPart::load: given file is wrong, %s', self.source)
            return False
        self.classname = mat[0][0]
        self.segmentation = mat[2].astype('float')
        self.parts = {}
        if mat[3].size and mat[3][0].size:
            for part in mat[3][0]:
                self.parts[part[0][0]] = part[1].astype('float')

    def save(self, image=True, parts=True, sum=False, segmentation=False):
        bn, en = os.path.splitext(self.source)
        itemsave = imsave if image else np.save
        ext = '.png' if image else ''
        if segmentation:
            itemsave(bn + ext, self.segmentation)
        if parts:
            if sum:
                sumOfParts = self.segmentation * 0
                for part in self.parts:
                    sumOfParts += self.parts[part]
                itemsave(bn + '_' + '_'.join(self.parts) + ext, sumOfParts)
            else:
                for part in self.parts:
                    itemsave(bn + '_' + part + ext, self.parts[part])

    def get(self, part):
        if part in self.parts:
            return self.parts[part]
        else:
            return self.segmentation * 0

    def reduce(self, parts):
        newparts = {}
        for part in parts:
            newparts[part] = self.get(part)
        self.parts = newparts
