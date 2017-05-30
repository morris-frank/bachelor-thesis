'''
    Based on code from Evan Shelhamer
    fcn.berkeleyvision.org
'''
from ba import BA_ROOT
import ba.utils
import caffe
import numpy as np
from PIL import Image
import random
from scipy.misc import imread
import yaml


class DirDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass


class SegDataLayer(caffe.Layer):
    '''
    '''
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.images = params['images']
        self.labels = params['labels']
        self.splitfile = params['splitfile']
        if isinstance(params['mean'], str):
            self.mean = np.load(params['mean'])
        else:
            self.mean = np.array(params['mean'])
        self.extension = params.get('extension', 'jpg')
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.idx = 0
        # load indices for images and labels
        self.indices = open(self.splitfile, 'r').read().splitlines()

        # make eval deterministic
        if 'train' not in self.splitfile:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices) - 1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        '''
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        '''
        im = Image.open('{}/{}.{}'.format(self.images, idx, self.extension))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        try:
            in_ -= self.mean
        except Exception:
            pass
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_label(self, idx):
        '''
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        '''
        label = imread('{}/{}.png'.format(self.labels, idx))
        label = (label / 255).astype(np.uint8)
        label = label[np.newaxis, ...]
        return label


class PosPatchDataLayer(SegDataLayer):
    '''
    '''
    def setup(self, bottom, top):
        super().setup(bottom, top)

        with open(self.labels, 'r') as f:
            self.slices = yaml.load(f)

    def load_label(self, idx):
        label = np.zeros(self.data.shape[1:], dtype=np.uint8)
        label[self.slices[idx]].fill(1)
        label = label[np.newaxis, ...]
        return label


class SingleImageLayer(caffe.Layer):
    '''
    '''
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.images = params['images']
        self.ext = params.get('extension', 'jpg')
        self.batch_size = params.get('batch_size', 20)
        self.patch_size = params.get('patch_size', (224, 224))
        self.ppI = params.get('ppI', None)
        self.slicefile = params['slicefile']
        self.splitfile = params['splitfile']
        self.negatives = params.get(
            'negatives',
            BA_ROOT + 'data/tmp/var_neg/')
        if isinstance(params['mean'], str):
            self.mean = np.load(params['mean'])
        else:
            self.mean = np.array(params['mean'])

        with open(self.splitfile, 'r') as f:
            imlist = [l[:-1] for l in f.readlines() if l.strip()]

        self.flow = ba.utils.SamplesGenerator(
            self.slicefile,
            imlist,
            self.images,
            self.negatives,
            ppI=self.ppI,
            patch_size=self.patch_size,
            ext=self.ext,
            mean=self.mean,
            batch_size=self.batch_size)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

    def reshape(self, bottom, top):
        top[0].reshape(
            self.batch_size, 3, self.patch_size[0], self.patch_size[1])
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):
        # assign output
        x, y = self.flow.next()
        filled = x.shape[0]
        top[0].data[:filled, ...] = x
        top[1].data[:filled, ...] = y[..., np.newaxis]

        while filled < self.batch_size:
            x, y = self.flow.next()
            filling = x.shape[0]
            if filling + filled >= self.batch_size:
                rem = self.batch_size - filled
                top[0].data[filled:, ...] = x[:rem, ...]
                top[1].data[filled:, ...] = y[:rem, np.newaxis]
            else:
                top[0].data[filled:filled + filling, ...] = x[...]
                top[1].data[filled:filled + filling, ...] = x[..., np.newaxis]
            filled += filling

    def backward(self, top, propagate_down, bottom):
        pass
