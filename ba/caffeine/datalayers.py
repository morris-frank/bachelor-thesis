'''
    Based on code from Evan Shelhamer
    fcn.berkeleyvision.org
'''
import caffe
from glob import glob
import numpy as np
import os.path
from PIL import Image
import random
from scipy.misc import imread
from scipy.misc import imresize
import yaml


class SegDataLayer(caffe.Layer):
    '''
    '''
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.images = params['images']
        self.labels = params['labels']
        self.splitfile = params['splitfile']
        if isinstance(params['mean'], str):
            self.mean = params['mean']
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
            self.idx = random.randint(0, len(self.indices)-1)

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
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_label(self, idx):
        '''
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        '''
        label = imread('{}/{}.png'.format(self.labels, idx))
        label = (label/255).astype(np.uint8)
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
        from keras.preprocessing.image import ImageDataGenerator
        params = eval(self.param_str)
        self.images = params['images']
        self.ext = params['extension']
        self.slices = params['slices']
        # self.negatives = params['negatives']
        self.negatives = 'data/tmp/pascpart/patches/aeroplane_stern/img_augmented/neg/'
        if isinstance(params['mean'], str):
            self.mean = np.load(params['mean'])
        else:
            self.mean = np.array(params['mean'])
        self.batch_size = params.get('batch_size', 32)
        self.batch_size = params.get('batch_size', 24)
        self.patch_size = params.get('patch_size', (224, 224))
        self.ppI = params.get('ppI', 20)

        n = len(self.slices)
        self.samples = np.zeros((n * self.ppI * 2,
                                 self.patch_size[0], self.patch_size[1], 3))
        for it, (path, bb) in zip(range(n), self.slices.items()):
            im = self.imread('{}{}.{}'.format(self.images, path, self.ext))
            self.samples[it:it + self.ppI, ...] = self.generateSamples(im, bb)
        self.labels = np.append(np.ones(n * self.ppI),
                                np.zeros(n * self.ppI))

        negs = glob(self.negatives + '/*png')
        negs = random.sample(negs, n * self.ppI)
        for it, neg in zip(range(1, len(negs) + 1), negs):
            im = self.imread(neg)
            im = imresize(im, (self.patch_size[0], self.patch_size[1], 3))
            im = im.astype(np.float32)
            im -= self.mean
            self.samples[-it, ...] = im

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.generator = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True
            )

        top[0].reshape(
            self.batch_size, 3, self.patch_size[0], self.patch_size[1])
        top[1].reshape(self.batch_size, 1)

    def bbShape(self, bb):
        return (bb[0].stop - bb[0].start,
                bb[1].stop - bb[1].start)

    def generateSamples(self, im, bb, shiftfactor=0.1, count=None):
        if count is None:
            count = self.ppI
        if count % 2 != 0:
            print('Count in SingleImageLayer is not divisble by two')
            return
        bbshape = self.bbShape(bb)
        shift = (bbshape[0] * shiftfactor, bbshape[1] * shiftfactor)

        xsamples = (shift[0] * (2 * np.random.random(count) - 1)).astype(np.int8)
        ysamples = (shift[1] * (2 * np.random.random(count) - 1)).astype(np.int8)

        samples = np.zeros((count, self.patch_size[0], self.patch_size[1], 3))
        for it, xsample, ysample in zip(range(len(xsamples)), xsamples, ysamples):
            xstart = max(bb[0].start + xsample, 0)
            xstop = min(bb[0].stop + xsample, im.shape[0])
            ystart = max(bb[1].start + ysample, 0)
            ystop = min(bb[1].stop + ysample, im.shape[1])
            patch = im[xstart:xstop, ystart:ystop]
            if patch.shape[:-1] != self.patch_size:
                patch = imresize(patch, (self.patch_size[0],
                                         self.patch_size[1], 3)).astype(np.float32)
                patch -= self.mean
            samples[it, ...] = patch
        return samples

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # assign output
        x, y = next(self.generator.flow(
            self.samples, self.labels, batch_size=self.batch_size))
        x = x.transpose((0, 3, 1, 2))
        top[0].data[...] = x
        top[1].data[...] = y[..., np.newaxis]

    def backward(self, top, propagate_down, bottom):
        pass

    def imread(self, path):
        im = imread(path)
        im = np.array(im, dtype=np.float32)
        im = im[:, :, ::-1]
        # im -= self.mean
        return im
