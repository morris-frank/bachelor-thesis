'''
    Based on code from Evan Shelhamer
    fcn.berkeleyvision.org
'''
import ba.utils
import caffe
from itertools import count
from glob import glob
import numpy as np
from PIL import Image
import random
from scipy.misc import imread
import skimage.transform as tf
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
        from keras.preprocessing.image import ImageDataGenerator
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
            'data/tmp/var_neg/')
        if isinstance(params['mean'], str):
            self.mean = np.load(params['mean'])
        else:
            self.mean = np.array(params['mean'])

        # Load slices
        if not isinstance(self.slicefile, list):
            self.slicefile = [self.slicefile]
        self.slices = []
        n = 0
        for slicefile in self.slicefile:
            add_slices, n_slices = self.load_slices(self.splitfile, slicefile)
            n += n_slices
            if add_slices != {}:
                self.slices.append(add_slices)

        if self.ppI is None:
            if n >= 100:
                self.ppI = 2
            elif n >= 50:
                self.ppI = 4
            else:
                self.ppI = 16

        self.samples = np.zeros((n * self.ppI * 2, 3,
                                 self.patch_size[0], self.patch_size[1]))
        it = 0
        for slices in self.slices:
            for path, bblist in slices.items():
                im = self.imread('{}{}.{}'.format(self.images, path, self.ext))
                for bb in bblist:
                    subslice = slice(it, n * self.ppI, n)
                    self.samples[subslice, ...] = self.generate_samples(im, bb)
                    it += 1
        self.labels = np.append(np.ones(n * self.ppI),
                                np.zeros(n * self.ppI))
        negs = glob(self.negatives + '/*png')
        negs = random.sample(negs, n * self.ppI)
        for it, neg in zip(range(1, len(negs) + 1), negs):
            im = self.imread(neg)
            im /= 255
            im = tf.resize(im, (self.patch_size[0], self.patch_size[1], 3),
                           mode='reflect')
            im *= 255
            im -= self.mean
            self.samples[-it, ...] = im.transpose((2, 0, 1))

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.flow = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            data_format='channels_first'
            ).flow(self.samples, self.labels, batch_size=self.batch_size)

    def load_slices(self, splitfile, slicefile):
        with open(splitfile, 'r') as f:
            imlist = [l[:-1] for l in f.readlines() if l.strip()]
        slicelist = ba.utils.load(slicefile)
        slice_dict = {im: slicelist[im] for im in imlist if im in slicelist}
        n_slices = sum([len(sl) for sl in slice_dict.values()])
        return slice_dict, n_slices

    def bounding_box_shape(self, bb):
        return (bb[0].stop - bb[0].start,
                bb[1].stop - bb[1].start)

    def generate_samples(self, im, bb, shiftfactor=0.25, nsamples=None):
        if nsamples is None:
            nsamples = self.ppI
        if nsamples % 2 != 0:
            raise ValueError('Count in SingleImageLayer is not divisble by 2.')
        bbshape = self.bounding_box_shape(bb)
        padded_im = np.pad(im, ((bbshape[0], bbshape[0]),
                                (bbshape[1], bbshape[1]), (0, 0),),
                           mode='reflect')

        rands = (2 * np.random.random((2, nsamples)) - 1) * shiftfactor + 1
        xsamples = (bbshape[0] * rands[0]).astype(np.int)
        ysamples = (bbshape[1] * rands[1]).astype(np.int)

        samples = np.zeros((nsamples, 3,
                            self.patch_size[0], self.patch_size[1]))
        for it, xsample, ysample in zip(count(), xsamples, ysamples):
            _x = [bb[0].start, bb[0].stop] + xsample
            _y = [bb[1].start, bb[1].stop] + ysample
            patch = np.copy(padded_im[_x[0]:_x[1], _y[0]:_y[1], :])
            patch /= 255
            sized_patch = tf.resize(patch, (self.patch_size[0],
                                            self.patch_size[1], 3))
            sized_patch *= 255
            sized_patch -= self.mean
            samples[it, ...] = sized_patch.transpose((2, 0, 1))
        return samples

    def reshape(self, bottom, top):
        top[0].reshape(
            self.batch_size, 3, self.patch_size[0], self.patch_size[1])
        top[1].reshape(self.batch_size, 1)

    def forward(self, bottom, top):
        # assign output
        x, y = next(self.flow)
        filled = x.shape[0]
        top[0].data[:filled, ...] = x
        top[1].data[:filled, ...] = y[..., np.newaxis]

        while filled < self.batch_size:
            x, y = next(self.flow)
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

    def imread(self, path):
        im = imread(path)
        im = np.array(im, dtype=np.float32)
        im = im[:, :, ::-1]
        return im
