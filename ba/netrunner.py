from ba import caffeine
from ba.set import SetList
from ba import utils
import caffe
from collections import namedtuple
import copy
import datetime
from functools import partial
import numpy as np
import os
from os.path import normpath
from PIL import Image
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import sys
import tempfile
from tqdm import tqdm
import warnings


class SolverSpec(utils.Bunch):

    def __init__(self, dir, adict={}):
        '''Constructs a new SolverSpec object.

        Args:
            dir (str): The filepath of the model directory
            adict (dict, optional):  Optional values already set.
        '''
        self._dir = dir
        self.train_net = normpath(self._dir + '/train.prototxt')
        self.test_net = normpath(self._dir + '/val.prototxt')
        self.snapshot = 1000
        self.snapshot_prefix = normpath(self._dir + '/snapshots/train')
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.display = 20
        self.average_loss = self.display
        self.base_lr = 0.001
        self.lr_policy = 'fixed'
        self.gamma = 0.1
        self.test_initialization = False
        self.iter_size = 1
        self.max_iter = 4000
        self.test_iter = 100
        self.test_interval = 500
        super().__init__(adict)

    def write(self):
        '''Writes this SolverSpec to the disc at the path set at self.target.

        '''
        with open(normpath(self._dir + '/solver.prototxt'), 'w') as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                if not key.startswith('_'):
                    if isinstance(value, str):
                        f.write('{}: "{}"\n'.format(key, value))
                    else:
                        f.write('{}: {}\n'.format(key, value))


class NetRunner(object):
    '''A Wrapper for a caffe Network'''

    def __init__(self, name, **kwargs):
        '''Constructs a new NetRunner

        Args:
            name (str): The name of this network (for save paths etc..)
            kwargs...
        '''
        self.name = name
        self._spattr = {}
        defaults = {
            'baselr': 1,
            'dir': './',
            'epochs': 1,
            'generator_switches': {},
            'images': './',
            'labels': './',
            'mean': [],
            'net_generator': None,
            'net_weights': '',
            'net': None,
            'random': True,
            'results': './',
            'solver_weights': '',
            'solver': None,
            'testset': '',
            'trainset': '',
            'valset': ''
            }
        self.__dict__.update(defaults)
        for (attr, value) in kwargs.items():
            if attr in defaults:
                setattr(self, attr, value)
            else:
                self._spattr[attr] = value

        if not isinstance(self.trainset, SetList):
            self.trainset = SetList(self.trainset)
        if not isinstance(self.testset, SetList):
            self.testset = SetList(self.testset)
        if not isinstance(self.valset, SetList):
            self.valset = SetList(self.valset)

    def clear(self):
        '''Clears the nets'''
        del self.net
        del self.solver

    def createNet(self, model, weights, gpu):
        '''Creates the net inside the runner.

        Args:
            model (str): The path to the model definition
            weights (str): The path to the weights
            gpu (int): The ID of the GPU to use
        '''
        self.net_weights = weights
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model, weights, caffe.TEST)

    def createSolver(self, solverpath, weights, gpu):
        '''Creates a Solver to Train a network

        Args:
            solverpath (str): The path to the solver definition file
            weights (str): The path to the weights
            gpu (int): The ID of the GPU to use
        '''
        self.solver_weights = weights
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.solver = caffe.SGDSolver(solverpath)
        self.solver.net.copy_from(weights)

    def addListFile(self, fpath, split='test_train'):
        '''Adds a list to run the net on

        Args:
            fpath (str): The path to the list file
        '''
        if 'test' in split:
            self.testset = SetList(fpath)
        if 'train' in split:
            self.trainset = SetList(fpath)

    def loadimg(self, path, mean):
        '''Loads an image and prepares it for caffe.

        Args:
            path (str): The path to the image
            mean (list): The mean pixel of the dataset

        Returns:
            A tuple made of the input data and the image
        '''
        # im = Image.open(path)
        # Seems faster in my opinion:
        im = imread(path)
        data = np.array(im, dtype=np.float32)
        data = data[:, :, ::-1]
        data -= np.array(mean)
        data = data.transpose((2, 0, 1))
        return (data, im)

    def forward(self, data):
        '''Forwards a loaded image through the network.

        Args:
            data (np.array): The preprocessed image

        Returns:
            The results from the score layer
        '''
        self.net.blobs['data'].reshape(1, *data.shape)
        self.net.blobs['data'].data[...] = data

        # run net and take argmax for prediction
        self.net.forward()
        return self.net.blobs[self.net.outputs[0]].data[0].argmax(axis=0)

    def split(self, count=0):
        '''Splits the train list self.trainset into samples that we will train on
        and sample to do validation. Passing 0 will just copy the list.

        Args:
            count (int, optional): The count of samples
        '''
        if self.random:
            self.trainset.shuffle()
        self.valset = copy.copy(self.trainset)
        if count < len(self.trainset) and count > 0:
            self.valset.list = self.trainset.list[count:]
            self.trainset.list = self.trainset.list[:count]

    def calculate_mean(self):
        '''Calculates the mean for the training set. As that is necessary for
        the bias of the training phase.

        Returns:
            The newly calculated mean for the trainset
        '''
        try:
            self.trainset.mean = np.load(self.dir + 'mean.npy')
        except FileNotFoundError:
            imgext = '.' + utils.prevalentExtension(self.images)
            self.trainset.addPreSuffix(self.images, imgext)
            self.trainset.calculate_mean()
            self.trainset.rmPreSuffix(self.images, imgext)
            np.save(self.dir + 'mean.npy', self.trainset.mean)
        return self.trainset.mean

    def getMean(self):
        '''Returns the mean for this NetRunner. If we have a train set with a
        mean that is returned otherwise we try to import the mean saved from
        the training phase.

        Returns:
            the mean
        '''
        if self.mean != []:
            return self.mean
        elif self.trainset.mean != []:
            print('Using mean from samples')
            return self.trainset.mean
        else:
            return self.calculate_mean()


class FCNPartRunner(NetRunner):
    '''A NetRunner specific for FCNs'''
    buildroot = 'data/models/'
    resultroot = 'data/results/'

    def __init__(self, name, **kwargs):
        '''Constructs a new FCNPartRunner

        Args:
            name (str): The name of this network (for saving paths...)
            kwargs...
        '''
        super().__init__(name=name, **kwargs)
        self.images = 'data/datasets/voc2010/JPEGImages/'
        self.targets()

    def targets(self):
        '''Building the saving paths from the current state.'''
        self.dir = self.buildroot + '/' + self.name + '/'
        self.snapshots = self.dir + 'snapshots/'
        self.results = self.resultroot + '/' + self.name + '/'
        self.heatmaps = self.results + 'heatmaps/'

    def FCNparams(self, split):
        '''Builds the dict for a net_generator.

        Args:
            split (str): The split (test|train|deploy)

        Returns:
            The parameter dictionary
        '''
        splitfile = self.dir
        splitfile += 'train.txt' if split == 'train' else 'test.txt'
        imgext = utils.prevalentExtension(self.images)
        params = dict(images=self.images,
                      extension=imgext,
                      labels=self.labels,
                      splitfile=splitfile,
                      split=split,
                      mean=tuple(self.getMean()))
        return params

    def write(self, split):
        '''Writes the model file for one split to disk

        Args:
            split (str): The split (test|train|deploy)
        '''
        with open(self.dir + split + '.prototxt', 'w') as f:
            f.write(str(self.net_generator(self.FCNparams(split),
                                           self.generator_switches)))

    def prepare(self, split='train_test_deploy'):
        '''Prepares the enviroment for phases.

        Args:
            split (str): The split to prepare for.
        '''
        self.targets()
        # Create build and snapshot direcotry:
        if 'train' in split:
            utils.touch(self.snapshots)
            self.trainset.target = self.dir + 'train.txt'
            self.trainset.write()
            self.write('train')
            self.writeSolver()
        if 'train' in split or 'test' in split:
            self.valset.target = self.dir + 'test.txt'
            self.valset.write()
            self.write('val')
        if 'deploy' in split:
            bnw = os.path.basename(self.net_weights[:-len('.caffemodel')])
            self.heatmaps = self.results + bnw + '/' + 'heatmaps/'
            utils.touch(self.heatmaps)
            utils.touch(self.heatmaps[:-1] + '_overlays/')
            self.write('deploy')

    def writeSolver(self):
        '''Writes the solver definition to disk.'''
        s = SolverSpec(self.dir, self._spattr)
        s.base_lr = float(self.baselr)
        # s.lr_policy = 'fixed'
        # # s.lr_policy = self.lr_policy
        s.write()

    def train(self):
        '''Will train the network and make snapshots accordingly'''
        self.prepare('train')
        self.createSolver(self.dir + 'solver.prototxt',
                          self.solver_weights,
                          self.gpu[0])
        # TODO: ONLY DO INTERP WHEN STARTING FROM ZERO
        interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
        caffeine.surgery.interp(self.solver.net, interp_layers)
        for _ in range(self.epochs):
            self.solver.step(len(self.trainset))
        self.solver.snapshot()

    def test(self):
        # TODO(doc): Add docstring
        self.prepare('test')

    def forwardVal(self):
        '''Will forward the whole validation set through the network'''
        imgext = '.' + utils.prevalentExtension(self.images)
        self.valset.addPreSuffix(self.images, imgext)
        self.forwardList(setlist=self.valset)
        self.valset.rmPreSuffix(self.images, imgext)

    def forwardIDx(self, idx, mean=None):
        '''Will forward one single idx-image from the source set and saves the
        scoring heatmaps and heatmaps to disk.

        Args:
            idx (str): The index (basename) of the image to forward
            mean (tuple, optional): The mean
        '''
        if mean is None:
            mean = self.getMean()
        data, im = self.loadimg(idx, mean=mean)
        self.forward(data)
        score = self.net.blobs[self.net.outputs[0]].data[0][1, ...]
        bn = os.path.basename(os.path.splitext(idx)[0])
        bn_hm = self.heatmaps + bn
        imsave(bn_hm + '.png', score)
        bn_ov = self.heatmaps[:-1] + '_overlays/' + bn
        utils.apply_overlay(im, score, bn_ov + '.png')

    def forwardList(self, setlist=None):
        '''Will forward a whole setlist through the network. Will default to the
        validation set.

        Args:
            setlist (SetList, optional): The set to put forward
        '''
        if setlist is None:
            return self.forwardVal()
        self.prepare('deploy')
        self.createNet(self.dir + 'deploy.prototxt',
                       self.net_weights,
                       self.gpu)
        mean = self.getMean()
        print('Forwarding all in {}'.format(setlist))
        for idx in tqdm(setlist):
            self.forwardIDx(idx, mean=mean)


class SlidingFCNPartRunner(FCNPartRunner):
    '''A subclass of FCNPartRunner that forwards images in a sliding window kind
    of way'''

    def __init__(self, name, **kwargs):
        '''Constructs a new FCNPartRunner

        Args:
            name (str): The name of this network (for saving paths...)
            kwargs...
        '''
        super().__init__(name=name, **kwargs)
        self.stride = 25
        self.kernel_size = 50

    def FCNparams(self, split):
        '''Builds the dict for a net_generator.

        Args:
            split (str): The split (test|train|deploy)

        Returns:
            The parameter dictionary
        '''
        params = super().FCNparams(split)
        params['batch_size'] = 10
        if split == 'train':
            params['lmdb'] = self.trainset.source[:-4]
        elif split == 'val':
            params['lmdb'] = self.valset.source[:-4]
        return params

    def train(self):
        '''Trains the network'''
        self.prepare('train')
        logf = '{}_{}_train.log'.format(
            datetime.datetime.now().strftime('%y_%m_%d_'), self.name)
        os.system('caffe train -solver {} -weights {} -gpu {} 2>&1 | tee {}'.format(
            self.dir + 'solver.prototxt',
            self.solver_weights,
            ','.join(str(x) for x in self.gpu),
            self.dir + logf))

    def forwardWindow(self, window):
        '''Forwards a single window from the sliding window through the network.

        Args:
            window (image): The window (channels shall be last dimension)

        Returns:
            the score for that window sized for the window
        '''
        inshape = window.shape[:-1]
        window = imresize(window, (224, 224, 3))
        window = window.transpose((2, 0, 1))
        self.forward(window)
        score = self.net.blobs[self.net.outputs[0]].data[0][1, ...]
        return score * np.ones(inshape)

    def forwardIDx(self, idx, mean=None):
        '''Will slide a window over the idx-image from the source and forward
        that slice through the network. Saves the scoring heatmaps and heatmaps
        to disk.

        Args:
            idx (str): The index (basename) of the image to forward
            mean (tuple, optional): The mean
        '''
        if mean is None:
            mean = self.getMean()
        data, im = self.loadimg(idx, mean=mean)
        data = data.transpose((1, 2, 0))
        hm = np.zeros(data.shape[:-1])
        for (x, y, window) in utils.sliding_window(data, self.stride,
                                                   self.kernel_size):
            score = self.forwardWindow(window)
            hm[y:y + self.kernel_size, x:x + self.kernel_size] += score
        bn = os.path.basename(os.path.splitext(idx)[0])
        bn_hm = self.heatmaps + bn
        imsave(bn_hm + '.png', hm)
        bn_ov = self.heatmaps[:-1] + '_overlays/' + bn
        utils.apply_overlay(im, hm, bn_ov + '.png')
