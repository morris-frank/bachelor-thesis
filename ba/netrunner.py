from ba.set import SetList
from PIL import Image
from scipy.misc import imsave
from scipy.misc import imread
from tqdm import tqdm
from ba import caffeine
import caffe
from collections import namedtuple
import copy
import numpy as np
import os
import sys
from functools import partial
import tempfile
from ba import utils
import warnings


class NetRunner(object):
    '''A Wrapper for a caffe Network'''

    def __init__(self, name, **kwargs):
        '''Constructs a new NetRunner

        Args:
            name (str): The name of this network (for save paths etc..)
            kwargs...
        '''
        defaults = {
            'epochs': 1,
            'baselr': 1,
            'solver': None,
            'net': None,
            'test': '',
            'train': '',
            'val': '',
            'images': './',
            'labels': './',
            'dir': './',
            'results': './',
            'random': True,
            'net_weights': '',
            'solver_weights': ''
            }
        for (attr, default) in defaults.items():
            setattr(self, attr, kwargs.get(attr, default))

        self.name = name

        if not isinstance(self.train, SetList):
            self.train = SetList(self.train)
        if not isinstance(self.test, SetList):
            self.test = SetList(self.test)
        if not isinstance(self.val, SetList):
            self.val = SetList(self.val)

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
            self.test = SetList(fpath)
        if 'train' in split:
            self.train = SetList(fpath)

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
        '''Splits the train list self.train into samples that we will train on
        and sample to do validation. Passing 0 will just copy the list.

        Args:
            count (int, optional): The count of samples
        '''
        if self.random:
            self.train.shuffle()
        self.val = copy.copy(self.train)
        if count < len(self.train) and count > 0:
            self.val.list = self.train.list[count:]
            self.train.list = self.train.list[:count]

    def calculate_mean(self):
        '''Calculates the mean for the training set. As that is necessary for
        the bias of the training phase.'''
        try:
            self.train.mean = np.load(self.dir + 'mean.npy')
        except FileNotFoundError:
            imgext = '.' + utils.prevalentExtension(self.images)
            self.train.addPreSuffix(self.images, imgext)
            self.train.calculate_mean()
            self.train.rmPreSuffix(self.images, imgext)
            np.save(self.dir + 'mean.npy', self.train.mean)

    def mean(self):
        '''Returns the mean for this NetRunner. If we have a train set with a
        mean that is returned otherwise we try to import the mean saved from
        the training phase.

        Returns:
            the mean
        '''
        if self.train.mean != []:
            print('Using mean from samples')
            return self.train.mean
        else:
            print('Loading mean from training phase')
            return np.load(self.dir + 'mean.npy')


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
        self.net_generator = caffeine.fcn.fcn8s
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
        imgext = '.' + utils.prevalentExtension(self.images)
        params = {'img_dir': self.images,
                  'img_ext': imgext,
                  'label_dir': self.labels,
                  'splitfile': self.dir + split + '.txt'
                  }
        self.calculate_mean()
        params['mean'] = tuple(self.train.mean)
        return params

    def write(self, split):
        '''Writes the model file for one split to disk

        Args:
            split (str): The split (test|train|deploy)
        '''
        with open(self.dir + split + '.prototxt', 'w') as f:
            f.write(str(self.net_generator(self.FCNparams(split))))

    def prepare(self, split='train_test_deploy'):
        '''Prepares the enviroment for phases.

        Args:
            split (str): The split to prepare for.
        '''
        self.targets()
        # Create build and snapshot direcotry:
        if 'train' in split:
            utils.touch(self.snapshots)
            self.train.target = self.dir + 'train.txt'
            self.train.save()
            self.write('train')
        if 'train' in split or 'test' in split:
            self.val.target = self.dir + 'test.txt'
            self.val.save()
            self.write('val')
        if 'deploy' in split:
            bnw = os.path.basename(self.net_weights[:-len('.caffemodel')])
            self.heatmaps = self.results + bnw + '/' + 'heatmaps/'
            utils.touch(self.heatmaps)
            utils.touch(self.heatmaps[:-1] + '_overlays/')
            self.write('deploy')

    def writeSolver(self):
        '''Writes the solver definition to disk.'''
        train_iter = str(len(self.train))
        train_net = self.dir + 'train.prototxt'
        val_net = self.dir + 'val.prototxt'
        prefix = self.snapshots + 'train'
        avloss = str(np.min([20, len(self.train)]))
        with open(self.dir + 'solver.prototxt', 'w') as f:
            f.write((
                "train_net: '" + train_net + "'\n"
                "test_net: '" + val_net + "'\n"
                "test_iter: " + train_iter + "\n"
                "test_interval: " + str(999999) + "\n"
                "display: " + avloss + "\n"
                "average_loss: " + avloss + "\n"
                "lr_policy: 'fixed'\n"
                "base_lr: " + str(self.baselr) + "\n"
                "momentum: 0.99\n"
                "iter_size: 1\n"
                "max_iter: 100000\n"
                "weight_decay: 0.0005\n"
                "snapshot: " + train_iter + "\n"
                "snapshot_prefix: '" + prefix + "'\n"
                "test_initialization: false\n"
                ))

    def train(self):
        '''Will train the network and make snapshots accordingly'''
        self.prepare('train')
        self.writeSolver()
        self.createSolver(self.dir + 'solver.prototxt',
                          self.solver_weights,
                          self.gpu)
        # TODO: ONLY DO INTERP WHEN STARTING FROM ZERO
        interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
        caffeine.surgery.interp(self.solver.net, interp_layers)
        for _ in range(self.epochs):
            self.solver.step(len(self.train))
        self.solver.snapshot()

    def test(self):
        # TODO(doc): Add docstring
        self.prepare('test')

    def forwardVal(self):
        '''Will forward the whole validation set through the network'''
        imgext = '.' + utils.prevalentExtension(self.images)
        self.val.addPreSuffix(self.images, imgext)
        self.forwardList(setlist=self.val)
        self.val.rmPreSuffix(self.images, imgext)

    def forwardIDx(self, idx, mean=None):
        '''Will forward one single idx-image from the source set and saves the
        scoring heatmaps and heatmaps to disk.

        Args:
            idx (str): The index (basename) of the image to forward
            mean (tuple, optional): The mean
        '''
        if mean is None:
            mean = self.mean()
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
        # TODO(doc): Add docstring
        if setlist is None:
            return self.forwardVal()
        self.prepare('deploy')
        self.createNet(self.dir + 'deploy.prototxt',
                       self.net_weights,
                       self.gpu)
        mean = self.mean()
        print('Forwarding all in {}'.format(setlist))
        for idx in tqdm(setlist.list):
            self.forwardIDx(idx, mean=mean)
