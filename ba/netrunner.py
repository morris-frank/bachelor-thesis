import ba.eval
from ba import caffeine
from ba.set import SetList
import ba.utils
import caffe
import copy
import datetime
import numpy as np
import os
from os.path import normpath
from PIL import Image
import re
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import skimage
import subprocess
import sys
from tqdm import tqdm
import yaml


class SolverSpec(ba.utils.Bunch):

    def __init__(self, dir, adict={}):
        '''Constructs a new SolverSpec object.

        Args:
            dir (str): The filepath of the model directory
            adict (dict, optional):  Optional values already set.
        '''
        self._dir = dir
        self.train_net = normpath(self._dir + '/train.prototxt')
        self.test_net = normpath(self._dir + '/val.prototxt')
        self.snapshot = 500
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
        self.max_iter = 2000
        self.test_iter = 100
        self.test_interval = 5000
        super().__init__(adict)

    def write(self):
        '''Writes this SolverSpec to the disc at the path set at self.target.

        '''
        with open(normpath(self._dir + '/solver.prototxt'), 'w') as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                if not key.startswith('_'):
                    if isinstance(value, str):
                        if re.match('[0-9]+e[+-]?[0-9]+', value) is not None:
                            f.write('{}: {}\n'.format(key, float(value)))
                        else:
                            f.write('{}: "{}"\n'.format(key, value))
                    else:
                        f.write('{}: {}\n'.format(key, value))


class NetRunner(ba.utils.NotifierClass):
    '''A Wrapper for a caffe Network'''
    buildroot = 'data/models/'
    resultroot = 'data/results/'
    resultDB = 'data/results/experimentDB.yaml'

    def __init__(self, name, **kwargs):
        '''Constructs a new NetRunner

        Args:
            name (str): The name of this network (for save paths etc..)
            kwargs...
        '''
        super().__init__(**kwargs)
        self.name = name
        self._solver_attr = {}
        defaults = {
            'dir': './',
            'generator_switches': {},
            'generator_attr': {},
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
            'valset': '',
            'meanarray': None
            }
        self.__dict__.update(defaults)
        for (attr, value) in kwargs.items():
            if attr in defaults:
                setattr(self, attr, value)

    @property
    def trainset(self):
        return self.__trainset

    @trainset.setter
    def trainset(self, trainset):
        if isinstance(trainset, SetList):
            self.__trainset = trainset
        else:
            self.__trainset = SetList(trainset)

    @property
    def testset(self):
        return self.__testset

    @testset.setter
    def testset(self, testset):
        if isinstance(testset, SetList):
            self.__testset = testset
        else:
            self.__testset = SetList(testset)

    @property
    def valset(self):
        return self.__valset

    @valset.setter
    def valset(self, valset):
        if isinstance(valset, SetList):
            self.__valset = valset
        else:
            self.__valset = SetList(valset)

    def clear(self):
        '''Clears the nets and joins notifier threads'''
        if self.net is not None:
            del self.net
        if self.solver is not None:
            del self.solver
        if len(self.notifier_threads) > 0:
            for thread in self.notifier_threads:
                thread.join(timeout=1.0)

    def create_net(self, model, weights, gpu):
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

    def create_solver(self, solverpath, weights, gpu):
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

    def load_img(self, path, mean):
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
        if not isinstance(mean, bool) or mean:
            if mean.shape == (224, 224, 3):
                mean = imresize(mean, data.shape)
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
            imgext = '.' + ba.utils.prevalent_extension(self.images)
            self.trainset.add_pre_suffix(self.images, imgext)
            self.trainset.calculate_mean()
            self.trainset.rm_pre_suffix(self.images, imgext)
            np.save(self.dir + 'mean.npy', self.trainset.mean)
        return self.trainset.mean

    def get_mean(self):
        '''Returns the mean for this NetRunner. If we have a train set with a
        mean that is returned otherwise we try to import the mean saved from
        the training phase.

        Returns:
            the mean
        '''
        if self.mean != []:
            if self.meanarray is not None:
                return self.meanarray, self.mean
            elif isinstance(self.mean, str) and os.path.isfile(self.mean):
                self.meanarray = np.load(self.mean)
                return self.meanarray, self.mean
            else:
                return self.mean, False
        elif self.trainset.mean != []:
            print('Using mean from samples')
            return self.trainset.mean, False
        else:
            return self.calculate_mean(), False


class FCNPartRunner(NetRunner):
    '''A NetRunner specific for FCNs'''

    def __init__(self, name, **kwargs):
        '''Constructs a new FCNPartRunner

        Args:
            name (str): The name of this network (for saving paths...)
            kwargs...
        '''
        super().__init__(name=name, **kwargs)
        self.name = name
        self.images = 'data/datasets/voc2010/JPEGImages/'

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        '''Sets the name of this network and all Variables that depend on it

        Args:
            name (str): The new name
        '''
        self.__name = name
        self.dir = self.buildroot + '/' + self.name + '/'
        self.results = self.resultroot + '/' + self.name + '/'
        self.snapshots = self.dir + 'snapshots/'
        self.heatmaps = self.results + 'heatmaps/'

    @property
    def net_weights(self):
        return self.__net_weights

    @net_weights.setter
    def net_weights(self, net_weights):
        self.__net_weights = net_weights
        bnw = os.path.basename(net_weights[:-len('.caffemodel')])
        self.results = '/'.join([self.resultroot, self.name, bnw]) + '/'
        self.heatmaps = self.results + 'heatmaps/'

    def generator_params(self, split):
        '''Builds the dict for a net_generator.

        Args:
            split (str): The split (test|train|deploy)

        Returns:
            The parameter dictionary
        '''
        splitfile = self.dir
        splitfile += 'train.txt' if split == 'train' else 'test.txt'
        imgext = ba.utils.prevalent_extension(self.images)
        mean, path = self.get_mean()
        if path:
            mean = path
        elif not isinstance(mean, bool):
            mean = tuple(mean)
        params = dict(
            images=self.images,
            extension=imgext,
            labels=self.labels,
            splitfile=splitfile,
            split=split,
            mean=mean
            )
        params.update(self.generator_attr)
        return params

    def write(self, split):
        '''Writes the model file for one split to disk

        Args:
            split (str): The split (test|train|deploy)
        '''
        ba.utils.touch(self.dir)
        with open(self.dir + split + '.prototxt', 'w') as f:
            f.write(str(self.net_generator(self.generator_params(split),
                                           self.generator_switches)))

    def write_solver(self):
        '''Writes the solver definition to disk.'''
        s = SolverSpec(self.dir, self._solver_attr)
        s.write()

    def prepare(self, split='train_test_deploy'):
        '''Prepares the enviroment for phases.

        Args:
            split (str): The split to prepare for.
        '''
        # Create build and snapshot direcotry:
        if 'train' in split:
            ba.utils.touch(self.snapshots)
            self.trainset.target = self.dir + 'train.txt'
            # self.trainset.write()
            self.write('train')
            self.write_solver()
        if 'train' in split or 'test' in split:
            self.valset.target = self.dir + 'test.txt'
            self.valset.write()
            self.write('val')
        if 'deploy' in split:
            ba.utils.touch(self.heatmaps)
            ba.utils.touch(self.heatmaps[:-1] + '_overlays/')
            self.write('deploy')

    def train(self):
        '''Trains the network'''
        self.prepare('train')
        logf = '{}_{}_train.log'.format(
            datetime.datetime.now().strftime('%y_%m_%d_'), self.name)
        ba.utils.touch(self.dir + logf, clear=True)
        self.LOGNotifiy(self.dir + logf)
        return_code = subprocess.call(
            'caffe train -solver {} -weights {} -gpu {} 2>&1 | tee {}'.format(
            self.dir + 'solver.prototxt',
            self.solver_weights,
            ','.join(str(x) for x in self.gpu),
            self.dir + logf), shell=True)
        if return_code >= 0:
            self.notifier._send_telegram_photo(self.notifier.lossgraph(logf), logf)
        return return_code

    def test(self, slicefile=None):
        '''Will test the net.

        Args:
            slicefile (str, optional): The path for the seg.yaml, if given
                will perform SelecSearch and BB errors..
        '''
        self.prepare('deploy')
        self.forward_test()
        scoreboxf = self.results[:-1] + '.scores.yaml'
        weightname = os.path.splitext(os.path.basename(self.net_weights))[0]
        iteration = weightname.split('_')[-1]
        if slicefile is not None:
            db = ba.utils.load_YAML(self.resultDB)
            if db is None:
                db = {}
            if self.name not in db:
                db[self.name] = {}
            ofile = ba.eval.evalYAML(
                scoreboxf, slicefile, self.images, self.heatmaps)
            # ofile = '.'.join(scoreboxf.split('.')[:-2] + ['evals', 'yaml'])
            trainset = ba.SetList(self.dir + 'train.txt')
            _soltestset = list(self.testset.set - trainset.set)
            if trainset is not None and self.testset is not None:
                v_iou, v_dist, v_scal, v_n = ba.eval.extract_mean_evals(
                    _soltestset, ofile)
                v_mat = [['Network', self.name],
                         ['Weights', weightname],
                         ['Datums', v_n],
                         ['MeanIOU', v_iou],
                         ['MeanDistErr', v_dist],
                         ['MeanScalErr', v_scal]]
                db[self.name][str(iteration + '_test')] = v_mat
                self.notify(matrix=v_mat)
            if trainset is not None and len(trainset.list) > 0:
                t_iou, t_dist, t_scal, t_n = ba.eval.extract_mean_evals(
                    trainset.list, ofile)
                t_mat = [['Network', self.name],
                         ['Weights', weightname],
                         ['Datums', t_n],
                         ['MeanIOU', t_iou],
                         ['MeanDistErr', t_dist],
                         ['MeanScalErr', t_scal]]
                db[self.name][str(iteration + '_train')] = t_mat
            with open(self.resultDB, 'w') as f:
                yaml.dump(db, f)

    def forward_single(self, idx, mean=None):
        '''Will forward one single idx-image from the source set and saves the
        scoring heatmaps and heatmaps to disk.

        Args:
            idx (str): The index (basename) of the image to forward
            mean (tuple, optional): The mean

        Return:
            The score and coordinates of the highest scoring region
        '''
        if mean is None:
            mean, meanpath = self.get_mean()
        data, im = self.load_img(idx, mean=mean)
        bn = os.path.basename(os.path.splitext(idx)[0])
        bn_hm = self.heatmaps + bn
        bn_ov = self.heatmaps[:-1] + '_overlays/' + bn
        self.forward(data)
        score = self.net.blobs[self.net.outputs[0]].data[0][1, ...]
        imsave(bn_hm + '.png', score)
        score = imresize(score, im.shape[:-1])
        score = skimage.img_as_float(score)
        ba.utils.apply_overlay(im, score, bn_ov + '.png')
        region, rscore = ba.eval.scoreToRegion(score, im)
        return {bn: {'region': list(region), 'score': float(rscore)}}

    def forward_list(self, setlist=None):
        '''Will forward a whole setlist through the network. Will default to the
        validation set.

        Args:
            setlist (SetList, optional): The set to put forward
        '''
        if setlist is None:
            return self.forward_test()
        self.prepare('deploy')
        self.create_net(self.dir + 'deploy.prototxt',
                       self.net_weights,
                       self.gpu[0])
        mean, meanpath = self.get_mean()
        scoreboxes = {}
        scoreboxf = self.results[:-1] + '.scores.yaml'
        weightname = os.path.splitext(os.path.basename(self.net_weights))[0]
        print('Forwarding all in {}'.format(setlist))
        for i, idx in enumerate(tqdm(setlist)):
            scoreboxes.update(self.forward_single(idx, mean=mean))
            if i % 10 == 0:
                with open(scoreboxf, 'w') as f:
                    yaml.dump(scoreboxes, f)
        self.notify('Forwarded {} for weights {} of {}'.format(setlist.source,
                                                               weightname,
                                                               self.name))

    def forward_val(self):
        '''Will forward the whole validation set through the network.'''
        imgext = '.' + ba.utils.prevalent_extension(self.images)
        self.valset.add_pre_suffix(self.images, imgext)
        self.forward_list(setlist=self.valset)
        self.valset.rm_pre_suffix(self.images, imgext)

    def forward_test(self):
        '''Will forward the whole validation set through the network.'''
        imgext = '.' + ba.utils.prevalent_extension(self.images)
        self.testset.add_pre_suffix(self.images, imgext)
        self.forward_list(setlist=self.testset)
        self.testset.rm_pre_suffix(self.images, imgext)


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

    def generator_params(self, split):
        '''Builds the dict for a net_generator.

        Args:
            split (str): The split (test|train|deploy)

        Returns:
            The parameter dictionary
        '''
        params = super().generator_params(split)
        if split == 'train':
            params['lmdb'] = self.trainset.source[:-4]
        elif split == 'val':
            params['lmdb'] = self.valset.source[:-4]
        return params

    def forward_window(self, window):
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

    def forward_single(self, idx, mean=None):
        '''Will slide a window over the idx-image from the source and forward
        that slice through the network. Saves the scoring heatmaps and heatmaps
        to disk.

        Args:
            idx (str): The index (basename) of the image to forward
            mean (tuple, optional): The mean

        Return:
            The score and coordinates of the highest scoring region
        '''
        if mean is None:
            mean, meanpath = self.get_mean()
        data, im = self.load_img(idx, mean=mean)
        data = data.transpose((1, 2, 0))
        hm = np.zeros(data.shape[:-1])
        bn = os.path.basename(os.path.splitext(idx)[0])
        bn_hm = self.heatmaps + bn
        bn_ov = self.heatmaps[:-1] + '_overlays/' + bn
        for ks in [50, 100, 250]:
            pad = int(ks)
            padded_data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            padded_hm = np.zeros(padded_data.shape[:-1])
            for (x, y, window) in ba.utils.sliding_window(padded_data, self.stride,
                                                       ks):
                score = self.forward_window(window)
                padded_hm[y:y + ks, x:x + ks] += score
            hm += padded_hm[pad:-pad, pad:-pad]

        imsave(bn_hm + '.png', hm)
        hm = imresize(hm, im.shape[:-1])
        hm = skimage.img_as_float(hm)
        ba.utils.apply_overlay(im, hm, bn_ov + '.png')
        region, rscore = ba.eval.scoreToRegion(hm, im)
        return {bn: {'region': list(region), 'score': float(rscore)}}
