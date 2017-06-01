from ba import BA_ROOT
from ba.set import SetList
import ba.utils
from ba.utils import grouper
import caffe
import copy
import datetime
import numpy as np
import os
from os.path import normpath
import random
import re
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import skimage
import subprocess
import time
import lmdb
from tqdm import tqdm


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
        specvals = sorted(self.__dict__.items(), key=lambda x: x[0])
        with open(normpath(self._dir + '/solver.prototxt'), 'w') as f:
            for key, value in specvals:
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
    buildroot = BA_ROOT + 'data/models/'
    resultroot = BA_ROOT + 'data/results/'
    resultDB = BA_ROOT + 'data/results/experimentDB.yaml'

    def __init__(self, name, **kwargs):
        '''Constructs a new NetRunner

        Args:
            name (str): The name of this network (for save paths etc..)
            kwargs...
        '''
        super().__init__(**kwargs)
        self.name = name
        self._solver_attr = {}
        self.stride = 25
        self.kernel_size = 50
        defaults = {
            'batch_size': 1,
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
            'meanarray': None,
            'quiet': False
            }
        self.__dict__.update(defaults)
        for (attr, value) in kwargs.items():
            if attr in defaults:
                setattr(self, attr, value)

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
        pass
        # if self.net is not None:
        #    del self.net
        # if self.solver is not None:
        #    del self.solver
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

    def load_img(self, path, mean=False):
        '''Loads an image and prepares it for caffe.

        Args:
            path (str): The path to the image
            mean (list): The mean pixel of the dataset

        Returns:
            A tuple made of the input data and the image
        '''
        if path[-3:] == 'npy':
            data = np.load(path)
            return (data, False)
        else:
            im = imread(path)
            if im.ndim == 2:
                w, h = im.shape
                _im = np.empty((w, h, 3), dtype=np.float32)
                _im[:, :, 0] = im
                _im[:, :, 1] = im
                _im[:, :, 2] = im
                im = _im
            else:
                im = np.array(im, dtype=np.float32)
            data = im[:, :, ::-1]
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
            The results the data of the first output
        '''
        if data.ndim == 3:
            self.net.blobs['data'].reshape(1, *data.shape)
        else:
            self.net.blobs['data'].reshape(*data.shape)
        self.net.blobs['data'].data[...] = data

        # run net and take argmax for prediction
        self.net.forward()
        return self.net.blobs[self.net.outputs[0]].data

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

    def prepare(self):
        '''Prepares the enviroment for phases.'''
        # Create build and snapshot direcotry:
        ba.utils.touch(self.snapshots)
        self.trainset.target = self.dir + 'train.txt'
        self.trainset.write()
        self.write('train')
        self.write_solver()

        self.valset.target = self.dir + 'test.txt'
        self.valset.write()
        self.write('val')

        ba.utils.touch(self.heatmaps)
        ba.utils.touch(self.heatmaps[:-1] + '_overlays/')
        self.write('deploy')

    def train(self):
        '''Trains the network'''
        self.prepare()
        logf = '{}_{}_train.log'.format(
            datetime.datetime.now().strftime('%y_%m_%d_'), self.name)
        ba.utils.touch(self.dir + logf, clear=True)
        if not self.quiet:
            self.LOGNotifiy(self.dir + logf)
        return_code = subprocess.call(
            'caffe train -solver {} -weights {} -gpu {} 2>&1 | tee {}'.format(
                self.dir + 'solver.prototxt',
                self.solver_weights,
                ','.join(str(x) for x in self.gpu),
                self.dir + logf), shell=True)
        if return_code >= 0 and not self.quiet:
            self.notifier._send_telegram_photo(self.notifier.lossgraph(logf),
                                               logf)
        return return_code

    def test(self, slicefile=None, **kwargs):
        '''Will test the net.

        Args:
            slicefile (str, optional): The path for the seg.yaml, if given
                will perform SelecSearch and BB errors..
        '''
        import ba.eval
        scores_path = self.forward_test(**kwargs)
        if slicefile is not None:
            ba.eval.evalDect(scores_path, slicefile)

    def forward_batch(self, path_batch, mean=None):
        datas = []
        max_h = 500
        max_w = 500
        for path in path_batch:
            if path is None:
                continue
            data, im = self.load_img(path, mean=mean)
            # ADAPTIVE VERSION
            if data.shape[1] > max_h:
                max_h = data.shape[1]
            if data.shape[2] > max_w:
                max_w = data.shape[2]
            datas.append(data)

        bs = len(datas)
        self.net.blobs['data'].reshape(bs, 3, max_h, max_w)
        for i, data in enumerate(datas):
            shape = data.shape[1:]
            self.net.blobs['data'].data[i, :, 0:shape[0], 0:shape[1]] = data
        scores = self.net.forward()
        scores = scores[:, 1, ...]
        scoreboxes = {}
        for path, score, data in zip(path_batch, scores, datas):
            bn = os.path.basename(os.path.splitext(path)[0])
            regions, rscores = self._postprocess_single_output(bn, score,
                                                               data.shape[1:])
            scoreboxes[bn] = {'region': regions, 'score': rscores}
        return scoreboxes

    def forward_single(self, path, mean=None):
        '''Will forward one single path-image from the source set and saves the
        scoring heatmaps and heatmaps to disk.

        Args:
            path (str): The path of the image to forward
            mean (tuple, optional): The mean

        Return:
            The score and coordinates of the highest scoring region
        '''
        if mean is None:
            mean, meanpath = self.get_mean()
        data, im = self.load_img(path, mean=mean)
        score = self.forward(data)
        score = score[0][1, ...]
        bn = os.path.basename(os.path.splitext(path)[0])
        regions, rscores = self._postprocess_single_output(bn, score,
                                                           data.shape[1:])
        return {bn: {'region': regions, 'score': rscores}}

    def _postprocess_single_output(self, bn, score, imshape):
        # bn_hm = self.heatmaps + bn
        # imsave(bn_hm + '.png', score)
        # score = imresize(score, imshape)
        # score = skimage.img_as_float(score)
        # bn_ov = self.heatmaps[:-1] + '_overlays/' + bn
        # ba.plt.apply_overlay(im, score, bn_ov + '.png')
        upscore = np.zeros(imshape, dtype=float)
        # ONLY WORKS AS SUCH WITH ResNet 50
        score = imresize(score, 32.0)
        x_stop = min(upscore.shape[0], score.shape[0])
        y_stop = min(upscore.shape[1], score.shape[1])
        upscore[0:x_stop, 0:y_stop] = score[0:x_stop, 0:y_stop]
        regions, rscores = ba.eval.scoreToRegion(upscore)
        return regions, rscores

    def forward_list(self, setlist, reset_net=True, shout=False):
        '''Will forward a whole setlist through the network. Will default to the
        validation set.

        Args:
            setlist (SetList): The set to put forward

        Returns:
            the filename of the ****scores.yaml File
        '''
        import ba.eval

        def append_finds(res):
            with open(BA_ROOT + 'current_finds.csv', 'a') as f:
                for bn, fd in res.items():
                    if len(fd['score']) == 0:
                        continue
                    if fd['score'].max() > 0.95:
                        region = fd['region'][np.argmax(fd['score'])]
                        score = fd['score'].max()
                        f.write('{};{};{}\n'.format(bn, score, region))
                    # Save all good patches:
                    # for region, score in zip(fd['region'], fd['score']):
                    #    if score > 0.98:
                    #        f.write('{};{};{}\n'.format(bn, score, region))

        def save_scoreboxes(scores_path, scoreboxes):
            for boxdict in scoreboxes.values():
                boxdict['region'] = boxdict['region'].tolist()
                boxdict['score'] = boxdict['score'].tolist()
            ba.utils.save(scores_path, scoreboxes)

        def forward_batch(x):
            return self.forward_batch(x, mean=mean)

        def forward_single(x):
            return self.forward_single(x[0], mean=mean)

        self.prepare()
        if reset_net:
            self.create_net(self.dir + 'deploy.prototxt',
                            self.net_weights,
                            self.gpu[0])
        mean, meanpath = self.get_mean()
        scoreboxes = {}
        tstr = time.strftime('%b%d_%H:%M_', time.localtime())
        path_split = os.path.split(os.path.normpath(self.results))
        scores_path = '{}/{}{}.scores.yaml'.format(path_split[0], tstr,
                                                   path_split[1])
        weightname = os.path.splitext(os.path.basename(self.net_weights))[0]

        if self.batch_size > 1:
            forward = forward_batch
        else:
            forward = forward_single

        ba.utils.rm(BA_ROOT + 'current_finds.csv')

        print('Forwarding for {} at {} list {}'.format(
            self.name, weightname, setlist.source))
        for idx in grouper(tqdm(setlist), self.batch_size, None):
            res = forward(idx)
            if res is not False:
                scoreboxes.update(res)
                if shout:
                    append_finds(res)
        save_scoreboxes(scores_path, scoreboxes)
        if not self.quiet:
            self.notify('Forwarded {} for weights {} of {}'.format(
                setlist.source, weightname, self.name))
        return scores_path

    def forward_val(self, **kwargs):
        '''Will forward the whole validation set through the network.'''
        imgext = '.' + ba.utils.prevalent_extension(self.images)
        self.valset.add_pre_suffix(self.images, imgext)
        self.forward_list(setlist=self.valset, **kwargs)
        self.valset.rm_pre_suffix(self.images, imgext)

    def forward_test(self, **kwargs):
        '''Will forward the whole validation set through the network.

        Returns:
            the filename of the ****scores.yaml File
        '''
        imgext = '.' + ba.utils.prevalent_extension(self.images)
        self.testset.add_pre_suffix(self.images, imgext)
        random.shuffle(self.testset.list)
        scores_path = self.forward_list(setlist=self.testset, **kwargs)
        self.testset.rm_pre_suffix(self.images, imgext)
        return scores_path

    def outputs_to_lmdb(self, setlist=None, maxlength=800, **kwargs):
        if setlist is None:
            setlist = self.testset

        db_path = os.path.splitext(setlist.source)[0] + '_lmdb'
        map_size = 2048 * 25 * 25 * 2 * len(setlist)
        env = lmdb.Environment(db_path, map_size=map_size)
        txn = env.begin(write=True, buffers=True)

        mean, meanpath = self.get_mean()
        for idx in tqdm(setlist):
            inputs, im = self.load_img(idx, mean=mean)
            if inputs.shape[1] > maxlength:
                inputs = imresize(inputs, (3, maxlength,
                                           int(maxlength / inputs.shape[2])))
            if inputs.shape[2] > maxlength:
                inputs = imresize(inputs, (3, int(maxlength / inputs.shape[1],
                                           maxlength)))
            outputs = self.forward(inputs)


class SlidingFCNPartRunner(NetRunner):
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
            padded_data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)),
                                 mode='reflect')
            padded_hm = np.zeros(padded_data.shape[:-1])
            for (x1, x2, window) in ba.utils.sliding_window(padded_data,
                                                            self.stride, ks):
                score = self.forward_window(window)
                padded_hm[x1:x1 + ks, x2:x2 + ks] += score
            hm += padded_hm[pad:-pad, pad:-pad]

        imsave(bn_hm + '.png', hm)
        hm = imresize(hm, im.shape[:-1])
        hm = skimage.img_as_float(hm)
        ba.utils.apply_overlay(im, hm, bn_ov + '.png')
        region, rscore = ba.eval.scoreToRegion(hm, im)
        return {bn: {'region': list(region), 'score': float(rscore)}}
