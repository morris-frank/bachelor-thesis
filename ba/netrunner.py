from ba.set import SetList
from PIL import Image
from scipy.misc import imsave
from tqdm import tqdm
import ba.caffeine
import caffe
import copy
import numpy as np
import os
import sys
from functools import partial
import tempfile
import ba.utils
import warnings


class NetRunner(object):
    """docstring for NetRunner."""

    def __init__(self):
        self.epochs = 1
        self.baselr = 1

    def createNet(self, model, weights, gpu):
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model, weights, caffe.TEST)

    def createSolver(self, solverpath, weights, gpu):
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.solver = caffe.SGDSolver(solverpath)
        solver.net.copy_from(weights)

    def addListFile(self, fpath):
        self.list = SetList(fpath)

    def loadimg(self, path, mean):
        im = Image.open(path)
        self.tmpim = im
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array(mean)
        in_ = in_.transpose((2, 0, 1))
        return in_

    def forward(self, in_):
        self.net.blobs['data'].reshape(1, *in_.shape)
        self.net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        self.net.forward()
        return self.net.blobs['score'].data[0].argmax(axis=0)


class FCNPartRunner(NetRunner):
    builddir = 'data/models/tmp/'
    results = 'data/results/'

    def __init__(self, tag, traintxt, valtxt, samples=0, random=True):
        super().__init__()
        self.name = tag
        self.random = random
        self.trainlist = SetList(traintxt)
        self.vallist = SetList(valtxt)
        self.samples = []
        self.net_generator = ba.caffeine.fcn.fcn8s
        self.imgdir = 'data/datasets/voc2010/JPEGImages/'
        self.weights = ''
        self.target = {}
        if self.random:
            self.vallist.shuffle()
        self.selectSamples(samples)

    def targets(self, builddir=None):
        if builddir is None:
            builddir = self.builddir
        self.target['dir'] = os.path.normpath(builddir + '/' + tag) + '/'
        self.target['outdir'] = os.path.normpath(self.results + '/' + tag) + '/'
        self.target['snapshots'] = self.target['dir'] + 'snapshots/'
        self.target['solver'] = self.target['dir'] + 'solver.prototxt/'
        self.target['train'] = self.target['dir'] + 'train.prototxt/'
        self.target['val'] = self.target['dir'] + 'val.prototxt/'
        self.target['deploy'] = self.target['dir'] + 'deploy.prototxt/'
        self.target['trainset'] = self.target['dir'] + 'train.txt/'
        self.target['valset'] = self.target['dir'] + 'val.txt/'
        self.target['segmentations'] = self.target['outdir'] + 'segmentations/'
        self.target['heatmaps'] = self.target['outdir'] + 'heatmaps/'
        return

    def selectSamples(self, count):
        if count > len(self.trainlist) or count < 1:
            warnings.Warning('More samples selected then possible...\n'
                             'Or less then one sample -> now doing all...')
            count = len(self.trainlist)
        self.samples = copy.copy(self.trainlist)
        if self.random:
            self.samples.shuffle()
        self.samples.list = self.samples.list[:count]

    def FCNparams(self, split):
        params = []
        params['img_dir'] = self.imgdir
        params['label_dir'] = self.labeldir
        params['splitfile'] = self.target[split + 'set']
        if not self.samples.mean:
            print('Calcultaing mean..')
            self.samples.calculate_mean()
        params['mean'] = self.samples.mean
        np.save(self.target['dir'] + 'mean.npy', self.samples.mean)
        return params

    def write(self, split):
        with open(self.target[split], 'w') as f:
            f.write(str(self.net_generator(self.FCNparams(split))))

    def prepare(self, split='train_test_deploy'):
        if self.target == {}:
            self.targets()
        # Create build and snapshot direcotry:
        ba.utils.touch(self.target['snapshots'])
        if 'train' in split:
            self.samples.target = self.target['trainset']
            self.samples.save()
            self.write('train')
        if 'test' in split:
            self.vallist.target = self.target['valset']
            self.vallist.save()
            self.write('val')
        if 'deploy' in split:
            self.write('deploy')

    def writeSolver(self):
        train_net = self.target['train']
        val_net = self.target['val']
        with open(self.target['solver'], 'w') as f:
            f.write((
                "train_net: '" + train_net + "\n"
                "test_net: '" + val_net + "\n"
                "test_iter: " + len(self.trainlist) + "\n"
                "test_interval: " + 99999999999 + "\n"
                "display: " + len(self.trainlist) + "\n"
                "average_loss: " + len(self.trainlist) + "\n"
                "lr_policy: 'fixed'\n"
                "base_lr: " + str(self.baselr) + "\n"
                "momentum: 0.99\n"
                "iter_size: 1"
                "max_iter: 100000\n"
                "weight_decay: 0.0005\n"
                "snapshot: " + len(self.trainlist) + "\n"
                "snapshot_prefix: "
                "'" + self.target['snapshots'] + "train" + "'\n"
                "test_initialization: false\n"
            ))

    def train(self):
        self.prepare('train')
        self.writeSolver()
        self.createSolver(self.target['solver'], self.weights, self.gpu)
        interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
        ba.caffeine.surgery.interp(self.solver.net, interp_layers)
        for _ in range(self.epochs):
            solver.step(len(self.samples))
        self.solver.snapshot()

    def test(self):
        self.prepare('test')
        pass

    def forwardList(self, list_=None, mean=None, heatmaplayer='upscore8_'):
        if list_ is None:
            self.list = self.vallist
            list_  = self.vallist.source
        else:
            self.addListFile(list_)
        if mean is None:
            if self.samples.mean != []:
                print('Using mean from samples')
            else:
                print('No mean.....')
                return
        self.prepare('deploy')
        self.createNet(self.target['deploy'], self.weights, self.gpu)
        ba.utils.touch(self.target['segmentations'])
        ba.utils.touch(self.target['heatmaps'])
        print('Forwarding all in {}'.format(list_))
        for idx in tqdm(self.list.list):
            bn = os.path.splitext(idx)[0]
            bn_seg = self.target['segmentations'] + bn
            bn_hm = self.target['heatmaps'] + bn
            self.forward(self.loadimg(idx))
            score = self.net.blobs['score'].data[0][1,...]
            heatmap = self.net.blobs[heatmaplayer].data[0][1,...]
            imsave(bn_seg + '.png', score)
            imsave(bn_hm + '.png', heatmap)
            ba.utils.apply_overlay(self.tmpim, score, bn_seg + '_overlay.png')
            ba.utils.apply_overlay(self.tmpim, score, bn_hm + '_overlay.png')
