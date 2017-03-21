import argparse
import ba
import ba.utils
from enum import Enum
from glob import glob
import os
import random
import sys


class RunMode(Enum):
    '''Contains the different possible modes for each step of an experiment.'''
    SINGLE = 0
    LMDB = 1
    LIST = 2


class Experiment(ba.utils.NotifierClass):
    '''A class to contain everything needed for an experiment'''

    def __init__(self, argv, **kwargs):
        '''Initialize the new experiment from the cli args

        Args:
            argv (str): The options string
        '''
        super().__init__(**kwargs)
        self.argv = argv
        self.parse_arguments()

    def parse_arguments(self):
        '''Parse the arguments for an experiment script'''
        parser = argparse.ArgumentParser(description='Runs a experiment')
        parser.add_argument('conf', type=str, nargs=1,
                            help='The YAML conf file')
        parser.add_argument('--data', action='store_true')
        parser.add_argument('--default', action='store_true')
        parser.add_argument('--prepare', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--tofcn', action='store_true')
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--gpu', type=int, nargs='+', default=0,
                            help='The GPU to use.')
        self.sysargs = parser.parse_args(args=self.argv)
        if isinstance(self.sysargs.conf, list):
            self.sysargs.conf = self.sysargs.conf[0]

    def load_slices(self):
        with open(self.conf['train']) as f:
            imlist = [l[:-1] for l in f.readlines() if l.strip()]
        slicelist = ba.utils.load_YAML(self.conf['slicefile'])
        return {im: slicelist[im] for im in imlist}

    def load_conf(self):
        '''Open a YAML Configuration file and make a Bunch from it'''
        defaults = ba.utils.load_YAML('data/experiments/defaults.yaml')
        self.conf = ba.utils.load_YAML(self.sysargs.conf)
        if 'tag' not in self.conf:
            self.conf['tag'] = os.path.basename(
                os.path.splitext(self.sysargs.conf)[0])
        for key, value in defaults.items():
            if key not in self.conf:
                self.conf[key] = value
        if not self.conf['net'].startswith('ba.'):
            print('Given Network is not from correct namespace you fucker')
            sys.exit()
        else:
            self.conf['net'] = eval(self.conf['net'])

        if 'test' not in self.conf:
            self.conf['test'] = self.conf['val']

        if 'slicefile' in self.conf:
            self.conf['slices'] = self.load_slices()

        if 'test_sizes' in self.conf:
            if not isinstance(self.conf['test_sizes'], list):
                print('test_sizes shall be a list of integers.')
                sys.exit()

        for step in ['train', 'test', 'val']:
            if os.path.isfile(self.conf[step]):
                if 'lmdb' in self.conf[step]:
                    setattr(self, step + 'mode', RunMode.LMDB)
                elif self.conf[step].endswith('txt'):
                    setattr(self, step + 'mode', RunMode.LIST)
                else:
                    setattr(self, step + 'mode', RunMode.SINGLE)
            else:
                print('{} is not pointing to a file.'.format(self.conf[step]))
                sys.exit(1)

    def prepare_network(self):
        '''Prepares a NetRunner from the given configuration.'''
        if self.conf['sliding_window']:
            runner = ba.netrunner.SlidingFCNPartRunner
        else:
            runner = ba.netrunner.FCNPartRunner
        self.cnn = runner(self.conf['tag'],
                          trainset=self.conf['train'],
                          valset=self.conf['val'],
                          testset=self.conf['test'],
                          solver_weights=self.conf['weights'],
                          net_generator=self.conf['net'],
                          images=self.conf['images'],
                          labels=self.conf['labels'],
                          mean=self.conf['mean']
                          )

        # Extra atrributes for the solver
        attrs = ['lr_policy', 'stepsize', 'weight_decay', 'base_lr',
                 'test_iter', 'test_interval', 'max_iter', 'snapshot']
        for attr in attrs:
            if attr in self.conf:
                self.cnn._solver_attr[attr] = self.conf[attr]

        # Extra attributes for the network generator
        attrs = ['slices']
        for attr in attrs:
            if attr in self.conf:
                self.cnn.generator_attr[attr] = self.conf[attr]

        # Boolean switsches for the network generator (default is false)
        switches = ['tofcn', 'learn_fc']
        for switch in switches:
            if switch in self.conf:
                self.cnn.generator_switches[switch] = self.conf.get(switch,
                                                                    False)

        self.cnn.gpu = self.sysargs.gpu

    def generate_data(self, name, source):
        '''Generates the training data for that experiment'''
        ppset = ba.PascalPartSet(name, source, ['lwing', 'rwing'], 'aeroplane')
        ppset.segmentations(augment=2)
        if self.conf['sliding_window']:
            ppset.bounding_boxes('data/datasets/voc2010/JPEGImages/',
                                 negatives=2, augment=2)

    def prepare(self):
        self._multi_scale_exec(self._prepare)

    def test(self):
        if 'set_sizes' in self.conf:
            self._multi_scale_exec(self._test, self.conf['set_sizes'])
        else:
            self._test()

    def train(self):
        if 'set_sizes' in self.conf:
            self._multi_scale_exec(self._train, self.conf['set_sizes'])
        else:
            self._train()

    def convert_to_FCN(self, newTag=None):
        if 'set_sizes' in self.conf:
            self._multi_scale_convert_to_FCN(
                self._train, self.conf['set_sizes'])
        else:
            self._convert_to_FCN()

    def _prepare(self):
        print('Preparing all phases for {}.'.format(self.conf['tag']))
        old_weights = self.conf['weights']
        self.conf['weights'] = ''
        self.prepare_network()
        self.cnn.prepare()
        self.conf['weights'] = old_weights

    def _test(self):
        '''Tests the given experiment, Normally depends on user input. If --default
        flag is set will test EVERY snapshot previously saved.
        '''
        snapdir = 'data/models/{}/snapshots/'.format(self.conf['tag'])
        weights = glob('{}*caffemodel'.format(snapdir))
        if len(weights) < 1:
            print('No weights found for {}'.format(self.conf['tag']))
            self.prepare_network()
            self.cnn.write('deploy')
            return False
        for w in weights:
            bn = os.path.basename(w)
            if not ba.utils.query_boolean('You want to test for {}?'.format(bn),
                                          default='yes', defaulting=self.sysargs.default):
                continue
            print('TESTING ' + bn)
            self.prepare_network()
            self.cnn.net_weights = w
            if self.conf['test_images'] != '':
                self.cnn.images = self.conf['test_images']
            if 'slicefile' in self.conf:
                self.cnn.test(self.conf['slicefile'])
            else:
                self.cnn.test()
            self.cnn.clear()

    def _train(self):
        '''Trains the given experiment'''
        self.prepare_network()
        self.cnn.train()

    def _convert_to_FCN(self):
        '''Converts the source weights to an FCN'''
        import caffe
        caffe.set_mode_cpu()
        oldTag = self.conf['tag']
        if newTag is None:
            newTag = oldTag + '_FCN'
        oldSnaps = 'data/models/{}/snapshots/'.format(oldTag)
        newSnaps = ba.utils.touch('data/models/{}/snapshots/'.format(newTag))
        oldDeploy = 'data/models/{}/deploy.prototxt'.format(oldTag)
        newDeploy = 'data/models/{}/deploy.prototxt'.format(newTag)
        weights = glob('{}*caffemodel'.format(oldSnaps))
        if len(weights) < 1:
            print('No weights found for {}'.format(self.conf['tag']))
            return False
        for w in weights:
            bn = os.path.basename(w)
            new_weights = newSnaps + 'classifier_' + bn
            if not ba.utils.query_boolean('You want to convert {}?'.format(w),
                                       default='yes',
                                       defaulting=self.sysargs.default):
                continue
            print('CONVERTING ' + bn)
            old_net = caffe.Net(oldDeploy, w, caffe.TEST)
            new_net = caffe.Net(newDeploy, w, caffe.TEST)
            old_params = ['fc']
            new_params = ['fc_conv']
            ba.caffeine.surgery.convertToFCN(
                new_net, old_net, new_params, old_params, new_weights)

    def _multi_scale_convert_to_FCN(self, set_sizes):
        from ba import SetList
        hyperTrainSet = SetList(self.conf['train'])
        bname = self.conf['tag']
        for set_size in set_sizes:
            if set_size == 0:
                self.cnn.trainset.list = hyperTrainSet.list
            else:
                self.cnn.trainset.list = random.sample(hyperTrainSet.list, set_size)
                # self.cnn.testset.set = self.cnn.testset.set - self.cnn.trainset.set
            self.conf['tag'] = '{}_{}samples'.format(bname, set_size)
            newTag = '{}_FCN_{}samples'.format(bname, set_size)
            self.convert_to_FCN(newTag=newTag)
        self.conf['tag'] = bname
        self.notify('Finished cascade FCN conversion for {}'.format(bname))

    def _multi_scale_exec(self, fptr, set_sizes):
        from ba import SetList
        hyperTrainSet = SetList(self.conf['train'])
        bname = self.conf['tag']
        for set_size in set_sizes:
            if set_size == 0:
                self.cnn.trainset.list = hyperTrainSet.list
            else:
                self.cnn.trainset.list = random.sample(hyperTrainSet.list, set_size)
                # self.cnn.testset.set = self.cnn.testset.set - self.cnn.trainset.set
            self.conf['tag'] = '{}_{}samples'.format(bname, set_size)
            fptr()
        self.conf['tag'] = bname
