import argparse
import ba
import ba.utils
import copy
from enum import Enum
from glob import glob
import multiprocessing as mp
import os
import random
import re
import shutil
import signal
import sys
import time


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
        self.cnn = None
        self.parse_arguments()
        if self.sysargs.conf is not None:
            self.load_conf()

    def exit(self):
        if self.cnn is not None:
            self.cnn.clear()

    def init_worker(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def parse_arguments(self):
        '''Parse the arguments for an experiment script'''
        parser = argparse.ArgumentParser(description='Runs a experiment')
        parser.add_argument('--data_classes', type=str, nargs='+', metavar='class')
        parser.add_argument('--data_parts', type=str, nargs='+', metavar='part')
        parser.add_argument('--default', action='store_true')
        parser.add_argument('--prepare', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--tofcn', action='store_true')
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--bs', type=int, nargs=1, default=1,
                            metavar='count',
                            help='The batch size for training or testing.')
        parser.add_argument('--threads', type=int, nargs=1, default=1,
                            metavar='count',
                            help='The count threads for execution.')
        parser.add_argument('--gpu', type=int, nargs='+', default=0,
                            help='The GPU to use.', metavar='id')
        parser.add_argument('conf', type=str, nargs='?',
                            help='The YAML conf file')
        self.sysargs = parser.parse_args(args=self.argv)
        for item in ['conf', 'bs', 'threads']:
            if isinstance(self.sysargs.__dict__[item], list):
                self.sysargs.__dict__[item] = self.sysargs.__dict__[item][0]
        self.threaded = self.sysargs.threads > 1

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

        for step in ['train', 'test', 'val']:
            if os.path.isfile(self.conf[step]):
                if 'lmdb' in self.conf[step]:
                    setattr(self, step + 'mode', RunMode.LMDB)
                elif self.conf[step].endswith('txt'):
                    setattr(self, step + 'mode', RunMode.LIST)
                else:
                    setattr(self, step + 'mode', RunMode.SINGLE)
            elif isinstance(self.conf[step], bool) and self.conf[step] is True:
                self.conf[step] = 'data/models/{}/{}.txt'.format(
                    self.conf['tag'], step)
            else:
                print('''Stepfile for {} ({}) is not pointing to a
                      file or True.'''.format(step, self.conf[step]))
                sys.exit(1)

        if 'train_sizes' in self.conf:
            if not isinstance(self.conf['train_sizes'], list):
                print('train_sizes shall be a list of integers.')
                sys.exit()

        if self.sysargs.bs > 0:
            self.conf['batch_size'] = self.sysargs.bs

    def prepare_network(self):
        '''Prepares a NetRunner from the given configuration.'''
        assert(self.sysargs.conf is not None)
        if self.conf['sliding_window']:
            runner = ba.netrunner.SlidingFCNPartRunner
        else:
            runner = ba.netrunner.FCNPartRunner
        self.cnn = runner(self.conf['tag'],
                          trainset=self.conf['train'],
                          valset=self.conf['val'],
                          testset=self.conf['test'],
                          solver_weights=self.conf.get('weights', ''),
                          net_generator=self.conf['net'],
                          images=self.conf['images'],
                          labels=self.conf['labels'],
                          mean=self.conf['mean']
                          )

        # Extra attributes for the cnn
        attrs = ['batch_size']
        for attr in attrs:
            if attr in self.conf:
                self.cnn.__dict__[attr] = self.conf[attr]

        # Extra atrributes for the solver
        attrs = ['lr_policy', 'stepsize', 'weight_decay', 'base_lr',
                 'test_iter', 'test_interval', 'max_iter', 'snapshot']
        for attr in attrs:
            if attr in self.conf:
                self.cnn._solver_attr[attr] = self.conf[attr]

        # Extra attributes for the network generator
        attrs = ['batch_size', 'patch_size', 'ppI',
                 'images', 'negatives', 'slicefile']
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

    def generate_data(self, name, source, classes=None, parts=None):
        '''Generates the training data for that experiment'''
        ppset = ba.PascalPartSet(name, source, classes=classes, parts=parts)
        ppset.segmentations()
        ppset.bounding_boxes('data/datasets/voc2010/JPEGImages/',
                             negatives=2, augment=2)

    def prepare(self):
        assert(self.sysargs.conf is not None)
        self._multi_or_single_scale_exec(self._prepare)

    def test(self):
        assert(self.sysargs.conf is not None)
        self._multi_or_single_scale_exec(self._test)

    def train(self):
        assert(self.sysargs.conf is not None)
        self._multi_or_single_scale_exec(self._train)

    def convert_to_FCN(self):
        assert(self.sysargs.conf is not None)
        self._multi_or_single_scale_exec(self._convert_to_FCN)

    def _prepare(self):
        print('Preparing all phases for {}.'.format(self.conf['tag']))
        old_weights = self.conf.get('weights', '')
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
            print('TESTING {} for {}'.format(bn, self.conf['tag']))
            self.prepare_network()
            self.cnn.net_weights = w
            if self.conf['test_images'] != '':
                self.cnn.images = self.conf['test_images']
            time.sleep(10)
            if 'slicefile' in self.conf:
                self.cnn.test(self.conf['slicefile'])
            else:
                self.cnn.test()
            self.cnn.clear()
            return True

    def _train(self):
        '''Trains the given experiment'''
        self.prepare_network()
        return self.cnn.train()

    def _convert_to_FCN(self, new_tag=None):
        '''Converts the source weights to an FCN'''
        import caffe
        caffe.set_mode_cpu()
        old_tag = self.conf['tag']
        if new_tag is None:
            mgroups = re.match('(.*_)([0-9]+samples)', old_tag).groups()
            new_tag =  '{}FCN_{}'.format(mgroups[0], mgroups[1])
        old_dir = 'data/models/{}/'.format(old_tag)
        new_dir = 'data/models/{}/'.format(new_tag)
        for f in ['train.txt', 'test.txt']:
            if os.path.isfile(old_dir + f):
                shutil.copy(src=old_dir + f, dst=new_dir)
        new_snap_dir = ba.utils.touch('{}snapshots/'.format(new_dir))
        weights = glob('{}*caffemodel'.format(old_dir + 'snapshots/'))
        if len(weights) < 1:
            print('No weights found for {}'.format(self.conf['tag']))
            return False
        for w in weights:
            bn = os.path.basename(w)
            new_weights = new_snap_dir + 'classifier_' + bn
            if not ba.utils.query_boolean('You want to convert {}?'.format(w),
                                       default='yes',
                                       defaulting=self.sysargs.default):
                continue
            print('CONVERTING ' + bn)
            old_net = caffe.Net(old_dir + 'deploy.prototxt', w, caffe.TEST)
            new_net = caffe.Net(new_dir + 'deploy.prototxt', w, caffe.TEST)
            old_params = ['fc']
            new_params = ['fc_conv']
            ba.caffeine.surgery.convertToFCN(
                new_net, old_net, new_params, old_params, new_weights)

    def _multi_or_single_scale_exec(self, fptr):
        if 'train_sizes' in self.conf:
            self._multi_scale_exec(fptr, self.conf['train_sizes'])
        else:
            fptr()

    def _thread_exec(self, fname, sema):
        with sema:
            self.__getattribute__(fname)()

    def _multi_scale_exec(self, fptr, set_sizes):
        from ba import SetList
        old_train_conf = self.conf['train']
        hyper_set = SetList(self.conf['train'])
        self.conf['train'] = copy.deepcopy(hyper_set)
        bname = self.conf['tag']
        fname = fptr.__name__
        if self.threaded:
            worker_pool = mp.Pool(self.sysargs.threads, self.init_worker)
            threads = []
            sema = mp.BoundedSemaphore(value=self.sysargs.threads)
        try:
            for set_size in set_sizes:
                self.conf['tag'] = '{}_{}samples'.format(bname, set_size)
                if not ba.utils.query_boolean('''You want to run {} for {}?'''.format(fptr.__name__, set_size), default='yes', defaulting=self.sysargs.default):
                    continue
                if set_size == 0 or set_size > len(hyper_set.list):
                    self.conf['train'].list = hyper_set.list
                else:
                    self.conf['train'].list = random.sample(hyper_set.list,
                                                            set_size)
                    # self.cnn.testset.set = self.cnn.testset.set - self.cnn.trainset.set
                if self.threaded:
                    orig_net = self.conf['net']
                    self.conf['net'] = ''
                    worker = copy.deepcopy(self)
                    worker.conf['net'] = orig_net
                    self.conf['net'] = orig_net
                    threads.append(mp.Process(target=worker._thread_exec,
                                              args=(fname, sema,)))
                    threads[-1].start()
                else:
                    return_code = fptr()
                    if return_code is not None and return_code < 0:
                        break
        except KeyboardInterrupt:
            print('\n\n_multi_scale_exec for {} was interrupted'.format(
                fptr.__name__))
            if self.threaded:
                for thread in threads:
                    thread.terminate()
                    thread.join()
        self.conf['tag'] = bname
        self.conf['train'] = old_train_conf
