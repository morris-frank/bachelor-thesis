import argparse
import ba
import ba.caffeine
import ba.netrunner
import ba.utils
import copy
from enum import Enum
from glob import glob
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time

MODELDIR = 'data/models/'


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
        self.cnn = None
        self.parse_arguments(argv)
        if self.args.conf is not None:
            self.load_conf(self.args.conf)

    def run(self):
        if self.args.repeat > 1:
            for idx in range(self.args.repeat):
                start_time = time.time()
                self._run_single()
                end_time = time.time()
                progres_str = ba.utils.format_meter(
                    idx, self.args.repeat, end_time - start_time)
                self.notify('{}: {}'.format(self.conf['tag'], progres_str))
        else:
            self._run_single()

    def _train_test_double(self):
        fc_conf = self.args.conf
        fcn_conf = self.args.conf[:-5] + '_FCN.yaml'
        self.load_conf(fc_conf)
        self.prepare()
        self.train()
        self.load_conf(fcn_conf)
        self.prepare()
        self.conv_test()
        self.clear()

    def _run_single(self):
        if self.args.train and self.args.test and self.args.tofcn:
            return self._train_test_double()

        if self.args.prepare:
            self.prepare()

        if self.args.train:
            self.train()

        if self.args.test and not self.args.tofcn:
            self.test()

        if self.args.test and self.args.tofcn:
            self.conv_test()

        if not self.args.test and self.args.tofcn:
            self.convert_to_FCN()
        self.clear()

    def clear(self):
        if self.cnn is not None:
            self.cnn.clear()

    def parse_arguments(self, argv):
        '''Parse the arguments for an experiment script'''
        parser = argparse.ArgumentParser(description='Runs a experiment')
        parser.add_argument('--default', action='store_true')
        parser.add_argument('--prepare', action='store_true')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--tofcn', action='store_true')
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--repeat', type=int, nargs=1, default=1,
                            metavar='count',
                            help='How often to repeat this experiment.')
        parser.add_argument('--bs', type=int, nargs=1, default=0,
                            metavar='count',
                            help='The batch size for training or testing.')
        parser.add_argument('--threads', type=int, nargs=1, default=1,
                            metavar='count',
                            help='The count threads for execution.')
        parser.add_argument('--gpu', type=int, nargs='+', default=0,
                            help='The GPU to use.', metavar='id')
        parser.add_argument('conf', type=str, nargs='?',
                            help='The YAML conf file')
        self.args = parser.parse_args(args=argv)
        for item in ['conf', 'bs', 'threads', 'repeat']:
            if isinstance(self.args.__dict__[item], list):
                self.args.__dict__[item] = self.args.__dict__[item][0]
        self.threaded = self.args.threads > 1
        self.quiet = self.args.repeat > 1

    def load_conf(self, config_file):
        '''Open a YAML Configuration file and make a Bunch from it'''
        defaults = ba.utils.load('data/experiments/defaults.yaml')
        self.conf = ba.utils.load(config_file)
        if 'tag' not in self.conf:
            self.conf['tag'] = os.path.basename(
                os.path.splitext(config_file)[0])
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
                self.conf[step] = '{}{}/{}.txt'.format(
                    MODELDIR, self.conf['tag'], step)
            else:
                print('''Stepfile for {} ({}) is not pointing to a
                      file or True.'''.format(step, self.conf[step]))
                sys.exit(1)

        if 'train_sizes' in self.conf:
            if not isinstance(self.conf['train_sizes'], list):
                print('train_sizes shall be a list of integers.')
                sys.exit()

        if self.args.bs > 0:
            self.conf['batch_size'] = self.args.bs

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
                          solver_weights=self.conf.get('weights', ''),
                          net_generator=self.conf['net'],
                          images=self.conf['images'],
                          labels=self.conf['labels'],
                          mean=self.conf['mean'],
                          quiet=self.quiet
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
        self.cnn.gpu = self.args.gpu

    def prepare(self):
        self._call_method(self._prepare)

    def test(self):
        '''Tests the given experiment, Normally depends on user input.
        If --default flag is set will test EVERY snapshot previously saved.
        '''
        self._call_method(self._test)

    def conv_test(self):
        '''Converts the network online into an FCN and tests the given
        experiment, Normally depends on user input. If --default flag is set
        will test EVERY snapshot previously saved.
        '''
        self._call_method(self._conv_test)

    def train(self):
        '''Trains the given experiment'''
        self._call_method(self._train)

    def convert_to_FCN(self):
        '''Converts the source weights to an FCN and saves them.'''
        self._call_method(self._convert_to_FCN)

    def _prepare(self):
        print('Preparing all phases for {}.'.format(self.conf['tag']))
        old_weights = self.conf.get('weights', '')
        self.conf['weights'] = ''
        self.prepare_network()
        self.cnn.prepare()
        self.conf['weights'] = old_weights

    def _conv_test(self):
        import caffe
        caffe.set_mode_gpu()
        caffe.set_device(self.args.gpu[0])

        def inline_convert_to_fcn():
            modeldef = '{}{}/deploy.prototxt'.format(
                MODELDIR, self.conf['tag'])
            orig_modeldef = modeldef.replace('FCN_', '')
            fc_net = caffe.Net(orig_modeldef, self.cnn.net_weights, caffe.TEST)
            self.cnn.net = caffe.Net(modeldef, caffe.TEST)

            old_params = ['fc']
            new_params = ['fc_conv']

            # Transplant all the unchanged weights
            for param_name in fc_net.params:
                if any([param_name == param for param in old_params]):
                    continue
                for mat_idx in range(len(fc_net.params[param_name])):
                    copied = fc_net.params[param_name][mat_idx].data
                    self.cnn.net.params[param_name][mat_idx].data[...] = copied

            self.cnn.net = ba.caffeine.surgery.convert_to_FCN(
                self.cnn.net, fc_net, new_params, old_params)

            return False

        self._meta_test(callback=inline_convert_to_fcn)

    def _train(self):
        self.prepare_network()
        return self.cnn.train()

    def _convert_to_FCN(self, new_tag=None):
        import caffe
        caffe.set_mode_cpu()
        caffe.set_device(self.args.gpu[0])
        old_tag = self.conf['tag']
        if new_tag is None:
            mgroups = re.match('(.*_)([0-9]+samples)', old_tag).groups()
            new_tag = '{}FCN_{}'.format(mgroups[0], mgroups[1])
        old_model_dir = '{}{}/'.format(MODELDIR, old_tag)
        new_model_dir = '{}{}/'.format(MODELDIR, new_tag)
        for f in ['train.txt', 'test.txt']:
            if os.path.isfile(old_model_dir + f):
                shutil.copy(src=old_model_dir + f, dst=new_model_dir)

        old_snap_dir = self.conf['snapshot_dir'].format(old_tag)
        new_snap_dir = ba.utils.touch(
            self.conf['snapshot_dir'].format(new_tag))
        weights = glob('{}*caffemodel'.format(old_snap_dir))
        if len(weights) < 1:
            print('No weights found for {}'.format(self.conf['tag']))
            return False
        for w in weights:
            bn = os.path.basename(w)
            new_weights = new_snap_dir + 'classifier_' + bn
            question = 'You want to convert {}?'.format(w)
            if not ba.utils.query_boolean(question, default='yes',
                                          defaulting=self.args.default):
                continue
            print('CONVERTING ' + bn)
            old_net = caffe.Net(
                old_model_dir + 'deploy.prototxt', w, caffe.TEST)
            new_net = caffe.Net(
                new_model_dir + 'deploy.prototxt', w, caffe.TEST)
            old_params = ['fc']
            new_params = ['fc_conv']
            converted_net = ba.caffeine.surgery.convert_to_FCN(
                new_net, old_net, new_params, old_params, new_weights)
            converted_net.save(new_weights)

    def _meta_test(self, callback=(lambda: True)):
        snapdir = self.conf['snapshot_dir'].format(self.conf['tag'])
        if self.args.tofcn:
            snapdir = snapdir.replace('FCN_', '')
        weights = glob('{}*caffemodel'.format(snapdir))
        if len(weights) < 1:
            print('No weights found for {}'.format(self.conf['tag']))
            self.prepare_network()
            self.cnn.write('deploy')
            return False
        for w in weights:
            bn = os.path.basename(w)
            if '500' not in bn:
                continue
            question = 'You want to test {}?'.format(bn)
            if not ba.utils.query_boolean(question, default='yes',
                                          defaulting=self.args.default):
                continue
            print('TESTING {} for {}'.format(bn, self.conf['tag']))
            self.prepare_network()
            self.cnn.net_weights = w
            if self.conf['test_images'] != '':
                self.cnn.images = self.conf['test_images']
            reset_net = callback()
            if 'slicefile' in self.conf:
                self.cnn.test(self.conf['slicefile'], reset_net=reset_net)
            else:
                self.cnn.test(reset_net=reset_net)
            self.cnn.clear()

    def _call_method(self, fptr):
        assert(self.args.conf is not None)
        if 'train_sizes' in self.conf:
            self._call_multi_scaled(fptr, self.conf['train_sizes'])
        else:
            fptr()

    def _call_multi_scaled(self, fptr, set_sizes):
        def _exec_threaded(self, fname, sema):
            with sema:
                self.__getattribute__(fname)()

        from ba import SetList
        old_train_conf = self.conf['train']
        hyper_set = SetList(self.conf['train'])
        self.conf['train'] = copy.deepcopy(hyper_set)
        bname = self.conf['tag']
        fname = fptr.__name__
        if self.threaded:
            threads = []
            sema = mp.BoundedSemaphore(value=self.args.threads)
        try:
            for set_size in set_sizes:
                self.conf['tag'] = '{}_{}samples'.format(bname, set_size)
                question = 'You want to run {} for {}?'.format(
                    fptr.__name__, set_size)
                if not ba.utils.query_boolean(question, default='yes',
                                              defaulting=self.args.default):
                    continue
                if set_size == 0 or set_size > len(hyper_set.list):
                    self.conf['train'].list = hyper_set.list
                else:
                    self.conf['train'].list = random.sample(hyper_set.list,
                                                            set_size)
                if self.threaded:
                    orig_net = self.conf['net']
                    self.conf['net'] = ''
                    worker = copy.deepcopy(self)
                    worker.conf['net'] = orig_net
                    self.conf['net'] = orig_net
                    threads.append(mp.Process(target=worker._exec_threaded,
                                              args=(fname, sema,)))
                    threads[-1].start()
                else:
                    return_code = fptr()
                    if return_code is not None and return_code < 0:
                        break
        except KeyboardInterrupt:
            print('\n\n_call_multi_scaled for {} was interrupted'.format(
                fptr.__name__))
            if self.threaded:
                for thread in threads:
                    thread.terminate()
                    thread.join()
        self.conf['tag'] = bname
        self.conf['train'] = old_train_conf
