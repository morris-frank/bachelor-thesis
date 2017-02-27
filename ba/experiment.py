import argparse
import ba
from ba import utils
from enum import Enum
from glob import glob
import os
import sys
import yaml


class RunMode(Enum):
    '''Contains the different possible modes for each step of an experiment.'''
    SINGLE = 0
    LMDB = 1
    LIST = 2


class Experiment(object):
    '''A class to contain everything needed for an experiment'''

    def __init__(self, argv):
        '''Initialize the new experiment from the cli args

        Args:
            argv (str): The options string
        '''
        self.argv = argv
        self.parseArgs()

    def parseArgs(self):
        '''Parse the arguments for an experiment script'''
        parser = argparse.ArgumentParser(description='Runs a experiment')
        parser.add_argument('conf', type=str, nargs=1,
                            help='The YAML conf file')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--data', action='store_true')
        parser.add_argument('--default', action='store_true')
        parser.add_argument('--gpu', type=int, nargs='+', default=0,
                            help='The GPU to use.')
        self.sysargs = parser.parse_args(args=self.argv)
        if isinstance(self.sysargs.conf, list):
            self.sysargs.conf = self.sysargs.conf[0]

    def _loadConfYaml(self, path):
        '''Loads a yaml file.

        Args:
            path (str): The filepath of the file

        Returns:
            The dictionary from the contents
        '''
        with open(path, 'r') as f:
            try:
                contents = yaml.load(f)
            except yaml.YAMLError as e:
                print(e)
                sendWarning('Configuration {} not loadable'.format(path))
                sys.exit()
        return contents

    def loadConf(self):
        '''Open a YAML Configuration file and make a Bunch from it'''
        defaults = self._loadConfYaml('data/experiments/defaults.yaml')
        self.conf = self._loadConfYaml(self.sysargs.conf)
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
                    setattr(self, step + 'mode', Mode.LMDB)
                elif self.conf[step].endswith('txt'):
                    setattr(self, step + 'mode', Mode.LIST)
                else:
                    setattr(self, step + 'mode', Mode.SINGLE)
            else:
                print('{} is not pointing to a file.'.format(self.conf[step]))
                sys.exit(1)

    def prepareCNN(self):
        '''Prepares a NetRunner from the given configuration.'''
        if self.conf['sliding_window']:
            runner = ba.SlidingFCNPartRunner
        else:
            runner = ba.FCNPartRunner
        self.cnn = runner(self.conf['tag'],
                          trainset=self.conf['train'],
                          valset=self.conf['val'],
                          solver_weights=self.conf['weights'],
                          net_generator=self.conf['net'],
                          images=self.conf['images'],
                          labels=self.conf['labels'],
                          mean=self.conf['mean']
                          )

        spattrs = ['lr_policy', 'stepsize', 'weight_decay', 'base_lr']
        for spattr in spattrs:
            if spattr in self.conf:
                self.cnn._spattr[spattr] = self.conf[spattr]

        switches = ['tofcn', 'learn_fc']
        for switch in switches:
            if switch in self.conf:
                self.cnn.generator_switches[switch] = self.conf.get(switch,
                                                                    False)

        self.cnn.gpu = self.sysargs.gpu

    def runTests(self):
        '''Tests the given experiment, Normally depends on user input. If --default
        flag is set will test EVERY snapshot previously saved.
        '''
        snapdir = 'data/models/{}/snapshots/'.format(self.conf['tag'])
        weights = glob('{}*caffemodel'.format(snapdir))
        if len(weights) < 1:
            print('No weights found for {}'.format(self.conf['tag']))
            self.prepareCNN()
            self.cnn.write('deploy')
            return False
        for w in weights:
            bn = os.path.basename(w)
            if not utils.query_boolean('You want to test for {}?'.format(bn),
                                       default='yes',
                                       defaulting=self.sysargs.default):
                continue
            print('TESTING ' + bn)
            self.prepareCNN()
            self.cnn.net_weights = w
            if self.conf['test_images'] != '':
                self.cnn.images = self.conf['test_images']
            self.cnn.forwardVal()
            self.cnn.clear()

    def runTrain(self):
        '''Trains the given experiment'''
        self.prepareCNN()
        lastiter = self.cnn.epochs * len(self.cnn.trainset)
        self.cnn.train()

    def genData(self, name, source):
        '''Generates the training data for that experiment'''
        ppset = ba.PascalPartSet(name, source, ['lwing', 'rwing'], 'aeroplane')
        ppset.saveSegmentations(augment=2)
        if self.conf['sliding_window']:
            ppset.saveBoundingBoxes('data/datasets/voc2010/JPEGImages/', negatives=2, augment=2)
