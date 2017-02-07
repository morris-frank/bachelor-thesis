import argparse
import ba
from ba import caffeine
from ba import netrunner
from ba.set import SetList
from ba import utils
from glob import glob
import os
import sys
import yaml


class Experiment(object):
    '''A class to contain everything needed for an experiment'''
    pass

    __init__(argv):
        '''Initialize the new experiment from the cli args

        Args:
            argv (str): The options string
        '''
        self.argv = argv
        self.parseArgs()

    def parseArgs():
        '''Parse the arguments for an experiment script'''
        parser = argparse.ArgumentParser(description='Runs a experiment')
        parser.add_argument('conf', type=str, nargs=1,
                            help='The YAML conf file')
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--default', action='store_true')
        parser.add_argument('--gpu', type=int, nargs='?', default=0,
                            help='The GPU to use.')
        sysargs = parser.parse_args(args=self.argv)
        if isinstance(sysargs.conf, list):
            sysargs.conf = sysargs.conf[0]
        self.sysargs = sysargs

    def _loadConfYaml(path):
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

    def loadConf():
        '''Open a YAML Configuration file and make a Bunch from it'''
        defaults = self._loadConfYaml('data/experiments/defaults.yaml')
        conf = self._loadConfYaml(sysargs.conf.path)
        conf = utils.Bunch({**defaults, **conf})
        if not conf.net.startswith('ba.'):
            print('Given Network is not from correct namespace you fucker')
            sys.exit()
        else:
            conf.net = eval(conf.net)
        self.conf = conf

    def self._prepareFCN():
        '''Prepares a NetRunner from the given configuration.'''
        if self.conf.sliding_window:
            runner = netrunner.SlidingFCNPartRunner
        else:
            runner = netrunner.FCNPartRunner
        self.fcn = runner(self.conf.tag,
                          train=self.conf.train,
                          val=self.conf.val,
                          solver_weights=self.conf.weights,
                          net_generator=self.conf.net,
                          baselr=self.conf.baselr,
                          epochs=self.conf.epochs,
                          images=self.conf.images,
                          labels=self.conf.labels
                          )
        self.fcn.gpu = self.sysargs.gpu
        self.fcn.generator_switches['learn_fc'] = self.conf.learn_fc

    def runTests():
        '''Tests the given experiment, Normally depends on user input. If --default
        flag is set will test EVERY snapshot previously saved.
        '''
        snapdir = 'data/models/{}/snapshots/'.format(self.conf.tag)
        weights = glob('{}*caffemodel'.format(snapdir))
        if len(weights) < 1:
            print('No weights found for {}'.formart(self.conf.tag))
            return False
        for w in weights:
            bn = os.path.basename(w)
            if not utils.query_boolean('You want to test for {}?'.format(bn),
                                       default='yes',
                                       defaulting=self.sysargs.default):
                continue
            print('TESTING ' + bn)
            self._prepareFCN()
            self.fcn.net_weights = w
            if self.conf.test_images != '':
                self.fcn.images = self.conf.test_images
            self.fcn.forwardVal()
            self.fcn.clear()

    def runTrain():
        '''Trains the given experiment'''
        self._prepareFCN()
        lastiter = self.fcn.epochs * len(self.fcn.train)
        self.fcn.train()
