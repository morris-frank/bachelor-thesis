import argparse
import ba
from ba import caffeine
from ba.netrunner import FCNPartRunner
from ba.set import SetList
from ba import utils
from glob import glob
import os
import sys
import yaml

class Experiment(object):
    '''A class to contain everything needed for an experiment'''
    pass

def parseArgs(argv=sys.argv):
    '''Parse the arguments for an experiment script

    Args:
        argv (list, optional): The given cli arguments, defaults to sys.argv

    Returns:
        The parsed arguments
    '''
    parser = argparse.ArgumentParser(description='Runs a experiment')
    parser.add_argument('conf', type=str, nargs=1, help='The YAML conf file')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--gpu', type=int, nargs='?', help='The GPU to use.', default=0)
    sysargs = parser.parse_args(args=argv)
    if isinstance(sysargs.conf, list):
        sysargs.conf = sysargs.conf[0]
    return sysargs

def loadConfYaml(path):
    with open(path, 'r') as f:
        try:
            contents = yaml.load(f)
        except yaml.YAMLError as e:
            print(e)
            sendWarning('Configuration {} couldn\'t be loaded'.format(path))
            sys.exit()
    return contents

def loadConf(path):
    '''Open a YAML Configuration file and make a Bunch from it

    Args:
        path (str): The path to the conf file

    Returns:
        The parsed configuration array
    '''
    defaults = loadConfYaml('data/experiments/defaults.yaml')
    conf  = loadConfYaml(path)
    conf = utils.Bunch({**defaults, **conf})
    if not conf.net.startswith('ba.'):
        print('Given Network is not from correct namespace you fucker')
        sys.exit()
    else:
        conf.net = eval(conf.net)
    return conf


def prepareFCN(sysargs, conf):
    '''Prepares a FCNPartRunner from the given configuration.

    Args:
        sysargs (list): The cli runtime options from parseArgs()
        conf (list): The configuration array from loadConf()

    Returns:
        the FCNPartRunner
    '''
    fcn = FCNPartRunner(conf.tag,
                        train=conf.train,
                        val=conf.val,
                        solver_weights=conf.weights,
                        net_generator=conf.net,
                        baselr=conf.baselr,
                        epochs=conf.epochs,
                        images=conf.images,
                        labels=conf.labels
                        )
    fcn.gpu=sysargs.gpu
    fcn.generator_switches['learn_fc'] = conf.learn_fc
    return fcn


def runTests(sysargs, conf):
    '''Tests the given experiment, Normally depends on user input. If --default
    flag is set will test EVERY snapshot previously saved.

    Args:
        sysargs (list): The cli runtime options from parseArgs()
        conf (list): The configuration array from loadConf()
    '''
    snapdir = 'data/models/{}/snapshots/'.format(conf.tag)
    weights = glob('{}*caffemodel'.format(snapdir))
    if len(weights) < 1:
        print('No weights found for {}'.formart(conf.tag))
        return False
    for w in weights:
        bn = os.path.basename(w)
        if not utils.query_boolean('You want to test for {}?'.format(bn),
                                   default='yes',
                                   defaulting=sysargs.default):
            continue
        print('TESTING ' + bn)
        fcn = prepareFCN(sysargs, conf)
        fcn.net_weights = w
        if conf.test_images != '':
            fcn.images = conf.test_images
        fcn.forwardVal()
        fcn.clear()


def runTrain(sysargs, conf):
    '''Trains the given experiment

    Args:
        sysargs (list): The cli runtime options from parseArgs()
        conf (list): The configuration array from loadConf()
    '''
    fcn = prepareFCN(sysargs, conf)
    lastiter = fcn.epochs * len(fcn.trainlist)
    fcn.train()
