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
    parser.add_argument('--default', action='store_true')
    parser.add_argument('--gpu', type=int, nargs='?', help='The GPU to use.', default=0)
    sysargs = parser.parse_args(args=argv)
    if isinstance(sysargs.conf, list):
        sysargs.conf = sysargs.conf[0]
    return sysargs


def loadConf(path):
    '''Open a YAML Configuration file and make a Bunch from it

    Args:
        path (str): The path to the conf file

    Returns:
        The parsed configuration array
    '''
    with open(path, 'r') as f:
        try:
            conf = utils.Bunch(yaml.load(f))
        except yaml.YAMLError as e:
            print(e)
            sendWarning('Configuration couldn\'t be loaded')
            sys.exit()
    if not conf.net.startswith('ba.'):
        print('Given Network is not from correct namespace you fucker')
        sys.exit()
    else:
        conf.net = eval(conf.net)
    return conf


def prepareFCN(sysargs, conf):
    # TODO(doc): Add docstring
    fcn = FCNPartRunner(conf.tag, conf.train, conf.val)
    fcn.weights = conf.weights
    fcn.net_generator = conf.net
    fcn.baselr = conf.baselr
    fcn.epochs = conf.epochs
    fcn.gpu = sysargs.gpu
    fcn.imgdir = conf.images
    fcn.imgext = conf.image_extension
    fcn.labeldir = conf.labels
    fcn.targets()
    return fcn


def runTests(sysargs, conf):
    # TODO(doc): Add docstring
    snapdir = 'data/models/{}/snapshots/'.format(conf.tag)
    weights = glob('{}*caffemodel'.format(snapdir))
    if len(weights) < 1:
        print('No weights found for {}'.formart(conf.tag))
        return False
    conf.train = ''
    for w in weights:
        bn = os.path.basename(w)
        if not utils.query_boolean('You want to test for {}?'.format(bn),
                                   default='yes',
                                   defaulting=sysargs.default):
            continue
        conf.weights = w
        print('TESTING ' + bn)
        fcn = prepareFCN(sysargs, conf)
        fcn.forwardList()


def runExperiment(sysargs, conf):
    '''Run an experiment

    Args:
        sysargs (list): The cli runtime options from parseArgs()
        conf (list): The configuration array from loadConf()
    '''
    fcn = prepareFCN(sysargs, conf)
    lastiter = fcn.epochs * len(fcn.trainlist)

    if utils.query_boolean('Do you want to train?',
                           default='yes',
                           defaulting=sysargs.default):
        fcn.train()
    else:
        fcn.prepare('train')
        fcn.writeSolver()
        fcn.createSolver(fcn.target['solver'], fcn.weights, fcn.gpu)
        interp_layers = [k for k in fcn.solver.net.params.keys() if 'up' in k]
        caffeine.surgery.interp(fcn.solver.net, interp_layers)
        if utils.query_boolean('Do you want to save the new net?'):
            fcn.solver.snapshot()

    if utils.query_boolean('Do you want to test?',
                           default='yes',
                           defaulting=sysargs.default):
        fcn.weights = 'data/models/{}/snapshots/train_iter_{}.caffemodel'.format(conf.tag, lastiter)
        # Merge Train and Val Set (unique items...)
        testlist = SetList()
        testlist.list = list(set(fcn.vallist.list + fcn.trainlist.list))
        imgdir = fcn.imgdir
        imgext = fcn.imgext
        if conf.test_image_extension != '':
            fcn.imgext = conf.test_image_extension
        if conf.test_images != '':
            fcn.imgdir = conf.test_images
        fcn.forwardList(list_=testlist)
        fcn.imgdir = imgdir
        fcn.imgext = imgext
