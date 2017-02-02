import argparse
import ba
from . import caffeine
from .netrunner import FCNPartRunner
from . import utils
import sys
import yaml

def parseArgs():
    parser = argparse.ArgumentParser(description='Runs a experiment')
    parser.add_argument('conf', nargs=1, help='The YAML conf file')
    parser.add_argument('--gpu', nargs='?', help='The GPU to use.', default=0)
    sysargs = parser.parse_args()
    if isinstance(sysargs.conf, list):
        sysargs.conf = sysargs.conf[0]
    return sysargs


def loadConf(path):
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
        conf.net = eval(conf.net[3:])
    return conf


def runExperiment(sysargs, conf):
    # sourcedir = 'data/datasets/pascalparts/Annotations_Part/'
    gpu = sysargs.gpu

    traintxt = 'data/tmp/pascpart_aeroplane_stern.txt'
    valtxt = 'data/tmp/pascpart_aeroplane_stern.txt'

    tag = 'airStern_2lFC'

    fcn = FCNPartRunner(conf.tag, conf.train, conf.val)
    fcn.weights = conf.weights
    fcn.net_generator = conf.net
    fcn.baselr = conf.baselr
    fcn.epochs = conf.epochs
    fcn.gpu = gpu
    fcn.imgdir = conf.images
    fcn.imgext = conf.image_extension
    fcn.labeldir = conf.labels
    lastiter = fcn.epochs * len(fcn.trainlist)

    if utils.query_boolean('Do you want to train?'):
        fcn.train()
    else:
        fcn.prepare('train')
        fcn.writeSolver()
        fcn.createSolver(fcn.target['solver'], fcn.weights, fcn.gpu)
        interp_layers = [k for k in fcn.solver.net.params.keys() if 'up' in k]
        caffeine.surgery.interp(fcn.solver.net, interp_layers)

    if utils.query_boolean('Do you want to test?'):
        fcn.weights = 'data/models/{}/snapshots/train_iter_{}.caffemodel'.format(conf.tag, lastiter)
        fcn.forwardList(list_=fcn.vallist)
        fcn.forwardList(list_=fcn.trainlist)


def main(args):
    sysargs = parseArgs()
    conf = loadConf(sysargs.conf)
    print(conf.net)
    runExperiment(sysargs, conf)


if __name__ == '__main__':
    main(sys.argv)
