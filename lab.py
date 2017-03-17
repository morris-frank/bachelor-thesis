#!/usr/bin/env python3
from ba.experiment import Experiment
import sys

DATASET = 'pascpart'
DSSSOURCE = 'data/datasets/pascalparts/Annotations_Part/'

def main(args):
    e = Experiment(argv=args)
    e.loadConf()
    if e.sysargs.data:
        e.genData(DATASET, DSSSOURCE)
    if e.sysargs.cascade:
        if e.sysargs.prepare:
            e.cascadePrepare()
        if e.sysargs.tofcn:
            e.cascadeToFCN()
        if e.sysargs.train:
            e.cascadeTraining()
        if e.sysargs.test:
            e.cascadeTesting()
        if not any([e.sysargs.train, e.sysargs.test,
                   e.sysargs.tofcn, e.sysargs.prepare]):
            print('Switch cascade needs switch train, test, tofcn or prepare.')
    else:
        if e.sysargs.prepare:
            e.prepare()
        if e.sysargs.tofcn:
            e.convertToFCN()
        if e.sysargs.train:
            e.runTrain()
        if e.sysargs.test:
            e.runTests()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No arguments given')
        sys.exit()
    main(sys.argv[1:])
