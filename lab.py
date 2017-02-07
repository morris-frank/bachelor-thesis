#!/usr/bin/env python3
from ba.experiment import Experiment
import sys


def main(args):
    e = Experiment(argv=args)
    e.loadConf()
    if e.sysargs.train:
        e.runTrain()
    if e.sysargs.test:
        e.runTests()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No arguments given')
        sys.exit()
    main(sys.argv[1:])
