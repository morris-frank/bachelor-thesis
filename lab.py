#!/usr/bin/env python3
import ba.experiment as experiment
import sys


def main(args):
    sysargs = experiment.parseArgs(argv=args)
    conf = experiment.loadConf(sysargs.conf)
    if sysargs.train:
        experiment.runTrain(sysargs, conf)
    if sysargs.test:
        experiment.runTests(sysargs, conf)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No arguments given')
        sys.exit()
    main(sys.argv[1:])
