#!/usr/bin/env python3
from ba.experiment import Experiment
import sys


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No arguments given')
        sys.exit()
    e = Experiment(sys.argv[1:])
    e.run()
