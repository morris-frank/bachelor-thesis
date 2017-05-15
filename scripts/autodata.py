#!/usr/bin/env python3
import json
from ba.data import Generator
import multiprocessing as mp

TMPEXP = './data/tmp/experiments/'
PPTMP = 'data/tmp/pascpart/'
PSIZE = 8


def run(fpath):
    with open(fpath) as f:
        parts = json.load(f)

    with mp.Pool(PSIZE) as p:
        p.map(runSingle, parts)


def runSingle(d):
    parts = ' '.join(d['shortparts'])
    classes = ' '.join(d['classes'])
    argv = ['--classes', classes, '--parts', parts, '--default']
    gen = Generator(argv)
    gen.run()


if __name__ == '__main__':
    run('./data/parts.json')
