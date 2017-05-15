#!/usr/bin/env python3
from ba.experiment import Experiment

EXPF = './data/experiments/search.yaml'
SEGYAML = 'data/tmp/search_seg.yaml'
SINGLEF = 'data/tmp/search_list.txt'


def write_seg_yaml(idx, rect):
    with open(SINGLEF, mode='w') as f:
        f.write(idx + '\n')
    with open(SEGYAML, mode='w') as f:
        prefix = '  - !!python/object/apply:builtins.slice '
        f.write("'{}':\n".format(idx))
        f.write("- !!python/tuple\n")
        f.write("{}[{}, {}, null]\n".format(prefix, rect[0], rect[2]))
        f.write("{}[{}, {}, null]\n".format(prefix, rect[1], rect[3]))


def runSingle(idx, rect, gpu):
    write_seg_yaml(idx, rect)
    argv = ['--gpu', str(gpu), '--train', '--test', '--tofcn',
            EXPF, '--default', '--quiet']
    e = Experiment(argv)
    e.run()


if __name__ == '__main__':
    idx = '2008_000008'
    rect = [41, 86, 158, 205]
    runSingle(idx, rect, 0)
