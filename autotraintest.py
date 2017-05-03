#!/usr/bin/env python3
import json
import ba.utils
from ba.experiment import Experiment
import multiprocessing as mp
from itertools import count

GPUS = [0]
TMPEXP = './data/tmp/experiments/'
PPTMP = 'data/tmp/pascpart/'

noti = ba.utils.NotifierClass()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def run(fpath):
    with open(fpath) as f:
        parts = json.load(f)

    parts_per_gpu = int(len(parts) / len(GPUS))

    for idx, partschunk in zip(count(), chunks(parts, parts_per_gpu)):
        runIter(partschunk, GPUS[idx])
        # p = mp.Process(target=runIter, args=(partschunk, GPUS[idx]))
        # p.start()


def runIter(parts, gpu):
    for part in parts:
        runSingle(classes=part['classes'],
                  parts=part['shortparts'], gpu=gpu)


def buildTag(classes, parts):
    classes = [c.lower() for c in classes]
    parts = [p.lower() for p in parts]
    classes.sort()
    parts.sort()

    return '_'.join(classes + parts)


def writeYAMLS(tag):
    fc_path = TMPEXP + tag + '.yaml'
    fc_dict = ba.utils.load('./data/experiments/base_fc.yaml')
    fc_dict['slicefile'] = '{}patches/{}/seg.yaml'.format(PPTMP, tag)
    fc_dict['test'] = '{}{}.txt'.format(PPTMP, tag)
    fc_dict['train'] = '{}{}.txt'.format(PPTMP, tag)
    fc_dict['val'] = '{}patches/{}/img_augmented_lmdb_test.txt'.format(
        PPTMP, tag)
    ba.utils.save(fc_path, fc_dict)

    fcn_path = TMPEXP + tag + '_FCN.yaml'
    fcn_dict = ba.utils.load('./data/experiments/base_fcn.yaml')
    fcn_dict['slicefile'] = '{}patches/{}/seg.yaml'.format(PPTMP, tag)
    fcn_dict['labels'] = '{}patches/{}/seg.yaml'.format(PPTMP, tag)
    fcn_dict['test'] = '{}{}.txt'.format(PPTMP, tag)
    ba.utils.save(fcn_path, fcn_dict)


def runSingle(classes, parts, gpu):
    tag = buildTag(classes, parts)
    writeYAMLS(tag)
    argv = ['--gpu', str(gpu), '--repeat', '--train', '--test', '--tofcn',
            TMPEXP + tag + '.yaml', '--default']
    e = Experiment(argv)
    e.run()
    noti.notify('Run ' + tag)


if __name__ == '__main__':
    run('./data/parts_second_run_3.json')
