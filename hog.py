#!/usr/bin/env python3
import sys
import ba.utils
from ba.baseline import Baseline
from glob import glob

noti = ba.utils.NotifierClass()


def get_count(tag, n):
    return len(glob('{}*{}samples*yaml'.format(tag, n)))


def run(tag):
    baseline = Baseline(tag)
    baseline.run(100)
    # for _ in range(max(0, 10 - get_count(tag, 1))):
    #     baseline.run(1)
    # for _ in range(max(0, 10 - get_count(tag, 10))):
    #     baseline.run(10)
    # for _ in range(max(0, 4 - get_count(tag, 25))):
    #     baseline.run(25)
    # for _ in range(max(0, 2 - get_count(tag, 50))):
    #     baseline.run(50)
    # for _ in range(max(0, 1 - get_count(tag, 100))):
    #     baseline.run(100)
    # noti.notify('Finished baseline for {}'.format(tag))


if __name__ == '__main__':
    run(sys.argv[1])
