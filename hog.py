#!/usr/bin/env python3
import ba.utils
from ba.baseline import Baseline

noti = ba.utils.NotifierClass()

tags = [
    'pottedplant_plant',
    'bottle_body',
    'aeroplane_lwing_rwing',
    'person_head',
    'train_hfrontside',
    'train_head',
    'person_hair'
    ]


def run(tag):
    baseline = Baseline(tag)
    for _ in range(10):
        baseline.run(1)
        baseline.run(10)
    for _ in range(4):
        baseline.run(25)
    for _ in range(2):
        baseline.run(50)
    baseline.run(100)
    noti.notify('Finished baseline for {}'.format(tag))


if __name__ == '__main__':
    for tag in tags:
        run(tag)
