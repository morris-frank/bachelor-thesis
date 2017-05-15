#!/usr/bin/env python3
from glob import glob

bn = 'data/tmp/baseline/{}_{}samples*.yaml'
bn_mp = 'data/tmp/baseline/{}_{}samples*.yaml'

ns = [1, 10, 25, 50, 100]
ks = [10, 10, 4, 2, 1]

tags = ['bird_head', 'cow_sheep_lhorn_rhorn', 'train_hfrontside',
        'aeroplane_lwing_rwing', 'bird_head', 'person_lear_rear',
        'bottle_body', 'train_coach', 'person_neck', 'train_head', 'bird_tail',
        'person_torso', 'aeroplane_stern', 'pottedplant_plant',
        'bicycle_bwheel_fwheel', 'bird_lwing_rwing', 'person_torso',
        'car_door', 'person_head', 'bird_beak', 'person_lfoot_rfoot',
        'person_lhand_rhand', 'person_hair',
        'motorbike_bwheel_fwheel']

for tag in tags:
    y_complete = True
    m_complete = True
    for n, k in zip(ns, ks):
        m_k = k - len(glob(bn_mp.format(tag, n)))
        if m_k > 0:
            m_complete = False
        y_k = k - len(glob(bn.format(tag, n)))
        if y_k > 0:
            y_complete = False
    if m_complete:
        print('{}'.format(tag))
