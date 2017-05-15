#!/usr/bin/env python3
import pandas as pd
import numpy as np
import ba.plt
import matplotlib.pyplot as plt
import seaborn as sns

samples = {
    'aeroplane_engine': 375,
    'aeroplane_lwing_rwing': 530,
    'aeroplane_stern': 527,
    'bicycle_bwheel_fwheel': 459,
    'bird_beak': 590,
    'bird_head': 647,
    'bird_lwing_rwing': 299,
    'bird_tail': 496,
    'bottle_body': 585,
    'bus_car_bliplate_fliplate': 663,
    'car_door': 414,
    'cow_sheep_lhorn_rhorn': 127,
    'motorbike_bwheel_fwheel': 434,
    'person_hair': 2752,
    'person_head': 3463,
    'person_lear_rear': 2492,
    'person_lfoot_rfoot': 1718,
    'person_lhand_rhand': 2927,
    'person_neck': 2230,
    'person_torso': 3504,
    'pottedplant_plant': 451,
    'train_coach': 334,
    'train_head': 311,
    'train_hfrontside': 265
    }


sns.set_context('paper')

reduce_tag = lambda x: x.split('_FCN_')[0]
timings = pd.read_csv('./timings.csv', sep=';')
timings['tag'] = timings['tag'].map(reduce_tag)
timings = timings.loc[timings['duration'] > 5]

train_timings = timings.loc[timings['function'] == '_train']
train_timings = train_timings.loc[train_timings['duration'] > 20]
train_timings = train_timings.loc[train_timings['duration'] < 300]
train_times = np.array(train_timings['duration'])


def image_wise_duration(x):
    x['duration'] /= samples[x['tag']]
    x['duration'] *= 1000
    return x


test_timings = timings.loc[timings.tag.str.contains('sample') == False]
train_timings = test_timings.loc[test_timings['duration'] < 100]
test_timings = test_timings.loc[timings['function'] == '_conv_test']
test_timings = test_timings.loc[timings['tag'] != 'search']
test_timings = test_timings.apply(image_wise_duration, axis=1)
test_times = np.array(test_timings['duration'])

fig = plt.figure(figsize=ba.plt.figsize(1, 0.6))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Testing timings')
ax1.set_ylabel('duration per image in ms')
sns.violinplot(x=test_times, ax=ax1, palette='Set2', scale='width')

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Training timings')
ax2.set_ylabel('duration per training in s')
sns.violinplot(x=train_times, ax=ax2, palette='Set2', scale='width')
ba.plt.savefig('./build/timings_fig')
