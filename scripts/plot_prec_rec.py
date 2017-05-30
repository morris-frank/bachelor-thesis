#!/usr/bin/env python3
import ba.plt
import datetime
import json

pivotdate = datetime.date(2017, 5, 8)


tags = ['bird_head', 'cow_sheep_lhorn_rhorn', 'train_hfrontside',
        'aeroplane_lwing_rwing', 'person_lear_rear', 'bottle_body',
        'train_coach', 'person_neck', 'train_head', 'bird_tail',
        'aeroplane_stern', 'pottedplant_plant', 'bicycle_bwheel_fwheel',
        'bird_lwing_rwing', 'person_torso', 'car_door', 'person_head',
        'bird_beak', 'person_lfoot_rfoot', 'person_lhand_rhand',
        'person_hair', 'motorbike_bwheel_fwheel',
        'bus_car_bliplate_fliplate']

# tags = ['bird_head', 'bottle_body', 'cow_sheep_lhorn_rhorn', 'person_hair',
#         'person_neck', 'pottedplant_plant']

r = {}
for mode in ['PR', 'ROC']:
    for tag in tags:
        r[tag] = ba.plt.plt_results_for_tag(tag, mode, startdate=pivotdate)

with open('results.json', 'w') as f:
    json.dump(r, f)
