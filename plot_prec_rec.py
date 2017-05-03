#!/usr/bin/env python3
import ba.plt

tags = ['bird_head', 'cow_sheep_lhorn_rhorn', 'train_hfrontside',
        'aeroplane_lwing_rwing', 'bird_head', 'person_lear_rear',
        'bottle_body', 'train_coach', 'person_neck', 'train_head', 'bird_tail',
        'person_torso', 'aeroplane_stern', 'pottedplant_plant']

for mode in ['PR', 'ROC', 'AUC']:
    for tag in tags:
        ba.plt.plt_results_for_tag(tag, mode)
