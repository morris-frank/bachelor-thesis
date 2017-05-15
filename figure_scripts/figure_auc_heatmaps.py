#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas
import ba.plt

cmap = sns.cubehelix_palette(100, start=2.1, rot=-0.2, gamma=0.6,
                             as_cmap=True)


sns.set_context('paper')


def process_results_json(results_json):
    results = pandas.DataFrame(results_json).T
    results.columns = results.columns.astype(int)
    results.sort_index(axis=1, inplace=True)
    meanrow = pandas.DataFrame(results.mean()).T
    meanrow.index = ['Mean']
    return results.append(meanrow)


with open('./auc_var.json') as f:
    result_json = json.load(f)
results_var = process_results_json(result_json)
results_var = results_var.drop(100, 1).drop(50, 1)

with open('./auc_mean.json') as f:
    result_json = json.load(f)
results_mean = process_results_json(result_json)


fig, ax = ba.plt.newfig(0.97, 1.9)
ax = sns.heatmap(results_var, annot=True, cbar=False, ax=ax, linewidths=0,
                 cmap=cmap)
ba.plt.savefig('build/auc_var_heatmap')
plt.show()

fig, ax = ba.plt.newfig(0.97, 1.9)
ax = sns.heatmap(results_mean, annot=True, cbar=False, ax=ax, linewidths=0,
                 cmap=cmap)
ba.plt.savefig('build/auc_heatmap')
plt.show()
