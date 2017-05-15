#!/usr/bin/env python3
import numpy as np
import ba.plt
import seaborn as sns

sns.set_context('paper')
sns.set_palette('Set2', 5)

relu = lambda x: np.maximum(x,0)
sigmoid = lambda x: 1 / (1 + np.exp( - x))

X = np.linspace(-6, 6, 100)

fig = ba.plt.plt.figure(figsize=ba.plt.figsize(1, 0.5))
ax = fig.add_subplot(121)
ba.plt.plt.plot(X, sigmoid(X))
ax2 = fig.add_subplot(122)
ba.plt.plt.plot(X, relu(X))
ba.plt.savefig('build/sigmoid_relu')
