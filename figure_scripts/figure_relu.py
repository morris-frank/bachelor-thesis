#!/usr/bin/env python3
import numpy as np
import ba.plt
from scipy.misc import imread, imsave


im = imread('./2008_006433.jpg')
relu = np.load('./relu_1.npy')
response = np.load('./response_1.npy')
params = np.load('./params_1.npy')

imsave('./build/activation_data.png', im[60:180, 90:200, :])

relu = relu[30:90, 40:100]
response = response[30:90, 40:100]
im = im[60:100, 30:90, :]

_min = response.min()
_max = response.max()

ba.plt._prepareImagePlot(im)
ba.plt.plt.show()

ba.plt.plt_hm(params[:, :, 2])
ba.plt.savefig('./build/activation_filter')

ba.plt.plt_hm(relu, diverg=True, vmin=_min, vmax=_max)
ba.plt.savefig('./build/activation_relu')

ba.plt.plt_hm(response, diverg=True, vmin=_min, vmax=_max)
ba.plt.savefig('./build/activation_response')
