from __future__ import division
import caffe
import numpy as np


def convertToFCN(new_net, old_net, new_params, old_params, path):
    fc_params = {pr: (old_net.params[pr][0].data, old_net.params[pr][1].data) for pr in old_params}
    for fc in old_params:
        print('{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape))

    conv_params = {pr: (new_net.params[pr][0].data, new_net.params[pr][1].data) for pr in new_params}
    for conv in new_params:
        print('{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape))

    for pr, pr_conv in zip(old_params, new_params):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    new_net.save(path)


def transplant(new_net, net, suffix=''):
    '''
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.

    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.

    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    '''
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print('dropping' + p)
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print('dropping', p, i)
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print('coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape)
            else:
                print('copying', p, ' -> ', p_new, i)
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat


def upsample_filt(size):
    '''
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def interp(net, layers):
    '''
    Set weights of each layer in layers to bilinear kernels for interpolation.
    '''
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print('input + output channels need to be the same or |output| == 1')
            raise
        if h != w:
            print('filters need to be square')
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


def expand_score(new_net, new_layer, net, layer):
    '''
    Transplant an old score layer's parameters, with k < k' classes, into a new
    score layer with k classes s.t. the first k' are the old classes.
    '''
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data
