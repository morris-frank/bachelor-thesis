'''
    Based on code from Evan Shelhamer
    fcn.berkeleyvision.org
'''
from caffe import layers as L, params as P
import warnings

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# def make_net(callback_net, path='./'):
#     if not callable(callback_net):
#         warnings.warn('Not callable object')
#         return
#     with open(path + 'train.prototxt', 'w') as f:
#         f.write(str(callback_net('train')))
#
#     with open(path + 'val.prototxt', 'w') as f:
#         f.write(str(callback_net('val')))
#
#     # with open(path + 'deploy.prototxt', 'w') as f:
#     #     f.write(str(callback_net('deploy')))
