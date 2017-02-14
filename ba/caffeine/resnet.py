import caffe
from caffe import layers as L
from caffe import params as P


def cbsr(bottom, nout, ks, pad, stride=1):
    s = cbs(nout, ks, pad, stride)
    r = L.ReLU(s, in_place=True)
    return r


def cbs(bottom, nout, ks, pad, stride=1):
    c = L.Convolution(
        bottom,
        num_output=nout,
        kernel_size=ksm,
        stride=stride,
        pad=pad,
        bias_term=False
        )
    b = L.BatchNorm(c, use_global_stats: True, in_place=True)
    s = L.Scale(b, bias_term=True, in_place=True)
    return s


def branch2(bottom, startout):
    b1 = cbsr(bottom, startout, ks=1, pad=0, )
    b2 = cbsr()
    b3 = cbs()
