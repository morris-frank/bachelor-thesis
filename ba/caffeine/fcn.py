import caffe
from caffe import layers as L
from caffe import params as P
from caffe.coord_map import crop
from ba.caffeine.utils import *

def vgg16(params, switches):
    '''Builds the FCN8s Network.
        Based on code from Evan Shelhamer
        fcn.berkeleyvision.org

    Args:
        params (dict): parameter for the network and the pylayer
        switsches (dict): Contains boolean switches for the network

    Retruns:
        The network as prototxt
    '''
    nclasses = 2
    n = caffe.NetSpec()
    pylayer = 'SegDataLayer'
    if 'train' in params['splitfile'] or 'val' in params['splitfile']:
        n.data, n.label = L.Python(module='ba.caffeine.voc_layers',
                                   layer=pylayer,
                                   ntop=2,
                                   param_str=str(params))
    else:
        n.data = L.Input(shape=[dict(dim=[1,3,500,500])])

    if 'learn_fc' not in switches:
        switches['learn_fc'] = False

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100, lrmult=0)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, lrmult=0)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, lrmult=0)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, lrmult=0)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, lrmult=0)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, lrmult=0)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, lrmult=0)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, lrmult=0)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, lrmult=0)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, lrmult=0)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, lrmult=0)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, lrmult=0)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, lrmult=0)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    # TODO: -BUILD REAL FCs
    if switches['learn_fc']:
        n.fc6_ = L.Inner_Product(n.pool5,
                                 inner_product_param=dict(num_output=4096))
        n.relu6 = L.ReLU(n.fc6_)
    else:
        n.fc6 = L.Inner_Product(n.pool5,
                                 inner_product_param=dict(num_output=4096))
        n.relu6 = L.ReLU(n.fc6)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)

    if switches['learn_fc']:
        n.fc7_ = L.Inner_Product(n.drop6,
                                 inner_product_param=dict(num_output=4096))
        n.relu7 = L.ReLU(n.fc7_)
    else:
        n.fc7 = L.Inner_Product(n.drop6,
                                 inner_product_param=dict(num_output=4096))
        n.relu7 = L.ReLU(n.fc7)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.fc8_ = L.Inner_Product(n.drop7,
                            inner_product_param=dict(num_output=nclasses))

    if 'deploy' not in params['splitfile']:
        n.loss = L.SoftmaxWithLoss(n.fc8_, n.label)
    else:
        n.score = L.Softmax(n.fc8_, n.score)

    return n.to_proto()


def fcn8s(params, switches):
    '''Builds the FCN8s Network.
        Based on code from Evan Shelhamer
        fcn.berkeleyvision.org

    Args:
        params (dict): parameter for the network and the pylayer
        switsches (dict): Contains boolean switches for the network

    Retruns:
        The network as prototxt
    '''
    nclasses = 2
    n = caffe.NetSpec()
    pylayer = 'SegDataLayer'
    if 'train' in params['splitfile'] or 'val' in params['splitfile']:
        n.data, n.label = L.Python(module='ba.caffeine.voc_layers',
                                   layer=pylayer,
                                   ntop=2,
                                   param_str=str(params))
    else:
        n.data = L.Input(shape=[dict(dim=[1,3,500,500])])

    if 'learn_fc' not in switches:
        switches['learn_fc'] = False

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100, lrmult=0)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, lrmult=0)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, lrmult=0)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, lrmult=0)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, lrmult=0)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, lrmult=0)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, lrmult=0)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, lrmult=0)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, lrmult=0)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, lrmult=0)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, lrmult=0)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, lrmult=0)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, lrmult=0)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    if switches['learn_fc']:
        n.fc6_, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0, lrmult=0)
    else:
        n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0, lrmult=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    if switches['learn_fc']:
        n.fc7_, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0, lrmult=0)
    else:
        n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0, lrmult=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    #From scratch:
    n.score_fr_ = L.Convolution(n.drop7, num_output=nclasses, kernel_size=1,
                                pad=0, param=[dict(lr_mult=1, decay_mult=1),
                                              dict(lr_mult=2, decay_mult=0)])
    #From scratch:
    n.upscore2_ = L.Deconvolution(n.score_fr_,
                                  convolution_param=dict(num_output=nclasses,
                                                         kernel_size=4,
                                                         stride=2,
                                                         bias_term=False),
                                  param=[dict(lr_mult=0)])

    #From scratch:
    n.score_pool4_ = L.Convolution(n.pool4, num_output=nclasses, kernel_size=1,
                                   pad=0, param=[dict(lr_mult=1, decay_mult=1),
                                                 dict(lr_mult=2, decay_mult=0)])
    n.score_pool4c = crop(n.score_pool4_, n.upscore2_)
    n.fuse_pool4 = L.Eltwise(n.upscore2_, n.score_pool4c,
                             operation=P.Eltwise.SUM)
    #From scratch:
    n.upscore_pool4_ = L.Deconvolution(n.fuse_pool4, convolution_param=dict(
                                       num_output=nclasses, kernel_size=4,
                                       stride=2, bias_term=False),
                                       param=[dict(lr_mult=0)])

    #From scratch
    n.score_pool3_ = L.Convolution(n.pool3, num_output=nclasses, kernel_size=1,
                                   pad=0, param=[dict(lr_mult=1, decay_mult=1),
                                                 dict(lr_mult=2, decay_mult=0)])
    n.score_pool3c = crop(n.score_pool3_, n.upscore_pool4_)
    n.fuse_pool3 = L.Eltwise(n.upscore_pool4_, n.score_pool3c,
                             operation=P.Eltwise.SUM)
    # From scratch
    n.upscore8_ = L.Deconvolution(n.fuse_pool3,
                                  convolution_param=dict(num_output=nclasses,
                                                         kernel_size=16,
                                                         stride=8,
                                                         bias_term=False),
                                  param=[dict(lr_mult=0)])

    n.score = crop(n.upscore8_, n.data)

    if 'deploy' not in params['splitfile']:
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
                                   loss_param=dict(normalize=False,
                                                   ignore_label=255))

    return n.to_proto()
