import caffe
from caffe import layers as L
from caffe import params as P
from caffe.coord_map import crop
from ba.caffeine.utils import *
from os.path import dirname


class VGG16(object):
    def __init__(self, nclasses=2):
        self.n = caffe.NetSpec()
        self.nclasses = nclasses
        self.params = {}
        self.switches = {'learn_fc': False, 'tofcn': False}

    def base_net(self):
        # the base net
        self.n.conv1_1, self.n.relu1_1 = conv_relu(self.n.data, 64, pad=100,
                                                   lrmult=0)
        self.n.conv1_2, self.n.relu1_2 = conv_relu(self.n.relu1_1, 64,
                                                   lrmult=0)
        self.n.pool1 = max_pool(self.n.relu1_2)

        self.n.conv2_1, self.n.relu2_1 = conv_relu(self.n.pool1, 128, lrmult=0)
        self.n.conv2_2, self.n.relu2_2 = conv_relu(self.n.relu2_1, 128,
                                                   lrmult=0)
        self.n.pool2 = max_pool(self.n.relu2_2)

        self.n.conv3_1, self.n.relu3_1 = conv_relu(self.n.pool2, 256, lrmult=0)
        self.n.conv3_2, self.n.relu3_2 = conv_relu(self.n.relu3_1, 256,
                                                   lrmult=0)
        self.n.conv3_3, self.n.relu3_3 = conv_relu(self.n.relu3_2, 256,
                                                   lrmult=0)
        self.n.pool3 = max_pool(self.n.relu3_3)

        self.n.conv4_1, self.n.relu4_1 = conv_relu(self.n.pool3, 512, lrmult=0)
        self.n.conv4_2, self.n.relu4_2 = conv_relu(self.n.relu4_1, 512,
                                                   lrmult=0)
        self.n.conv4_3, self.n.relu4_3 = conv_relu(self.n.relu4_2, 512,
                                                   lrmult=0)
        self.n.pool4 = max_pool(self.n.relu4_3)

        self.n.conv5_1, self.n.relu5_1 = conv_relu(self.n.pool4, 512, lrmult=0)
        self.n.conv5_2, self.n.relu5_2 = conv_relu(self.n.relu5_1, 512,
                                                   lrmult=0)
        self.n.conv5_3, self.n.relu5_3 = conv_relu(self.n.relu5_2, 512,
                                                   lrmult=0)
        self.n.pool5 = max_pool(self.n.relu5_3)

    def data(self):
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n.data, self.n.label = L.Data(batch_size=8,
                                               source=self.params['lmdb'],
                                               backend=P.Data.LMDB, ntop=2,
                                               transform_param=dict(
                                                   crop_size=224, mirror=True,
                                                   mean_value=list(
                                                       self.params['mean'])))
        else:
            self.n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])

    def tail(self):
        # fully conv
        fc6, self.n.relu6 = fc(self.n.pool5, std=0.05, lrmult=1)
        self.n.drop6 = L.Dropout(self.n.relu6, dropout_ratio=0.5,
                                 in_place=True)

        fc7, self.n.relu7 = fc(self.n.drop6, std=0.05, lrmult=1)
        self.n.drop7 = L.Dropout(self.n.relu7, dropout_ratio=0.5,
                                 in_place=True)

        if self.switches['learn_fc']:
            self.n.fc6_ = fc6
            self.n.fc7_ = fc7
        else:
            self.n.fc6 = fc6
            self.n.fc7 = fc7

        self.n.fc8_, _ = fc(self.n.drop7, nout=self.nclasses, std=0.01,
                            lrmult=10)
        # self.n.prob = L.Softmax(self.n.fc8_)

        if 'deploy' != self.params['split']:
            if 'test' == self.params['split']:
                self.n.accuracy = L.Accuracy(self.n.fc8_, self.n.label)
            else:
                self.n.loss = L.SoftmaxWithLoss(self.n.fc8_, self.n.label)

    def write(self, params={}, switches={}):
        '''Builds the VGG16 Network.

        Args:
            params (dict): parameter for the network and the pylayer
            switsches (dict): Contains boolean switches for the network

        Retruns:
            The network as prototxt
        '''
        self.params.update(params)
        self.switches.update(switches)
        self.data()
        self.base_net()
        self.tail()
        return self.n.to_proto()


class FCN32s(VGG16):
    def __init__(self, nclasses=2):
        super().__init__(nclasses=nclasses)

    def data(self):
        pylayer = 'SegDataLayer'
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n.data, self.n.label = L.Python(
                module='ba.caffeine.voc_layers',
                layer=pylayer,
                ntop=2,
                param_str=str(self.params)
                )
        else:
            self.n.data = L.Input(shape=[dict(dim=[1, 3, 500, 500])])

    def tail(self):
        if self.switches['tofcn']:
            na = '_conv'
        elif self.switches['learn_fc']:
            na = '_'
        else:
            na = ''
        # fully conv
        self.n['fc6' + na], self.n.relu6 = conv_relu(
            self.n.pool5, 4096, ks=7, pad=0, lrmult=0)
        self.n.drop6 = L.Dropout(
            self.n.relu6, dropout_ratio=0.5, in_place=True)

        self.n['fc7' + na], self.n.relu7 = conv_relu(
            self.n.drop6, 4096, ks=1, pad=0, lrmult=0)
        self.n.drop7 = L.Dropout(self.n.relu7, dropout_ratio=0.5,
                                 in_place=True)

        self.n.score_fr_ = L.Convolution(
            self.n.drop7, num_output=self.nclasses, kernel_size=1, pad=0,
            param=[dict(lr_mult=1, decay_mult=1),
                   dict(lr_mult=2, decay_mult=0)])

        if self.switches['tofcn']:
            na = ''
        else:
            na = '_'
        self.n['upscore' + na] = upsample(self.n.score_fr_, factor=32,
                                          nout=self.nclasses)

        self.n.score = crop(self.n['upscore' + na], self.n.data)

        if self.params['split'] != 'deploy':
            self.n.loss = L.SoftmaxWithLoss(self.n.score, self.n.label,
                                            loss_param=dict(normalize=False,
                                                            ignore_label=255))


class FCN32_PosPatch(FCN32s):
    def __init__(self, nclasses=2):
        super().__init__(nclasses=nclasses)

    def data(self):
        pylayer = 'PosPatchDataLayer'
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n.data, self.n.label = L.Python(
                module='ba.caffeine.voc_layers',
                layer=pylayer,
                ntop=2,
                param_str=str(self.params)
                )
        else:
            self.n.data = L.Input(shape=[dict(dim=[1, 3, 500, 500])])



class FCN8s(FCN32s):
    def __init__(self, nclasses=2):
        super().__init__(nclasses=nclasses)

    def tail(self):
        if self.switches['tofcn']:
            na = '_conv'
        elif self.switches['learn_fc']:
            na = '_'
        else:
            na = ''
        # fully conv
        self.n['fc6' + na], self.n.relu6 = conv_relu(self.n.pool5, 4096, ks=7,
                                                     pad=0, lrmult=0)
        self.n.drop6 = L.Dropout(self.n.relu6, dropout_ratio=0.5,
                                 in_place=True)

        self.n['fc7' + na], self.n.relu7 = conv_relu(self.n.drop6, 4096, ks=1,
                                                     pad=0, lrmult=0)
        self.n.drop7 = L.Dropout(self.n.relu7, dropout_ratio=0.5,
                                 in_place=True)

        # From scratch:
        self.n.score_fr_ = L.Convolution(self.n.drop7,
                                         num_output=self.nclasses,
                                         kernel_size=1, pad=0,
                                         param=[dict(lr_mult=1, decay_mult=1),
                                                dict(lr_mult=2, decay_mult=0)])
        # From scratch:
        self.n.upscore2_ = L.Deconvolution(self.n.score_fr_,
                                           convolution_param=dict(
                                               num_output=self.nclasses,
                                               kernel_size=4, stride=2,
                                               bias_term=False),
                                           param=[dict(lr_mult=0)])

        # From scratch:
        self.n.score_pool4_ = L.Convolution(self.n.pool4,
                                            num_output=self.nclasses,
                                            kernel_size=1, pad=0,
                                            param=[dict(lr_mult=1,
                                                        decay_mult=1),
                                                   dict(lr_mult=2,
                                                        decay_mult=0)])
        self.n.score_pool4c = crop(self.n.score_pool4_, self.n.upscore2_)
        self.n.fuse_pool4 = L.Eltwise(self.n.upscore2_, self.n.score_pool4c,
                                      operation=P.Eltwise.SUM)
        # From scratch:
        self.n.upscore_pool4_ = L.Deconvolution(self.n.fuse_pool4,
                                                convolution_param=dict(
                                                    num_output=self.nclasses,
                                                    kernel_size=4, stride=2,
                                                    bias_term=False),
                                                param=[dict(lr_mult=0)])

        # From scratch
        self.n.score_pool3_ = L.Convolution(self.n.pool3,
                                            num_output=self.nclasses,
                                            kernel_size=1, pad=0,
                                            param=[dict(lr_mult=1,
                                                        decay_mult=1),
                                                   dict(lr_mult=2,
                                                        decay_mult=0)])
        self.n.score_pool3c = crop(self.n.score_pool3_, self.n.upscore_pool4_)
        self.n.fuse_pool3 = L.Eltwise(self.n.upscore_pool4_,
                                      self.n.score_pool3c,
                                      operation=P.Eltwise.SUM)
        # From scratch
        self.n.upscore8_ = L.Deconvolution(self.n.fuse_pool3,
                                           convolution_param=dict(
                                               num_output=self.nclasses,
                                               kernel_size=16, stride=8,
                                               bias_term=False),
                                           param=[dict(lr_mult=0)])

        self.n.score = crop(self.n.upscore8_, self.n.data)

        if self.params['split'] != 'deploy':
            self.n.loss = L.SoftmaxWithLoss(self.n.score, self.n.label,
                                            loss_param=dict(normalize=False,
                                                            ignore_label=255))
