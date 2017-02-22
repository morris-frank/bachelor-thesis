from ba.caffeine.utils import *
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.coord_map import crop


class ReseNet50(object):

    def __init__(self, nclasses=2):
        self.nclasses = nclasses
        self.n = caffe.NetSpec()
        self.params = {}
        self.switches = {'learn_fc': False, 'tofcn': False}

    def bn(self, bottom, name):
        self.n[name] = L.BatchNorm(
            self.n[bottom],
            name=name,
            use_global_stats=True,
            in_place=True
            )

    def scale(self, bottom, name):
        self.n[name] = L.Scale(
            self.n[bottom],
            name=name,
            bias_term=True,
            in_place=True
            )

    def relu(self, bottom, name):
        self.n[name] = L.ReLU(
            self.n[bottom],
            name=name,
            in_place=True
            )

    def cbsr(self, bname, bottom, nout, ks, pad, stride=1, train=False):
        s = self.cbs(bname, bottom, nout, ks, pad, stride, train=train)
        n_rel = 'res{}_relu'.format(bname)
        self.relu(s, n_rel)
        return n_rel

    def cbs(self, bname, bottom, nout, ks, pad, stride=1, train=False):
        self.n['res' + bname] = L.Convolution(
            self.n[bottom],
            name='res' + bname,
            num_output=nout,
            kernel_size=ks,
            stride=stride,
            pad=pad,
            bias_term=False,
            param=dict(lr_mult=1 * train, decay_mult=1)
            )
        self.bn('res' + bname, name='bn' + bname)
        self.scale('bn' + bname, name='scale' + bname)
        return 'scale' + bname

    def branch2(self, bottom, resname, startout, ds=False, train=False):
        s = 2 if ds else 1
        ra = self.cbsr(resname + '_branch2a', bottom, startout, ks=1, pad=0, stride=s, train=train)
        rb = self.cbsr(resname + '_branch2b', ra, startout, ks=3, pad=1, train=train)
        sc = self.cbs(resname + '_branch2c', rb, startout * 4, ks=1, pad=0, train=train)
        return sc

    def resprimblock(self, bottom, name, nout, ds=False):
        s = 2 if ds else 1
        left = self.cbs(name + '_branch1', bottom, nout * 4, ks=1, pad=0, stride=s)
        right = self.branch2(bottom, name, nout, ds)
        n_elt = 'res{}'.format(name)
        self.n[n_elt] = L.Eltwise(self.n[left], self.n[right], name=n_elt)
        self.relu(n_elt, name='res{}_relu'.format(name))
        return 'res{}_relu'.format(name)

    def ressecblock(self, bottom, name, nout, train=False):
        right = self.branch2(bottom, name, nout, ds=False, train=train)
        n_elt = 'res{}'.format(name)
        self.n[n_elt] = L.Eltwise(self.n[bottom], self.n[right], name=n_elt)
        self.relu(n_elt, name='res{}_relu'.format(name))
        return 'res{}_relu'.format(name)

    def res(self, bottom, nblocks, name, nout, ds=True, train=False):
        alph = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        last = self.resprimblock(bottom, name + 'a', nout, ds)
        for i in range(1, nblocks):
            if i == nblocks:
                last = self.ressecblock(last, name + alph[i], nout, train=train)
            else:
                last = self.ressecblock(last, name + alph[i], nout)
        return last

    def data(self):
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n['data'], self.n['label'] = L.Data(
                batch_size=18, source=self.params['lmdb'], backend=P.Data.LMDB,
                ntop=2, transform_param=dict(
                    crop_size=224,
                    mirror=True,
                    mean_file="data/models/resnet/ResNet_mean.binaryproto"
                    ))
        else:
            self.n['data'] = L.Input(shape=[dict(dim=[1, 3, 224, 224])])

    def base_net(self):
        self.n.conv1 = L.Convolution(
            self.n.data, num_output=64, kernel_size=7, stride=2, pad=3)
        self.bn('conv1', name='bn_conv1')
        self.scale('bn_conv1', name='scale_conv1')
        self.relu('scale_conv1', name='conv1_relu')
        self.n['pool1'] = L.Pooling(
            self.n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        res2c = self.res('pool1', 3, '2', 64, ds=False)
        res3d = self.res(res2c, 4, '3', 128)
        res4f = self.res(res3d, 6, '4', 256)
        res5c = self.res(res4f, 3, '5', 512, train=False)
        self.n.pool5 = L.Pooling(
            self.n['res5c'], kernel_size=7, stride=1, pool=P.Pooling.AVE)

    def tail(self):
        self.n.fc = L.InnerProduct(
            self.n.pool5, inner_product_param=dict(
                num_output=self.nclasses,
                weight_filler=dict(type='gaussian', std=0.01),
                bias_filler=dict(type='constant', value=0),
                ),
            param=dict(lr_mult=10, decay_mult=10))

        # self.n.prob = L.Softmax(self.n.fc)

        if 'deploy' != self.params['split']:
            if 'test' == self.params['split']:
                self.n.accuracy = L.Accuracy(self.n.fc, self.n.label)
            else:
                self.n.loss = L.SoftmaxWithLoss(self.n.fc, self.n.label)

    def write(self, params, switches):
        self.params.update(params)
        self.switches.update(switches)
        self.data()
        self.base_net()
        self.tail()
        return self.n.to_proto()


class ReseNet50FCN(ReseNet50):
    def __init__(self, nclasses=2):
        super().__init__(nclasses=nclasses)

    def data(self):
        mean_file="data/models/resnet/ResNet_mean.binaryproto"

        pylayer = 'PosPatchDataLayer'
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n.data, self.n.label = L.Python(
                module='ba.caffeine.datalayers',
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
        self.n['fc' + na] = L.Convolution(
            self.n.pool5, kernel_size=1, stride=1, num_output=self.nclasses,
            pad=0, param=[dict(lr_mult=1, decay_mult=1),
                          dict(lr_mult=2, decay_mult=0)])

        self.n['upscore' + na] = upsample(self.n['fc' + na], factor=32,
                                          nout=self.nclasses)

        # self.n.score = self.n['upscore' + na]
        # self.n.score = crop(self.n['upscore' + na], self.n.data)

        if self.params['split'] != 'deploy':
            self.n.loss = L.SoftmaxWithLoss(self.n.score, self.n.label,
                                            loss_param=dict(normalize=False,
                                                            ignore_label=255))


class ReseNet50_Single(ReseNet50):
    def __init__(self, nclasses =2):
        super().__init__(nclasses=nclasses)

    def data(self):
        pylayer = 'SingleImageLayer'
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n.data, self.n.label = L.Python(
                module='ba.caffeine.datalayers',
                layer=pylayer,
                ntop=2,
                param_str=str(self.params)
                )
        else:
            self.n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
