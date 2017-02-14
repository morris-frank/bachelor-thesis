import caffe
from caffe import layers as L
from caffe import params as P


class ReseNet50(object):

    def __init__(self):
        self.n = caffe.NetSpec()
        self.n['data'] = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
        self.n['conv1'] = L.Convolution(
            self.n['data'],
            name='conv1',
            num_output=64,
            kernel_size=7,
            stride=2,
            pad=3
            )
        self.bn('conv1', name='bn_conv1')
        self.scale('bn_conv1', name='scale_conv1')
        self.relu('scale_conv1', name='conv1_relu')
        self.n['pool1'] = L.Pooling(
            self.n['conv1_relu'],
            name='pool1',
            kernel_size=3,
            stride=2,
            pool=P.Pooling.MAX
            )
        res2c = self.res('pool1', 3, '2', 64, ds=False)
        res3d = self.res(res2c, 4, '3', 128)
        res4f = self.res(res3d, 6, '4', 256)
        res5c = self.res(res4f, 3, '5', 512, train=Trues)
        self.n['pool5'] = L.Pooling(
            self.n['res5c'],
            name='pool5',
            kernel_size=7,
            stride=1,
            pool=P.Pooling.AVE
            )

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
        s = self.cbs(bname, bottom, nout, ks, pad, stride, train)
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
            param=[dict(lr_mult=1 * train, decay_mult=1),
                   dict(lr_mult=1 * train, decay_mult=1)]
            )
        self.bn('res' + bname, name='bn' + bname)
        self.scale('bn' + bname, name='scale' + bname)
        return 'scale' + bname

    def branch2(self, bottom, resname, startout, ds=False, train=False):
        s = 2 if ds else 1
        ra = self.cbsr(resname + '_branch2a', bottom, startout, ks=1, pad=0, stride=s, train)
        rb = self.cbsr(resname + '_branch2b', ra, startout, ks=3, pad=1, train)
        sc = self.cbs(resname + '_branch2c', rb, startout * 4, ks=1, pad=0, train)
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
        right = self.branch2(bottom, name + 'a', nout, ds=False, train)
        n_elt = 'res{}'.format(name)
        self.n[n_elt] = L.Eltwise(self.n[bottom], self.n[right], name=n_elt)
        self.relu(n_elt, name='res{}_relu'.format(name))
        return 'res{}_relu'.format(name)

    def res(self, bottom, nblocks, name, nout, ds=True, train=False):
        alph = ['a', 'b', 'c', 'd', 'e', 'f']
        last = self.resprimblock(bottom, name + 'a', nout, ds)
        for i in range(1, nblocks):
            if i == nblocks:
                last = self.ressecblock(last, name + alph[i], nout, train=True)
            else:
                last = self.ressecblock(last, name + alph[i], nout)
        return last

    def write(self, params, switches):
        nclasses = 2

        if params['split'] == 'train' or params['split'] == 'val':
            n.data, n.label = L.Data(
                batch_size=8,
                source=params['lmdb'],
                backend=P.Data.LMDB,
                ntop=2,
                transform_param=dict(
                    crop_size=224,
                    mirror=True,
                    mean_file="data/models/resnet/ResNet_mean.binaryproto"
                    )
                )

        self.n['fc'] = L.InnerProduct(
            self.n.pool5,
            inner_product_param=dict(
                num_output=nclasses
                weight_filler=dict(type='gaussian', std=0.01),
                bias_filler=dict(type='constant', value=0),
                ),
            param=[
                dict(lr_mult=10, decay_mult=10),
                dict(lr_mult=10, decay_mult=0)
                ]
            )

        self.n['prob'] = L.Softmax(self.n['fc'])

        if 'deploy' != params['split']:
            if 'test' == params['split']:
                self.n.accuracy = L.Accuracy(self.n['fc'], n.label)
            else:
                self.n.loss = L.SoftmaxWithLoss(self.n['fc'], n.label)

        with open('test.txt', 'w') as f:
            f.write(str(self.n.to_proto()))
        # return self.n.to_proto()
