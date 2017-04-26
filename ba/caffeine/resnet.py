import caffe
from caffe import layers as L
from caffe import params as P


class ResNet(object):
    '''Contains a standard Residual Net.'''
    def __init__(self, nclasses=2, nconv4=6, nconv3=4):
        self.nclasses = nclasses
        self.nconv3 = nconv3
        self.nconv4 = nconv4
        self.n = caffe.NetSpec()
        self.params = {}
        self.switches = {'learn_fc': False, 'tofcn': False}

    def BatchNorm(self, bottom, name):
        '''Adds a BatchNorm layer.

        Args:
            bottom (str): Name of the bottom blob
            name (str): Name of the top blob
        '''
        self.n[name] = L.BatchNorm(
            self.n[bottom],
            name=name,
            use_global_stats=True,
            in_place=True
            )

    def Scale(self, bottom, name):
        '''Adds a Scale layer.

        Args:
            bottom (str): Name of the bottom blob
            name (str): Name of the top blob
        '''
        self.n[name] = L.Scale(
            self.n[bottom],
            name=name,
            bias_term=True,
            in_place=True
            )

    def ReLU(self, bottom, name):
        '''Adds a Rectified Linear Unit layer.

        Args:
            bottom (str): Name of the bottom blob
            name (str): Name of the top blob
        '''
        self.n[name] = L.ReLU(
            self.n[bottom],
            name=name,
            in_place=True
            )

    def ConvReLUBlock(self, bname, bottom, nout, ks, pad, stride=1,
                      train=False):
        '''Adds a Block with a conv layer an BatchNorm layer a scale layer
        and a ReLU.

        Args:
            bname (str): Name of the top blob
            bottom (str): Name of the bottom blob
            nout (int): Number of outputs
            ks (int): Kernel size for the conv layer
            pad (int): Padding for the conv layer
            stride (int, optional): Stride for the conv layer
            train (bool, optional): Whether to train this block

        Returns:
            The name of the last bottom blob
        '''
        s = self.ConvBlock(bname, bottom, nout, ks, pad, stride, train=train)
        n_rel = 'res{}_relu'.format(bname)
        self.ReLU(s, n_rel)
        return n_rel

    def ConvBlock(self, bname, bottom, nout, ks, pad, stride=1, train=False):
        '''Adds a Block with a conv layer an BatchNorm layer and a scale
        layer.

        Args:
            bname (str): Name of the branch (e.g. 4b4_branch2a)
            bottom (str): Name of the bottom blob
            nout (int): Number of outputs
            ks (int): Kernel size for the conv layer
            pad (int): Padding for the conv layer
            stride (int, optional): Stride for the conv layer
            train (bool, optional): Whether to train this block

        Returns:
            The name of the last bottom blob
        '''
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
        self.BatchNorm('res' + bname, name='bn' + bname)
        self.Scale('bn' + bname, name='scale' + bname)
        return 'scale' + bname

    def RightBranch(self, bottom, resname, startout, ds=False, train=False):
        '''Adds the right branch of a residual block. Consisting of three
        conv layers, the first two with relus the last one without.

        Args:
            bottom (str): The name of the bottom blob
            resname (str): The name of this residual block (e.g. '4b3'/'4b')
            startout (int): The number of outputs for the first two conv layers
            ds (bool, optional): Whether this block will downscale
            train (bool, optional): Whether to train this block

        Returns:
            The name of the last bottom blob
        '''
        s = 2 if ds else 1
        ra = self.ConvReLUBlock(resname + '_branch2a', bottom, startout, ks=1,
                                pad=0, stride=s, train=train)
        rb = self.ConvReLUBlock(resname + '_branch2b', ra, startout, ks=3,
                                pad=1, train=train)
        sc = self.ConvBlock(resname + '_branch2c', rb, startout * 4, ks=1,
                            pad=0, train=train)
        return sc

    def ResHeadPartBlock(self, bottom, name, nout, ds=False):
        '''Adds the first block (left + right branch) of a Residual Block

        Args:
            bottom (str): The name of the bottom block
            name (str): The name of this block (e.g. 4a)
            nout (int): The number of outputs for the first conv layers of the
                right branch
            ds (bool, optional): Whether this block will downscale

        Returns:
            The name of the last bottom blob
        '''
        s = 2 if ds else 1
        left = self.ConvBlock(name + '_branch1', bottom, nout * 4, ks=1, pad=0,
                              stride=s)
        right = self.RightBranch(bottom, name, nout, ds)
        n_elt = 'res{}'.format(name)
        self.n[n_elt] = L.Eltwise(self.n[left], self.n[right], name=n_elt)
        self.ReLU(n_elt, name='res{}_relu'.format(name))
        return 'res{}_relu'.format(name)

    def ResPartBlock(self, bottom, name, nout, train=False):
        '''Adds a block (left + right branch) of a Residual Block

        Args:
            bottom (str): The name of the bottom block
            name (str): The name of this block (e.g. 4a)
            nout (int): The number of outputs for the first conv layers of the
                right branch
            train (bool, optional): Whether this block will be trained

        Returns:
            The name of the last bottom blob
        '''
        right = self.RightBranch(bottom, name, nout, ds=False, train=train)
        n_elt = 'res{}'.format(name)
        self.n[n_elt] = L.Eltwise(self.n[bottom], self.n[right], name=n_elt)
        self.ReLU(n_elt, name='res{}_relu'.format(name))
        return 'res{}_relu'.format(name)

    def ResBlock(self, bottom, nblocks, name, nout, ds=True, train=False,
                 alephnaming=True):
        '''Adds a block of same sized residuals

        Args:
            bottom (str): The name of the bottom block
            nblocks (int): The number of blocks in this block
            name (str): The name of this block (e.g. '2')
            nout (int): The number of outputs for the first conv layers of the
                right branch
            ds (bool, optional): Whether this block will downscale
            train (bool, optional): Whether this block will be trained
            alephnaming (bool, optional): Whether to use the alphabet naming
                as in ResNet50, instead of the naming in ResNet101..

        Returns:
            The name of the last bottom blob
        '''
        last = self.ResHeadPartBlock(bottom, name + 'a', nout, ds)
        for i in range(1, nblocks):
            if alephnaming:
                bname = name + str(chr(97 + i))
            else:
                bname = name + 'b' + str(i)
            if i == nblocks:
                last = self.ResPartBlock(last, bname, nout, train=train)
            else:
                last = self.ResPartBlock(last, bname, nout)
        return last

    def data(self):
        '''Adds the data layer'''
        if self.params['split'] == 'train' or self.params['split'] == 'val':
            self.n.data, self.n.label = L.Data(
                batch_size=18, source=self.params['lmdb'], backend=P.Data.LMDB,
                ntop=2, transform_param=dict(
                    crop_size=224,
                    mirror=True,
                    mean_file="data/models/resnet/ResNet_mean.binaryproto"
                    ))
        else:
            self.n.data = L.Input(shape=[dict(dim=[1, 3, 500, 500])])

    def base_net(self):
        '''Adds the main part of residual blocks. Everything between the data
        layer and the layers defined in self.tail()'''
        self.n.conv1 = L.Convolution(
            self.n.data, num_output=64, kernel_size=7, stride=2, pad=3)
        self.BatchNorm('conv1', name='bn_conv1')
        self.Scale('bn_conv1', name='scale_conv1')
        self.ReLU('scale_conv1', name='conv1_relu')
        self.n.pool1 = L.Pooling(
            self.n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)
        # Change naming convention if we build a Net larger ResNet50
        # e.g. ResNet101
        an = self.nconv4 <= 6
        res2c = self.ResBlock('pool1', 3, '2', 64, ds=False)
        res3d = self.ResBlock(res2c, self.nconv3, '3', 128, alephnaming=an)
        res4f = self.ResBlock(res3d, self.nconv4, '4', 256, alephnaming=an)
        self.ResBlock(res4f, 3, '5', 512, train=False)

    def tail(self):
        '''Adds the last layers after the residual blocks. 1 Pooling, 1 FC
        and Accuracy / SoftMax depending on current stage'''
        self.n.pool5 = L.Pooling(
            self.n.res5c, kernel_size=7, stride=1, pool=P.Pooling.AVE)
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
        '''Returns the Prototxt for this network'''
        self.__init__()
        self.params.update(params)
        self.switches.update(switches)
        self.data()
        self.base_net()
        self.tail()
        return self.n.to_proto()


class ResNet_FCN(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data(self):
        '''Adds the data layer'''
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
        '''Adds the last layers after the residual blocks. 1 Conv and
        SoftMax / SoftmaxWithLoss depending on current stage'''
        if self.switches['tofcn']:
            na = '_conv'
        elif self.switches['learn_fc']:
            na = '_'
        else:
            na = ''
        self.n['fc' + na] = L.Convolution(
            self.n.res5c, kernel_size=1, stride=1, num_output=self.nclasses,
            pad=0, param=[dict(lr_mult=1, decay_mult=1),
                          dict(lr_mult=2, decay_mult=0)])

        # self.n['upscore' + na] = upsample(self.n['fc' + na], factor=32,
        #                                   nout=self.nclasses)

        # self.n.score = crop(self.n['upscore' + na], self.n.data)

        if self.params['split'] == 'deploy':
            self.n.prob = L.Softmax(self.n['fc' + na])
        else:
            self.n.loss = L.SoftmaxWithLoss(self.n['fc' + na], self.n.label,
                                            loss_param=dict(normalize=False,
                                                            ignore_label=255))


class ResNet_Single(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data(self):
        '''Adds the data layer'''
        pylayer = 'SingleImageLayer'
        if self.params['split'] == 'train':
            self.n.data, self.n.label = L.Python(
                module='ba.caffeine.datalayers',
                layer=pylayer,
                ntop=2,
                param_str=str(self.params)
                )
        elif self.params['split'] == 'val':
            bs = self.params.get('batch_size', 18)
            self.n.data, self.n.label = L.Data(
                batch_size=bs,
                source=self.params['lmdb'],
                backend=P.Data.LMDB,
                ntop=2, transform_param=dict(
                    crop_size=224,
                    mirror=True,
                    mean_file="data/models/resnet/ResNet_mean.binaryproto"
                    ))
        else:
            self.n.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])
