import caffe
from .SetList import SetList
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.misc import imsave

class FCNPartRunner(object):
    """docstring for FCNPartRunner."""
    def __init__(self):
        super(FCNPartRunner, self).__init__()

    def createNet(self, model, weights, gpu):
        self.model = model
        self.weights = weights
        self.gpu = gpu
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model, weights, caffe.TEST)

    def addListFile(self, fpath):
        self.list = SetList(fpath)

    def loadimg(self, path):
        im = Image.open(path)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))
        return in_

    def forward(self, in_):
        self.net.blobs['data'].reshape(1, *in_.shape)
        self.net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        self.net.forward()
        return self.net.blobs['score'].data[0].argmax(axis=0)

    def forwardList(self, postfix=''):
        for i in tqdm(self.list.list):
            o = self.forward(self.loadimg(i))
            imsave(i + postfix + '.out.png', o)
