import caffe
from src.SetList import SetList

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
