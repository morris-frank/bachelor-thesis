import caffe
from ba import FCNPartRunner

def runexpCN(weights, postfix):
    netrn = FCNPartRunner()
    netrn.createNet('data/models/fcn8s/deploy.prototxt', weights, 1)
    netrn.addListFile('data/datasets/pascalparts/expmixset.txt')
    netrn.list.addPreSuffix('data/datasets/voc2010/JPEGImages/', '.jpg')
    netrn.forwardList(postfix)

w_4epochs = 'data/models/fcn8s/snapshot/4epochs/train_iter_1000.caffemodel'
w_1epoch = 'data/models/fcn8s/snapshot/1epoch/train_iter_448.caffemodel'
w_4x50 = 'data/models/fcn8s/snapshot/4x50/train_iter_200.caffemodel'
w_2x10 = 'data/models/fcn8s/snapshot/2x10/train_iter_20.caffemodel'
w_20x2 = 'data/models/fcn8s/snapshot/20x2/train_iter_40.caffemodel'
w_20x2_2 = 'data/models/fcn8s/snapshot/20x2_-2/train_iter_40.caffemodel'

#runexpCN(w_4epochs, '4epochs')
#runexpCN(w_1epoch, '1epoch')
#runexpCN(w_4x50, '4x50')
#runexpCN(w_2x10, '2x10')
#runexpCN(w_20x2, '20x2')
runexpCN(w_20x2_2, '20x2-2')
