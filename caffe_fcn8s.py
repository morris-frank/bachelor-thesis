import caffe
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def loadList(lpath):
    with open(lpath) as f:
        list = f.read().splitlines()
    return list


def saveList(lpath, flist):
    with open(lpath, 'w') as f:
        for row in flist:
            f.write("{}\n".format(row))
    print('List {} written...'.format(lpath))


def addPreSuffix(flist, prefix, suffix):
    return [prefix + x + suffix for x in flist]


def rmPreSuffix(flist, prefix, suffix):
    return [x[len(prefix):-len(suffix)] for x in flist]


def loadPascalPartMat(mpath):
    mat = sio.loadmat(mpath)
    segarray = {'name': '', 'seg': 0, 'parts': {}}
    try:
        mat = mat['anno'][0][0][1][0][0]
    except IndexError:
        print('loadPascalPartMat: given file is not a pp mat, %s', mpath)
        return False
    segarray['class'] = mat[0][0]
    segarray['seg'] = mat[2]
    if  mat[3].size and mat[3][0].size:
        for part in mat[3][0]:
            segarray['parts'][part[0][0]] = part[1]
    return segarray


def genPartList(flist, classname, parts):
    print('genPartList for {} in {}'.format(parts, classname))
    plist = []
    for i in tqdm(range(len(flist) - 1)):
        segarray = loadPascalPartMat(flist[i])
        if classname and segarray['class'] != classname:
            continue
        if not any(x in segarray['parts'] for x in parts):
            continue
        plist.append(flist[i])
    return plist


def genPlaneWingList():
    segtxt = 'data/datasets/pascalparts/set.txt'
    flist = loadList(segtxt)
    flist = addPreSuffix(flist, 'data/datasets/pascalparts/Annotations_Part/', '.mat')
    lwlist = genPartList(flist, 'aeroplane', ['lwing', 'rwing'])
    saveList('data/datasets/pascalparts/set_wings.txt', lwlist)
    return lwlist
    

class FCNPartRunner(object):
    """docstring for FCNPartRunner."""
    def __init__(self):
        super(FCNPartRunner, self).__init__()

    def createNet(model, weights, gpu):
        self.model = model
        self.weights = weights
        self.gpu = gpu
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model, weights, caffe.TEST)

    def addListFile(fpath):
        self.list = loadList(fpath)


def main():
    model   = 'data/models/fcn8s/deploy.prototxt'
    weights = 'data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'
    seglist = 'data/datasets/voc2010/ImageSets/Segmentation/val.txt'
    gpu = 0

    runner = FCNPartRunner()
    runner.createNet(model, weights, gpu)
    runner.addListFile(seglist)


if __name__ == '__main__':
    main()
