from scipy.misc import imsave
from src.FCNPartRunner import FCNPartRunner
from src.PascalPart import PascalPart
from src.SetList import SetList
from tqdm import tqdm
import caffe
import numpy as np
import os.path
import scipy.io as sio
import warnings

def reduceSavePascalPart(path):
    parts = ['lwing', 'rwing']
    pp = PascalPart(path)
    pp.reduce(parts)
    pp.source = pp.source[:-16] + '_Exports/' + pp.source[-15:]
    pp.save(image=True, parts=True, segmentation=True)


def genPlaneWingList():
    flist = SetList('data/datasets/pascalparts/set.txt')
    flist.addPreSuffix('data/datasets/pascalparts/Annotations_Part/', '.mat')
    lwlist = flist.genPartList('aeroplane', ['lwing', 'rwing'])
    lwlist.rmPreSuffix('data/datasets/pascalparts/Annotations_Part/', '.mat')
    lwlist.save()
    return lwlist


def savePlaneWingSegmentation():
    flist = SetList('data/datasets/pascalparts/set_wings.txt')
    flist.addPreSuffix('data/datasets/pascalparts/Annotations_Part/', '.mat')
    flist.each(reduceSavePascalPart)


def runcaffe():
    model   = 'data/models/fcn8s/deploy.prototxt'
    weights = 'data/models/fcn8s/fcn8s-heavy-pascal.caffemodel'
    seglist = 'data/datasets/voc2010/ImageSets/Segmentation/val.txt'
    gpu = 2
    runner = FCNPartRunner()
    runner.createNet(model, weights, gpu)
    runner.addListFile(seglist)
    return runner


def main():
    pass


if __name__ == '__main__':
    main()
