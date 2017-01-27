from .set import SetList
from glob import glob
import numpy as np
import os.path
import scipy.io as sio
from scipy.misc import imsave
import tempfile
from tqdm import tqdm


class PascalPartSet(object):
    """docstring for PascalPartSet."""
    builddir = 'data/models/tmp/'
    targets = {}
    rlist = None
    clist = None
    plist = None
    sourceext = '.mat'

    def __init__(self, tag_, dir_='.', parts_=[], classes_=[]):
        self.dir = dir_
        self.tag = tag_
        self.setParts(parts_)
        self.setClasses(classes_)
        self.targets()
        self.genRootList()
        self.genClassList()
        self.genPartList()

    def setParts(self, parts_):
        if parts_ is not None and not isinstance(parts_, list):
            self.parts = [parts_]
        else:
            self.parts = parts_

    def setClasses(self, classes_):
        if classes_ is not None and not isinstance(classes_, list):
            self.classes = [classes_]
        else:
            self.classes = classes_

    def targets(self):
        txtroot = self.builddir + self.tag
        segroot = self.dir + self.tag
        self.targets['root'] = txtroot + '.txt'
        self.targets['parts'] = self.targets['root']
        self.targets['classes'] = self.targets['root']
        if self.parts is not None:
            self.targets['parts'] = txtroot + '_'.join(self.parts) + '.txt
            self.targets['parts_seg'] = segroot + '_'.join(self.parts) + '/'
        if self.classes is not None:
            self.targets['classes'] = txtroot + '_'.join(self.classes) + '.txt
            self.targets['classes_seg'] = segroot + '_'.join(self.parts) + '/'

    def genRootList(self):
        # Get most prominent extension:
        exts = [os.path.splitext(x)[1][1:] for x in glob(self.dir + '*')]
        exts = [x for x in exts if x]
        self.ext = max(set(exts), key=exts.count)
        files = glob(self.dir + '*' + self.ext)
        # Remove path and extension:
        files = [row[len(self.dir):-len(self.sourceext)] for row in files]
        self.rlist = SetList(self.targets['root'])
        self.rlist.list = files
        self.rlist.save()
        return self.rlist

    def genClassList(self):
        if self.classes == []:
            return
        if not self.rlist:
            self.genRootList()
        self.clist = SetList(self.targets['classes'])
        self.clist.addPreSuffix(self.dir, self.sourceext)
        for row in tqdm(self.rlist):
            item = PascalPart(row)
            if item.classname in self.classes:
                self.clist.content.append(row)
        self.clist.rmPreSuffix(self.dir, self.sourceext)
        self.clist.save()
        return self.clist

    def genPartList(self):
        if self.parts == []:
            return
        if not self.rlist:
            self.genRootList()
        self.plist = SetList(self.targets['parts'])
        self.plist.addPreSuffix(self.dir, self.sourceext)
        for row in tqdm(self.rlist):
            item = PascalPart(row)
            if any(part in item.parts for part in self.parts):
                self.plist.content.append(row)
        self.plist.rmPreSuffix(self.dir, self.sourceext)
        self.plist.save()
        return self.plist

    def saveSegmentations(self):
        doClasses = len(self.classes) > 0
        doParts = len(self.parts) > 0
        if doClasses:
            os.makedirs(self.target['classes_seg'], exist_ok=True)
            self.clist.each(self.reduceSaveClassCallback)
        if doParts:
            os.makedirs(self.target['parts_seg'], exist_ok=True)
            self.plist.each(self.reduceSavePartCallback)

    def reduceSavePartCallback(imgid, parts_):
        item = PascalPart(self.dir + imgid + self.sourceext)
        item.reduce(parts_)
        item.source = self.target['parts_seg'] + imgid
        item.save(image=False, parts=True, sum=True)

    def reduceSaveClassCallback(imgid, parts_):
        item = PascalPart(self.dir + imgid + self.sourceext)
        item.reduce(parts_)
        item.source = self.target['classes_seg'] + imgid
        item.save(image=True, parts=False, sum=False)


class PascalPart(object):
    """docstring for PascalPart."""
    def __init__(self, source=''):
        self.source = source
        if source != '':
            self.load()

    def load(self):
        mat = sio.loadmat(self.source)
        try:
            mat = mat['anno'][0][0][1][0][0]
        except IndexError:
            print('PascalPart::load: given file is wrong, %s', self.source)
            return False
        self.classname = mat[0][0]
        self.segmentation = mat[2].astype('float')
        self.parts = {}
        if mat[3].size and mat[3][0].size:
            for part in mat[3][0]:
                self.parts[part[0][0]] = part[1].astype('float')

    def save(self, image=True, parts=True, sum=False, segmentation=False):
        bn = os.path.splitext(self.source)[0]
        itemsave = imsave if image else np.save
        ext = '.png' if image else ''
        if segmentation:
            itemsave(bn + ext, self.segmentation)
        if parts:
            if sum:
                sumOfParts = self.segmentation * 0
                for part in self.parts:
                    sumOfParts += self.parts[part]
                itemsave(bn + ext, sumOfParts)
            else:
                for part in self.parts:
                    itemsave(bn + ext, self.parts[part])

    def get(self, part):
        if part in self.parts:
            return self.parts[part]
        else:
            return self.segmentation * 0

    def reduce(self, parts):
        newparts = {}
        for part in parts:
            newparts[part] = self.get(part)
        self.parts = newparts
