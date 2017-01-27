from .set import SetList
from src.utils import query_overwrite, touch
from glob import glob
import numpy as np
import os.path
import scipy.io as sio
from scipy.misc import imsave
import tempfile
from tqdm import tqdm


def reduceSaveCallback(imgid, params):
    item = PascalPart(params['dir'] + imgid)
    item.reduce(params['parts'])
    doparts = params['parts']
    if doparts:
        item.source = params['parts_target'] + imgid
        if item.parts != {}:
            item.save(image=True, parts=True, sum=True, segmentation=False)
    if params['class']:
        item.source = params['class_target'] + imgid
        item.save(image=True, parts=False, sum=False, segmentation=True)


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
        self.genTargets()
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

    def genTargets(self):
        txtroot = self.builddir + self.tag
        segroot = self.builddir + 'segmentations/' + self.tag
        self.targets['root'] = txtroot + '.txt'
        self.targets['parts'] = self.targets['root']
        self.targets['classes'] = self.targets['root']
        if self.parts is not None:
            self.targets['parts'] = txtroot + '_'.join([''] + self.classes + self.parts) + '.txt'
            self.targets['parts_seg'] = segroot + '_'.join([''] + self.classes + self.parts) + '/'
        if self.classes is not None:
            self.targets['classes'] = txtroot + '_'.join([''] + self.classes) + '.txt'
            self.targets['classes_seg'] = segroot + '_'.join([''] + self.classes) + '/'

    def genRootList(self):
        overwrite = query_overwrite(self.targets['root'])
        self.rlist = SetList(self.targets['root'])
        if not overwrite:
            return self.rlist
        # Get most prominent extension:
        exts = [os.path.splitext(x)[1][1:] for x in glob(self.dir + '*')]
        exts = [x for x in exts if x]
        self.ext = max(set(exts), key=exts.count)
        files = glob(self.dir + '*' + self.ext)
        # Remove path and extension:
        files = [row[len(self.dir):-len(self.sourceext)] for row in files]
        self.rlist.list = files
        self.rlist.save()
        return self.rlist

    def genClassList(self):
        if self.classes == []:
            return
        if not self.rlist:
            self.genRootList()
        overwrite = query_overwrite(self.targets['classes'])
        touch(self.targets['classes'], clear=overwrite)
        self.clist = SetList(self.targets['classes'])
        if not overwrite:
            return self.clist
        self.rlist.addPreSuffix(self.dir, self.sourceext)
        print('Generating ClassList {}...'.format(self.targets['classes']))
        for row in tqdm(self.rlist.list):
            item = PascalPart(row)
            if item.classname in self.classes:
                self.clist.list.append(row)
        self.rlist.rmPreSuffix(self.dir, self.sourceext)
        self.clist.rmPreSuffix(self.dir, self.sourceext)
        self.clist.save()
        return self.clist

    def genPartList(self):
        if self.parts == []:
            return
        if not self.rlist:
            self.genRootList()
        if not self.clist:
            self.genClassList()
        if self.classes == []:
            rootlist = self.rlist
        else:
            rootlist = self.clist
        overwrite = query_overwrite(self.targets['parts'])
        touch(self.targets['parts'], clear=overwrite)
        self.plist = SetList(self.targets['parts'])
        if not overwrite:
            return self.plist
        rootlist.addPreSuffix(self.dir, self.sourceext)
        print('Generating PartList {}...'.format(self.targets['parts']))
        for row in tqdm(rootlist.list):
            item = PascalPart(row)
            if any(part in item.parts for part in self.parts):
                self.plist.list.append(row)
        rootlist.rmPreSuffix(self.dir, self.sourceext)
        self.plist.rmPreSuffix(self.dir, self.sourceext)
        self.plist.save()
        return self.plist

    def saveSegmentations(self):
        doClasses = len(self.classes) > 0
        touch(self.targets['parts_seg'])
        if doClasses:
            touch(self.targets['classes_seg'])
            rootlist = self.clist
        else:
            rootlist = self.plist
        params = {}
        params['dir'] = self.dir
        params['parts'] = self.parts
        params['parts_target'] = self.targets['parts_seg']
        params['class_target'] = self.targets['classes_seg']
        params['class'] = doClasses
        print('Generating and extracting the segmentations...')
        rootlist.each(lambda imgid, params=params: reduceSaveCallback(imgid, params))


class PascalPart(object):
    """docstring for PascalPart."""
    parts = {}
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
                sumOfParts = next(iter(self.parts.values())) * 0
                # sumOfParts = self.segmentation * 0
                for part in self.parts:
                    sumOfParts += self.parts[part]
                itemsave(bn + ext, sumOfParts)
            else:
                for part in self.parts:
                    itemsave(bn + ext, self.parts[part])

    def get(self, part):
        if part in self.parts:
            return self.parts[part]
        # else:
        #     return self.segmentation * 0

    def reduce(self, parts=None):
        if parts is None or parts == []:
            return
        newparts = {}
        for part in parts:
            newparts[part] = self.get(part)
        self.parts = newparts
