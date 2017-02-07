from ba.set import SetList
from ba import utils
from glob import glob
import numpy as np
import os.path
import scipy.io as sio
import scipy.ndimage
from scipy.misc import imsave
from scipy.misc import imread
import tempfile
from tqdm import tqdm
from functools import partial


def getSingularBB(img):
    # TODO(doc): Add docstring
    # TODO: Transform to method of class
    slices = scipy.ndimage.find_objects(img)
    # TODO: Yeah so, that cant be rught: :P
    slice_ = slices[-1]
    return img[slice_], slice_


def reduceSaveCallback(imgid, params):
    # TODO(doc): Add docstring
    # TODO: Transform to method of class
    item = PascalPart(params['dir'] + imgid + '.mat')
    item.reduce(params['parts'])
    if len(item.parts) > 0:
        item.source = params['parts_target'] + imgid
        item.save(sum=True)
    if params['class']:
        item.source = params['class_target'] + imgid
        item.save(parts=False, segmentation=True)


def reduceBBSaveCallback(imgid, params):
    # TODO(doc): Add docstring
    # TODO: Transform to method of class
    item = PascalPart(params['dir'] + imgid + '.mat')
    item.reduce(params['parts'])
    im = imread(params['imdir'] + imgid + '.jpg')
    if len(item.parts) > 0:
        item.source = params['parts_bb_target'] + imgid
        slice_ = item.saveBB(sum=True)
        imsave(params['parts_patch_target'] + imgid + '.png', im[slice_])
    if params['class']:
        item.source = params['classes_bb_target'] + imgid
        slice_ = item.saveBB(parts=False, segmentation=True)
        imsave(params['classes_patch_target'] + imgid + '.png', im[slice_])


class PascalPartSet(object):
    # TODO(doc): Add docstring
    builddir = 'data/tmp/'
    sourceext = '.mat'

    def __init__(self, tag_, dir_='.', parts_=[], classes_=[]):
        # TODO(doc): Add docstring
        self.dir = dir_
        self.tag = tag_
        self.targets = {}
        self.rlist = None
        self.clist = None
        self.plist = None
        self.setParts(parts_)
        self.setClasses(classes_)
        self.genTargets()
        self.genRootList()
        self.genClassList()
        self.genPartList()

    def setParts(self, parts_):
        # TODO(doc): Add docstring
        if parts_ is not None and not isinstance(parts_, list):
            self.parts = [parts_]
        else:
            self.parts = parts_

    def setClasses(self, classes_):
        # TODO(doc): Add docstring
        if classes_ is not None and not isinstance(classes_, list):
            self.classes = [classes_]
        else:
            self.classes = classes_

    def genTargets(self):
        # TODO(doc): Add docstring
        txtroot = self.builddir + self.tag
        segroot = self.builddir + 'segmentations/' + self.tag
        classstr = '_'.join([''] + self.classes)
        classnpartstr = '_'.join([''] + self.classes + self.parts)
        self.targets['root'] = txtroot + '.txt'
        self.targets['parts'] = self.targets['root']
        self.targets['classes'] = self.targets['root']
        if self.parts is not None:
            self.targets['parts'] = txtroot + classnpartstr + '.txt'
            self.targets['parts_seg'] = segroot + classnpartstr + '/'
        if self.classes is not None:
            self.targets['classes'] = txtroot + classstr + '.txt'
            self.targets['classes_seg'] = segroot + classstr + '/'

    def genRootList(self):
        # TODO(doc): Add docstring
        overwrite = utils.query_overwrite(self.targets['root'], default='no')
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
        # TODO(doc): Add docstring
        if self.classes == []:
            return False
        if not self.rlist:
            self.genRootList()
        overwrite = utils.query_overwrite(self.targets['classes'],
                                             default='no')
        utils.touch(self.targets['classes'], clear=overwrite)
        self.clist = SetList(self.targets['classes'])
        if not overwrite:
            return self.clist
        self.rlist.addPreSuffix(self.dir, self.sourceext)
        print('Generating ClassList {}...'.format(self.targets['classes']))
        for row in tqdm(self.rlist):
            item = PascalPart(row)
            if item.classname in self.classes:
                self.clist.list.append(row)
        self.rlist.rmPreSuffix(self.dir, self.sourceext)
        self.clist.rmPreSuffix(self.dir, self.sourceext)
        self.clist.save()
        return self.clist

    def genPartList(self):
        # TODO(doc): Add docstring
        if self.parts == []:
            return False
        if not self.rlist:
            self.genRootList()
        if not self.clist:
            self.genClassList()
        if self.classes == []:
            rootlist = self.rlist
        else:
            rootlist = self.clist
        overwrite = utils.query_overwrite(self.targets['parts'],
                                             default='no')
        utils.touch(self.targets['parts'], clear=overwrite)
        self.plist = SetList(self.targets['parts'])
        if not overwrite:
            return self.plist
        rootlist.addPreSuffix(self.dir, self.sourceext)
        print('Generating PartList {}...'.format(self.targets['parts']))
        for row in tqdm(rootlist):
            item = PascalPart(row)
            if any(part in item.parts for part in self.parts):
                self.plist.list.append(row)
        rootlist.rmPreSuffix(self.dir, self.sourceext)
        self.plist.rmPreSuffix(self.dir, self.sourceext)
        self.plist.save()
        return self.plist

    def saveSegmentations(self):
        # TODO(doc): Add docstring
        doClasses = len(self.classes) > 0
        params_ = {}
        if not utils.query_overwrite(self.targets['parts_seg'],
                                        default='no'):
            params_['parts'] = []
        else:
            utils.touch(self.targets['parts_seg'])
            params_['parts'] = self.parts
        if doClasses:
            doClasses = utils.query_overwrite(self.targets['classes_seg'],
                                                 default='no')
            utils.touch(self.targets['classes_seg'])
            rootlist = self.clist
        else:
            rootlist = self.plist
        params_['dir'] = self.dir
        params_['parts_target'] = self.targets['parts_seg']
        params_['class_target'] = self.targets['classes_seg']
        params_['class'] = doClasses
        if params_['parts'] != [] or doClasses:
            print('Generating and extracting the segmentations...')
            rootlist.each(partial(reduceSaveCallback, params=params_))
        else:
            print('Will not save any Segmentations...')

    def saveBoundingBoxes(self, imgdir):
        # TODO(doc): Add docstring
        params_ = {'dir': self.dir,
                   'imdir': imgdir,
                   'parts_bb_target': None,
                   'classes_bb_target': None,
                   'parts_patch_target': None,
                   'classes_patch_target': None,
                   'class': len(self.classes) > 0
                   }
        params_['parts_bb_target'] = self.targets['parts_seg'][:-1] + '_bb/'
        params_['classes_bb_target'] = self.targets['classes_seg'][:-1] + '_bb/'
        params_['parts_patch_target'] = self.targets['parts_seg'][:-1] + '_bb_patches/'
        params_['classes_patch_target'] = self.targets['classes_seg'][:-1] + '_bb_patches/'
        if not utils.query_overwrite(params_['parts_bb_target'], default='no'):
            params_['parts'] = []
        else:
            utils.touch(params_['parts_bb_target'])
            utils.touch(params_['parts_patch_target'])
            params_['parts'] = self.parts
        if params_['class']:
            params_['class'] = utils.query_overwrite(params_['classes_bb_target'],
                                                        default='no')
            utils.touch(params_['classes_bb_target'])
            utils.touch(params_['classes_patch_target'])
            rootlist = self.clist
        else:
            rootlist = self.plist
        print('Generating and extracting the segmentation bounding boxes...')
        rootlist.each(partial(reduceBBSaveCallback, params=params_))


class PascalPart(object):
    # TODO(doc): Add docstring

    def __init__(self, source=''):
        # TODO(doc): Add docstring
        self.parts = {}
        self.source = source
        if source != '':
            self.load()

    def load(self):
        # TODO(doc): Add docstring
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
        # TODO(doc): Add docstring
        bn = os.path.splitext(self.source)[0]
        itemsave = imsave if image else np.save
        ext = '.png' if image else ''
        if segmentation:
            itemsave(bn + ext, self.segmentation)
        if parts and len(self.parts) > 0:
            if sum:
                sumOfParts = next(iter(self.parts.values())) * 0
                # sumOfParts = self.segmentation * 0
                for part in self.parts:
                    sumOfParts += self.parts[part]
                itemsave(bn + ext, sumOfParts)
            else:
                for part in self.parts:
                    # TODO(saveEach): What??? saving all on hte same imagE??
                    itemsave(bn + ext, self.parts[part])

    def saveBB(self, image=True, parts=True, sum=False, segmentation=False):
        # TODO(doc): Add docstring
        bn = os.path.splitext(self.source)[0]
        itemsave = imsave if image else np.save
        ext = '.png' if image else ''
        if segmentation:
            bb, slice_ = getSingularBB(self.segmentation.astype(int))
            itemsave(bn + ext, bb)
            return slice_
        if parts and len(self.parts) > 0:
            if sum:
                sumOfParts = next(iter(self.parts.values())) * 0
                # sumOfParts = self.segmentation * 0
                for part in self.parts:
                    sumOfParts += self.parts[part]
                bb, slice_ = getSingularBB(sumOfParts.astype(int))
                itemsave(bn + ext, bb)
                return slice_

    def reduce(self, parts=[]):
        # TODO(doc): Add docstring
        newparts = {}
        if len(parts) > 0 and len(self.parts) > 0:
            for part in parts:
                if part in self.parts:
                    newparts[part] = self.parts[part]
        self.parts = newparts
