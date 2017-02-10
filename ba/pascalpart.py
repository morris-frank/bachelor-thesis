from ba.set import SetList
from ba import utils
import copy
from functools import partial
from glob import glob
import numpy as np
import os.path
import scipy.io as sio
from scipy.misc import imread
from scipy.misc import imsave
import scipy.ndimage
import tempfile
from tqdm import tqdm


def getSingularBB(img):
    # TODO(doc): Add docstring
    # TODO: Transform to method of class
    slices = scipy.ndimage.find_objects(img)
    # TODO: Yeah so, that cant be rught: :P
    slice_ = slices[-1]
    return img[slice_], slice_


def sliceOverlap(x1, x2, w):
    x1_2 = [x1[0] + w[0], x1[1] + w[1]]
    x2_2 = [x2[0] + w[0], x2[1] + w[1]]
    SI = max(0, min(x1_2[0], x2_2[0]) - max(x1[0], x2[0])) * max(0, min(x1_2[1], x2_2[1]) - max(x1[1], x2[1]))
    S = 2 * w[0] * w[1] - SI
    return SI / S

class PascalPartSet(object):
    # TODO(doc): Add docstring
    _builddir = 'data/tmp/'

    def __init__(self, name, root='.', parts=[], classes=[]):
        # TODO(doc): Add docstring
        self.name = name
        self.source = root
        self.parts = parts
        self.classes = classes
        self.genLists()

    @property
    def parts(self):
        return self.__parts

    @parts.setter
    def parts(self, parts):
        if isinstance(parts, list):
            self.__parts = parts
        else:
            self.__parts = [parts]
        self.tag = '_'.join(self.classes + self.parts)

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, classes):
        if isinstance(classes, list):
            self.__classes = classes
        else:
            self.__classes = [classes]
        self.tag = '_'.join(self.classes + self.parts)

    @property
    def source(self):
        return self.__source

    @source.setter
    def source(self, source):
        if not os.path.isdir(source):
            raise OSError('source attribute must be path to a dir')
        else:
            self.__source = source
            self.extension = '.' + utils.prevalentExtension(source)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.name = str(name)
        self.build = self._builddir + '/' + self.name + '/'
        utils.touch(self.build)

    def write(self):
        for l in [self.sourcelist, self.classlist, self.partslist]:
            l.rmPreSuffix(self.source, self.extension)
            l.write()
            l.addPreSuffix(self.source, self.extension)

    def genLists(self):
        '''Generates the *.txt lists for this set.'''
        f = {
            source = self.build + self.name + '.txt',
            classes = self.build + '_'.join(self.classes) + '.txt',
            parts = self.build + self.tag + '.txt'
            }
        overwrite = {
            source = utils.query_overwrite(f['source'], default='no'),
            classes = utils.query_overwrite(f['classes'], default='no'),
            parts = utils.query_overwrite(f['parts'], default='no')
            }
        if not sum(f.values()):
            return True

        self.sourcelist = SetList(f['source'])
        if overwrite['source']:
            self.sourcelist.loadDir(self.source)
        self.sourcelist.addPreSuffix(self.source, self.extension)

        self.classlist = SetList(f['classes'])
        self.partslist = SetList(f['parts'])

        if overwrite['classes'] or overwrite['parts']:
            print('Generating List for {} and {}'.format(f['classes'],
                                                         f['parts']))
            for row in tqdm(self.sourcelist):
                item = PascalPart(row)
                if item.classname in self.classes or len(self.classes) < 1:
                    self.classlist.list.append(row)
                    if any(part in item.parts for part in self.parts):
                        self.partslist.list.append(row)
        self.write()

    def saveSegmentations(self):
        '''Saves the segmentations for selected classes or parts.'''
        d = {
            classes = '{}segmentations/{}/'.format(self.build, '_'.join(self.classes))
            parts = '{}segmentations/{}/'.format(self.build, self.tag)
            }
        overwrite = {
            classes = utils.query_overwrite(d['classes'], default='no')
            parts = utils.query_overwrite(d['parts'], default='no')
            }
        if not sum(f.values()):
            return True

        print('Generating and extracting the segmentations...')
        for item in tqdm(self.sourcelist):
            idx = os.path.splitext(os.path.basename(item))[0]
            item = PascalPart(item)
            if overwrite['parts']:
                item.reduce(self.parts)
                item.source = d['parts'] + idx
                item.save(sum=True)
            if overwrite['classes']:
                item.source = d['classes'] + idx
                item.save(parts=False, segmentation=True)

    def saveBoundingBoxes(self, imgdir, negatives=0):
        '''Saves the bounding box patches for classes and parts.

        Args:
            imgdir (str): The directory where the original images live
            negatives (int, optional): How many negative samples to generate
        '''
        cbdir = '{}patches/{}/'.format(self.build, '_'.join(self.classes))
        pbdir = '{}patches/{}/'.format(self.build, self.tag)
        d = {
            img_pos = utils.touch(pbdir + 'img/pos/'),
            img_neg = utils.touch(pbdir + 'img/neg/'),
            seg_pos = utils.touch(pbdir + 'seg/pos/'),
            img_cla = utils.touch(cbdir + 'img/'),
            seg_cla = utils.touch(cbdir + 'seg/')
            }
        if not utils.query_overwrite(pbdir, default='yes'):
            return True

        print('Generating and extracting the segmentation bounding boxes...')
        for item in self.sourcelist:
            idx = os.path.splitext(os.path.basename(item))[0]
            item = PascalPart(item)
            im = imread(imgdir + idx + '.' + utils.prevalentExtension(imgdir))
            item.reduce(self.parts)

            # Save Class patches
            item.source = d['seg_cla'] + idx
            bb = item.saveBB(parts=False, segmentation=True)
            imsave(d['img_cla'] + idx + '.png', im[bb])

            # Save positive patch
            item.source = d['seg_pos'] + idx
            bb = item.saveBB(sum=True)
            imsave(d['img_pos'] + idx + '.png', im[bb])

            # Save neagtive patches
            for i in range(0, negatives):
                x2 = x1 = [bb[0].start, bb[1].start]
                w = [bb[0].stop - x1[0], bb[1].stop - x1[1]]
                subim = [im.shape[0] - w[0], im.shape[1] - w[1]]
                checkidx = 0
                while checkidx < 30 and sliceOverlap(x1, x2, w) > 0.3:
                    checkidx += 1
                    x2 = (np.random.random(2) * subim).astype(int)
                if checkidx >= 30:
                    continue
                negpatch = im[x2[0]:x2[0] + w[0], x2[1]:x2[1] + w[1]]
                imsave(d['img_neg'] + '{}_{}.png'.format(idx, i), negpatch)


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
