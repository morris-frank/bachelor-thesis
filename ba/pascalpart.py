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
from tqdm import tqdm

class PascalPartSet(object):
    _builddir = 'data/tmp/'
    _testtrain = 0.2

    def __init__(self, name, root='.', parts=[], classes=[], dolists=True):
        '''Constructs a new PascalPartSet

        Args:
            name (str): The name of this set. User for saving paths
            root (str, optional): The path for the dir with the mat files
            parts (list, optional): The parts we are interested in
            classes (classes, optional): The classes we are interested in
        '''
        self.name = name
        self.source = root
        self.classes = classes
        self.parts = parts
        if dolists:
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
        # yeah, so I think Im misusing this whole construct: ^^
        # self.tag = '_'.join(self.classes + self.parts)

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
        self.__name = str(name)
        self.build = self._builddir + '/' + self.__name + '/'
        utils.touch(self.build)

    def write(self):
        '''Writes the generated lists to disk'''
        for l in [self.sourcelist, self.classlist, self.partslist]:
            l.rmPreSuffix(self.source, self.extension)
            l.write()
            l.addPreSuffix(self.source, self.extension)

    def loadLists(self):
        '''Loads the *.txt lists for this set.'''
        f = {
            'source': self.build + self.name + '.txt',
            'classes': self.build + '_'.join(self.classes) + '.txt',
            'parts': self.build + self.tag + '.txt'
            }
        self.sourcelist = SetList(f['source'])
        self.classlist = SetList(f['classes'])
        self.partslist = SetList(f['parts'])
        self.sourcelist.addPreSuffix(self.source, self.extension)
        self.classlist.addPreSuffix(self.source, self.extension)
        self.partslist.addPreSuffix(self.source, self.extension)

    def genLists(self):
        '''Generates the *.txt lists for this set.'''
        f = {
            'source': self.build + self.name + '.txt',
            'classes': self.build + '_'.join(self.classes) + '.txt',
            'parts': self.build + self.tag + '.txt'
            }
        overwrite = {
            'source': utils.query_overwrite(f['source'], default='no'),
            'classes': utils.query_overwrite(f['classes'], default='no'),
            'parts': utils.query_overwrite(f['parts'], default='no')
            }

        self.loadLists()
        if not sum(overwrite.values()):
            return True

        if overwrite['source']:
            print('Generating List {}'.format(f['source']))
            self.sourcelist.loadDir(self.source)
            self.sourcelist.addPreSuffix(self.source, self.extension)

        if overwrite['classes'] or overwrite['parts']:
            self.classlist.list = []
            self.partslist.list = []
            print('Generating List {} and {}'.format(f['classes'], f['parts']))
            for row in tqdm(self.sourcelist):
                item = PascalPart(row)
                if item.classname in self.classes or len(self.classes) < 1:
                    self.classlist.list.append(row)
                    if any(part in item.parts for part in self.parts):
                        self.partslist.list.append(row)
        self.write()

    def saveSegmentations(self, augment=0):
        '''Saves the segmentations for selected classes or parts.

        Args:
            augment (int, optional): How many augmentations per image
        '''
        d = {
            'classes': '{}segmentations/{}/'.format(self.build, '_'.join(self.classes)),
            'parts': '{}segmentations/{}/'.format(self.build, self.tag)
            }
        overwrite = {
            'classes': utils.query_overwrite(d['classes'], default='no'),
            'parts': utils.query_overwrite(d['parts'], default='no')
            }
        if not sum(overwrite.values()):
            return True

        print('Generating and extracting the segmentations...')
        for item in tqdm(self.classlist):
            idx = os.path.splitext(os.path.basename(item))[0]
            item = PascalPart(item)
            if overwrite['parts']:
                item.reduce(self.parts)
                item.target = d['parts'] + idx
                item.save(mode='parts')
            if overwrite['classes']:
                item.target = d['classes'] + idx
                item.save(mode='class')

    def saveBoundingBoxes(self, imgdir, negatives=0, augment=0):
        '''Saves the bounding box patches for classes and parts.

        Args:
            imgdir (str): The directory where the original images live
            negatives (int, optional): How many negative samples to generate
            augment (int, optional): How many augmentations per image
        '''
        import yaml
        cbdir = '{}patches/{}/'.format(self.build, '_'.join(self.classes))
        pbdir = '{}patches/{}/'.format(self.build, self.tag)
        d = {
            'patch_pos': utils.touch(pbdir + 'img/pos/'),
            'patch_neg': utils.touch(pbdir + 'img/neg/'),
            'class_img': utils.touch(cbdir + 'img/'),
            'patch_seg': utils.touch(pbdir + 'seg/'),
            'class_seg': utils.touch(cbdir + 'seg/')
            }
        if utils.query_overwrite(pbdir, default='yes'):
            ext = utils.prevalentExtension(imgdir)

            classSegDB = {}
            classDBPath = d['class_seg'][:-1] + '.yaml'
            patchSegDB = {}
            patchDBPath = d['patch_seg'][:-1] + '.yaml'

            print('Generating and extracting the segmentation bounding boxes.')
            for item in tqdm(self.classlist):
                idx = os.path.splitext(os.path.basename(item))[0]
                item = PascalPart(item)
                im = imread(imgdir + idx + '.' + ext)
                item.reduce(self.parts)

                # Save Class patches
                item.target = d['class_seg'] + idx
                bb = item.saveBB(mode='class')
                classSegDB[idx] = bb
                imsave(d['class_img'] + idx + '.png', im[bb])

                # Save positive patch
                item.target = d['patch_seg'] + idx
                bb = item.saveBB(mode='parts')
                if bb is None:
                    continue
                patchSegDB[idx] = bb
                imsave(d['patch_pos'] + idx + '.png', im[bb])

                # Save neagtive patches
                for i in range(0, negatives):
                    x2 = x1 = [bb[0].start, bb[1].start]
                    w = [bb[0].stop - x1[0], bb[1].stop - x1[1]]
                    subim = [im.shape[0] - w[0], im.shape[1] - w[1]]
                    checkidx = 0
                    while checkidx < 30 and utils.sliceOverlap(x1, x2, w) > 0.3:
                        checkidx += 1
                        x2 = (np.random.random(2) * subim).astype(int)
                    if checkidx >= 30:
                        continue
                    negpatch = im[x2[0]:x2[0] + w[0], x2[1]:x2[1] + w[1]]
                    imsave(d['patch_neg'] + '{}_{}.png'.format(idx, i), negpatch)

            with open(classDBPath, 'w') as f:
                yaml.dump(classSegDB, f)
            with open(patchDBPath, 'w') as f:
                yaml.dump(patchSegDB, f)

        if utils.query_overwrite(pbdir + 'img_augmented/', default='yes'):
            self.augmentSingle(pbdir + 'img/pos/', len(self.classlist) * augment)
            self.augmentSingle(pbdir + 'img/neg/', len(self.classlist) * augment)
            # self.augmentDual(d['patch_pos'], d['patch_seg'], len(self.classlist)*augment)
            self.genLMDB(pbdir + 'img_augmented/')

    def genLMDB(self, path):
        '''Generates the LMDB for the trainingset.

        Args:
            path (str): The path to the image directory. (Contains dirs pos and
                       neg)
        '''
        print('Generating LMDB for {}'.format(path))
        absp = os.path.abspath(path)
        target = {'train': absp + '_lmdb_train', 'test': absp + '_lmdb_test'}
        for d in target.values():
            utils.rm(d)
            utils.rm(d + '.txt')
        wh = 224
        trainlist = SetList(absp + '/pos/')
        trainlist.addPreSuffix(absp + '/pos/', '.png 1')
        negList = SetList(absp + '/neg/')
        negList.addPreSuffix(absp + '/neg/', '.png 0')
        n = {'pos': int(len(trainlist) * 0.2), 'neg': int(len(negList) * 0.2)}
        testlist = copy.deepcopy(trainlist)
        testlist.list = testlist.list[:n['pos']] + negList.list[:n['neg']]
        trainlist.list = trainlist.list[n['pos']:] + negList.list[n['neg']:]
        trainlist.target = target['train'] + '.txt'
        testlist.target = target['test'] + '.txt'
        trainlist.write()
        testlist.write()
        os.system('convert_imageset --resize_height={} --resize_height={} --shuffle "/" "{}" "{}" '.format(wh, wh, trainlist.target, target['train']))
        os.system('convert_imageset --resize_height={} --resize_height={} --shuffle "/" "{}" "{}" '.format(wh, wh, testlist.target, target['test']))

    def augmentSingle(self, imdir, n):
        '''Generates augmentet images

        Args:
            imdir (str): The path to the images
            n (int): Number of images to produce
        '''
        import keras.preprocessing.image
        augmenter = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True
            )
        par_imdir = '/'.join(os.path.normpath(imdir).split('/')[:-1])
        bn_imdir = os.path.normpath(imdir).split('/')[-1]
        save_imdir = os.path.normpath(par_imdir) + '_augmented'
        utils.rm(save_imdir + '/' + bn_imdir)
        os.makedirs(save_imdir + '/' + bn_imdir)
        img_generator = augmenter.flow_from_directory(
            directory=par_imdir,
            target_size=(224, 224),
            class_mode='binary',
            classes=[bn_imdir],
            save_to_dir=save_imdir + '/' + bn_imdir,
            save_format='png',
            batch_size=50
            )
        for _ in range(0, int(n / 50)):
            img_generator.next()


class PascalPart(object):

    def __init__(self, source=''):
        '''Constructs a new PascalPart.

        Args:
            source (str, optional): The path to mat file
        '''
        self.parts = {}
        self.source = source
        self.target = source
        self.itemsave = lambda path, im: imsave(path + '.png', im)

    @property
    def source(self):
        return self.__source

    @source.setter
    def source(self, source):
        self.__source = source
        self.load()

    def load(self):
        '''Loads the inouts from the mat file'''
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
        self.genSumOfParts()

    def save(self, mode='parts'):
        '''Saves the segmentations binarly samesized to input

        Args:
            mode (str, optional): Either doing the whole object or only the
                                  parts
        '''
        if mode == 'class':
            self.itemsave(self.target, self.segmentation)
        elif len(self.parts) > 0:
            self.itemsave(self.target, self.sumOfParts)

    def saveBB(self, mode='parts'):
        '''Saves the segmentations in their respective patches (bounding boxes)

        Args:
            mode (str, optional): Either doing the whole object or only the
                                  parts

        Returns:
            The bounding box slice for that patch
        '''
        if mode == 'class':
            patch, bb = self.getSingularBB(self.segmentation.astype(int))
            self.itemsave(self.target, patch)
            return bb
        elif len(self.parts) > 0:
            patch, bb = self.getSingularBB(self.sumOfParts.astype(int))
            self.itemsave(self.target, patch)
            return bb

    def reduce(self, parts=[]):
        '''Reduces the segmentations to the specified list of parts

        Args:
            parts (list, optional): List of part names that shall be saved.
        '''
        newparts = {}
        if len(parts) > 0 and len(self.parts) > 0:
            for part in parts:
                if part in self.parts:
                    newparts[part] = self.parts[part]
        self.parts = newparts
        self.genSumOfParts()

    def genSumOfParts(self):
        '''Generate the sum of the parts or the union of segmentations.'''
        if len(self.parts) > 0:
            self.sumOfParts = next(iter(self.parts.values())) * 0
            for part in self.parts:
                self.sumOfParts += self.parts[part]

    def getSingularBB(self, img):
        '''Produces the cut part and bounding box slice for a single connected
        component in an image

        Args:
            img (image): The image to search in

        Returns:
            The part of the input image and the slice it fits.
        '''
        slices = scipy.ndimage.find_objects(img)
        # No idead why we need this:
        bb = slices[-1]
        return img[bb], bb
