from ba.set import SetList
import ba.utils
import copy
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
        self.sourcelist = None
        self.classes = classes
        self.classlist = None
        self.parts = parts
        self.partslist = None
        if dolists:
            self.generate_lists()

    @property
    def parts(self):
        return self.__parts

    @parts.setter
    def parts(self, parts):
        if isinstance(parts, list):
            self.__parts = parts
        else:
            self.__parts = [parts]
        partstr = self.__parts
        if partstr[0] is None:
            partstr = ['']
        self.tag = '_'.join(self.classes + partstr)

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
            self.extension = '.' + ba.utils.prevalent_extension(source)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = str(name)
        self.build = self._builddir + '/' + self.__name + '/'
        ba.utils.touch(self.build)

    def write(self):
        '''Writes the generated lists to disk'''
        for name in ['source', 'class', 'parts']:
            self.__dict__[name + 'list'].rm_pre_suffix(self.source,
                                                       self.extension)
            self.__dict__[name + 'list'].write()
            self.__dict__[name + 'list'].add_pre_suffix(self.source,
                                                        self.extension)

    def generate_lists(self):
        '''Generates the *.txt lists for this set.'''
        f = {
            'source': self.build + self.name + '.txt',
            'class': self.build + '_'.join(self.classes) + '.txt',
            'parts': self.build + self.tag + '.txt'
            }
        overwrite = {}
        for name in ['source', 'class', 'parts']:
            overwrite[name] = ba.utils.query_overwrite(f[name], default='no')
            self.__dict__[name + 'list'] = SetList(f[name])
            self.__dict__[name + 'list'].add_pre_suffix(self.source,
                                                        self.extension)

        if not sum(overwrite.values()):
            return True

        if overwrite['source']:
            print('Generating List {}'.format(f['source']))
            self.sourcelist.load_directory(self.source)
            self.sourcelist.add_pre_suffix(self.source, self.extension)

        if overwrite['class'] or overwrite['parts']:
            self.classlist.list = []
            self.partslist.list = []
            print('Generating List {} and {}'.format(f['class'], f['parts']))
            for row in tqdm(self.sourcelist):
                item = PascalPart(row)
                if item.classname in self.classes or len(self.classes) < 1:
                    self.classlist.list.append(row)
                    if any(part in item.parts for part in self.parts):
                        self.partslist.list.append(row)
        self.write()

    def segmentations(self):
        '''Saves the segmentations for selected classes or parts.'''
        d = {
            'classes': '{}segmentations/{}/'.format(self.build,
                                                    '_'.join(self.classes)),
            'parts': '{}segmentations/{}/'.format(self.build, self.tag)
            }
        overwrite = {
            'classes': ba.utils.query_overwrite(d['classes'], default='no'),
            'parts': ba.utils.query_overwrite(d['parts'], default='no')
            }
        if not sum(overwrite.values()):
            return True

        print('Generating and extracting the segmentations for ' + self.tag)
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

    def bounding_boxes(self, imgdir, negatives=0, augment=0):
        '''Saves the bounding box patches for classes and parts.

        Args:
            imgdir (str): The directory where the original images live
            negatives (int, optional): How many negative samples to generate
            augment (int, optional): How many augmentations per image
        '''
        cbdir = '{}patches/{}/'.format(self.build, '_'.join(self.classes))
        pbdir = '{}patches/{}/'.format(self.build, self.tag)
        d = {
            'patch_pos': ba.utils.touch(pbdir + 'img/pos/'),
            'patch_neg': ba.utils.touch(pbdir + 'img/neg/'),
            'class_img': ba.utils.touch(cbdir + 'img/'),
            'patch_seg': ba.utils.touch(pbdir + 'seg/'),
            'class_seg': ba.utils.touch(cbdir + 'seg/')
            }
        if ba.utils.query_overwrite(pbdir, default='yes'):
            ext = ba.utils.prevalent_extension(imgdir)

            class_db = {}
            class_db_path = d['class_seg'][:-1] + '.yaml'
            patch_db = {}
            patch_db_path = d['patch_seg'][:-1] + '.yaml'

            print('''Generating and extracting the segmentation bounding
                  boxes for ''' + self.tag)
            for item in tqdm(self.classlist):
                idx = os.path.splitext(os.path.basename(item))[0]
                item = PascalPart(item)
                im = imread(imgdir + idx + '.' + ext)
                item.reduce(self.parts)

                # Save Class patches
                item.target = d['class_seg'] + idx
                bb = item.bounding_box(mode='class')
                class_db[idx] = bb
                imsave(d['class_img'] + idx + '.png', im[bb])

                # Save positive patch
                item.target = d['patch_seg'] + idx
                bb = item.bounding_box(mode='parts')
                if bb is None:
                    continue
                patch_db[idx] = bb
                imsave(d['patch_pos'] + idx + '.png', im[bb])

                # Save neagtive patches
                for i in range(0, negatives):
                    x2 = x1 = [bb[0].start, bb[1].start]
                    w = [bb[0].stop - x1[0], bb[1].stop - x1[1]]
                    subim = [im.shape[0] - w[0], im.shape[1] - w[1]]
                    checkidx = 0
                    overlap = ba.utils.slice_overlap(x1, x2, w)
                    while checkidx < 30 and overlap > 0.3:
                        checkidx += 1
                        x2 = (np.random.random(2) * subim).astype(int)
                    if checkidx >= 30:
                        continue
                    negpatch = im[x2[0]:x2[0] + w[0], x2[1]:x2[1] + w[1]]
                    imsave(d['patch_neg'] + '{}_{}.png'.format(idx, i),
                           negpatch)

            ba.utils.save(class_db_path, class_db)
            ba.utils.save(patch_db_path, patch_db)

        if ba.utils.query_overwrite(pbdir + 'img_augmented/', default='yes'):
            naugment = len(self.classlist) * augment
            self.augment_single(pbdir + 'img/pos/', naugment)
            self.augment_single(pbdir + 'img/neg/', naugment)
            self.generate_LMDB(pbdir + 'img_augmented/')

    def generate_LMDB(self, path):
        '''Generates the LMDB for the trainingset.

        Args:
            path (str): The path to the image directory. (Contains dirs pos and
                       neg)
        '''
        print('Generating LMDB for {}'.format(path))
        absp = os.path.abspath(path)
        target = {'train': absp + '_lmdb_train', 'test': absp + '_lmdb_test'}
        for d in target.values():
            ba.utils.rm(d)
            ba.utils.rm(d + '.txt')
        wh = 224
        trainlist = SetList(absp + '/pos/')
        trainlist.add_pre_suffix(absp + '/pos/', '.png 1')
        negList = SetList(absp + '/neg/')
        negList.add_pre_suffix(absp + '/neg/', '.png 0')
        n = {'pos': int(len(trainlist) * 0.2), 'neg': int(len(negList) * 0.2)}
        testlist = copy.deepcopy(trainlist)
        testlist.list = testlist.list[:n['pos']] + negList.list[:n['neg']]
        trainlist.list = trainlist.list[n['pos']:] + negList.list[n['neg']:]
        trainlist.target = target['train'] + '.txt'
        testlist.target = target['test'] + '.txt'
        trainlist.write()
        testlist.write()
        cmdstr = 'convert_imageset --resize_height=' + wh
        cmdstr += ' --resize_height=' + wh + ' --shuffle "/"'
        os.system('{} "{}" "{}" '.format(cmdstr, trainlist.target,
                                         target['train']))
        os.system('{} "{}" "{}" '.format(cmdstr, testlist.target,
                                         target['test']))

    def augment_single(self, imdir, n):
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
        ba.utils.rm(save_imdir + '/' + bn_imdir)
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
        self.unionize()

    def save(self, mode='parts'):
        '''Saves the segmentations binarly samesized to input

        Args:
            mode (str, optional): Either doing the whole object or only the
                                  parts
        '''
        if mode == 'class':
            self.itemsave(self.target, self.segmentation)
        elif len(self.parts) > 0:
            self.itemsave(self.target, self.union)

    def bounding_box(self, mode='parts'):
        '''Saves the segmentations in their respective patches (bounding boxes)

        Args:
            mode (str, optional): Either doing the whole object or only the
                                  parts

        Returns:
            The bounding box slice for that patch
        '''
        if mode == 'class':
            patch, bb = self._singularize(self.segmentation.astype(int))
            self.itemsave(self.target, patch)
            return bb
        elif len(self.parts) > 0:
            patch, bb = self._singularize(self.union.astype(int))
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
        self.unionize()

    def unionize(self):
        '''Generate the sum of the parts or the union of segmentations.'''
        if len(self.parts) > 0:
            self.union = next(iter(self.parts.values())) * 0
            for part in self.parts:
                self.union += self.parts[part]
            self.union = self.union.astype(bool)

    def _singularize(self, img):
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
