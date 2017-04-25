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
import collections
import random


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
        if self.__parts[0] is None:
            self.__parts = ['']
        self.tag = '_'.join(self.classes + self.__parts)

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
                if len(set(self.classes) &
                       item.classnames) > 0 or len(self.classes) < 1:
                    self.classlist.list.append(row)
                    contains_any_parts = False
                    for classname in self.classes:
                        if contains_any_parts:
                            break
                        for parts_dict in item.parts[classname]:
                            if len(set(self.parts) &
                                   set(parts_dict.keys())) > 0:
                                self.partslist.list.append(row)
                                contains_any_parts = True
                                break
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
            item.reduce(self.parts, self.classes)
            if overwrite['parts']:
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
        class_patches_base_dir = '{}patches/{}/'.format(
            self.build, '_'.join(self.classes))
        part_patches_base_dir = '{}patches/{}/'.format(
            self.build, self.tag)

        d = {
            'patch_pos': ba.utils.touch(part_patches_base_dir + 'img/pos/'),
            'patch_neg': ba.utils.touch(part_patches_base_dir + 'img/neg/'),
            'patch_seg': ba.utils.touch(part_patches_base_dir + 'seg/'),
            'class_img': ba.utils.touch(class_patches_base_dir + 'img/'),
            'class_seg': ba.utils.touch(class_patches_base_dir + 'seg/')
            }
        if ba.utils.query_overwrite(part_patches_base_dir, default='yes'):
            ext = ba.utils.prevalent_extension(imgdir)

            class_db = {}
            patch_db = {}
            class_db_path = d['class_seg'][:-1] + '.yaml'
            patch_db_path = d['patch_seg'][:-1] + '.yaml'

            print('''Generating and extracting the segmentation bounding
                  boxes for ''' + self.tag)
            for item in tqdm(self.classlist):
                idx = os.path.splitext(os.path.basename(item))[0]
                item = PascalPart(item)
                im = imread(imgdir + idx + '.' + ext)
                item.reduce(self.parts, self.classes)
                multpl_classes = len(self.classes) > 1

                # Save Class patches
                item.target = d['class_seg'] + idx
                class_bound_boxes = item.bounding_box(mode='class')
                item.target = d['patch_seg'] + idx
                part_bound_boxes = item.bounding_box(mode='parts')
                class_db[idx] = []
                patch_db[idx] = []
                patch_bb_list = []
                for classname in self.classes:
                    class_target = d['class_img'] + idx
                    patch_target = d['patch_pos'] + idx
                    if multpl_classes:
                        class_target += '_' + classname
                        patch_target += '_' + classname
                    for it, bb in enumerate(class_bound_boxes[classname]):
                        class_db[idx].append(bb)
                        imsave(class_target + '_' + str(it) + '.png', im[bb])
                    for it, bb in enumerate(part_bound_boxes[classname]):
                        patch_db[idx].append(bb)
                        patch_bb_list.append(bb)
                        imsave(patch_target + '_' + str(it) + '.png', im[bb])

                if len(patch_bb_list) > 0:
                    self._generate_negatives(d['patch_neg'] + idx, im,
                                             patch_bb_list, negatives)

            ba.utils.save(class_db_path, class_db)
            ba.utils.save(patch_db_path, patch_db)
            self.augment_and_lmdb(part_patches_base_dir, augment)

    def augment_and_lmdb(self, part_patches_base_dir, augment):
        if ba.utils.query_overwrite(part_patches_base_dir + 'img_augmented/',
                                    default='yes'):
            naugment = len(self.classlist) * augment
            self.augment_single(part_patches_base_dir + 'img/pos/', naugment)
            self.augment_single(part_patches_base_dir + 'img/neg/', naugment)
            self.generate_LMDB(part_patches_base_dir + 'img_augmented/')

    def _generate_negatives(self, basepath, im, boxes, count):
        def overlaps(coords, shape):
            return [ba.utils.slice_overlap((b[0].start, b[1].start),
                                           coords, shape) for b in boxes]

        # Save neagtive patches
        for i in range(count):
            box = random.choice(boxes)
            neg_coords = (box[0].start, box[1].start)
            shape = (box[0].stop - box[0].start,
                     box[1].stop - box[1].start)
            subim = [im.shape[0] - shape[0],
                     im.shape[1] - shape[1]]
            checkidx = 0
            while checkidx < 30 and max(overlaps(neg_coords, shape)) > 0.3:
                checkidx += 1
                neg_coords = (np.random.random(2) * subim).astype(int)
            if checkidx >= 30:
                continue
            negative_patch = im[neg_coords[0]:neg_coords[0] + shape[0],
                                neg_coords[1]:neg_coords[1] + shape[1]]
            imsave(basepath + '_{}.png'.format(i), negative_patch)

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
        cmdstr = 'convert_imageset --resize_height={}'
        cmdstr += '--resize_height={} --shuffle "/"'
        cmdstr.format(wh, wh)
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
        self.itemsave = lambda path, im: imsave(path + '.png', im)
        self.classnames = set()
        self.segmentations = collections.defaultdict(list)
        self.parts = collections.defaultdict(list)
        self.unions = collections.defaultdict(list)
        self.target = source
        self.source = source

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
            mat = mat['anno'][0][0][1][0]
        except IndexError:
            print('PascalPart::load: given file is wrong, %s', self.source)
            return False
        for submat in mat:
            classname, segmentation, parts = self._load_object(submat)
            self.classnames.add(classname)
            self.parts[classname].append(parts)
            self.segmentations[classname].append(segmentation)
        self.unionize()

    def _load_object(self, submat):
        classname = submat[0][0]
        segmentation = submat[2].astype('float')
        parts = {}
        if submat[3].size and submat[3][0].size:
            for part in submat[3][0]:
                parts[part[0][0]] = part[1].astype('float')
        return classname, segmentation, parts

    def save(self, mode='parts'):
        '''Saves the segmentations binarly same-sized to input

        Args:
            mode (str, optional): 'parts' or 'class'
        '''
        if mode == 'class':
            sources = self.segmentations
        else:
            sources = self.unions
        target = self.target
        for classname in self.classnames:
            if len(self.classnames) > 1:
                target = self.target + '_' + classname
            if len(sources[classname]) == 0:
                continue
            sum_source = sources[classname][0] * 0
            for source in sources[classname]:
                sum_source += source
            self.itemsave(target, sum_source.astype(bool))

    def bounding_box(self, mode='parts'):
        '''Saves the segmentations in their respective patches (bounding boxes)

        Args:
            mode (str, optional): Either doing the whole object or only the
                                  parts

        Returns:
            The bounding box slice for that patch
        '''
        target = self.target
        bbs = collections.defaultdict(list)
        if mode == 'class':
            sources = self.segmentations
        else:
            sources = self.unions
        for classname in self.classnames:
            if len(self.classnames) > 1:
                target = self.target + '_' + classname
            if len(sources[classname]) > 0:
                for it, source in enumerate(sources[classname]):
                    patch, bb = self._singularize(source.astype(int))
                    self.itemsave(target + '_' + str(it), patch)
                    bbs[classname].append(bb)
            return bbs

    def reduce(self, keep_parts=[], keep_classes=None):
        '''Reduces the segmentations to the specified list of parts

        Args:
            keep_parts (list, optional): List of part names that are keept.
            keep_classes (list, optional): List of classes that are keept.
        '''
        keep_parts = set(keep_parts)
        keep_classes = set(keep_classes)
        if keep_classes is None:
            keep_classes = set(self.classnames)
        self.classnames &= keep_classes
        new_parts = collections.defaultdict(list)
        for classname in self.classnames:
            for parts_dict in self.parts[classname]:
                new_parts_dict = {}
                for partname in set(parts_dict.keys()) & keep_parts:
                    new_parts_dict[partname] = parts_dict[partname]
                new_parts[classname].append(new_parts_dict)
        self.parts = new_parts
        self.unionize()

    def unionize(self):
        '''Generate the sum of the parts or the union of segmentations.'''
        self.unions = collections.defaultdict(list)
        for classname in self.classnames:
            for parts_dict in self.parts[classname]:
                if len(parts_dict) == 0:
                    continue
                union = self.segmentations[classname][0] * 0
                for part in parts_dict.values():
                    union += part.astype(int)
                union = union.astype(bool)
                self.unions[classname].append(union)

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
