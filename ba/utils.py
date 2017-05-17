from ba import BA_ROOT
from itertools import zip_longest
from glob import glob
import msgpack
import os.path
import sys
import threading
import yaml
from itertools import count
import numpy as np
import skimage.transform as tf
import scipy.misc
import skimage.color
from threading import Thread
import random

sys.path.append(BA_ROOT + '../telenotify')
notifier_config = BA_ROOT + '../telenotify/config.yaml'


class NotifierClass(object):
    '''A class containing an notifer'''
    def __init__(self, *args, **kwargs):
        self.notifier = None
        self.notifier_threads = []

    def _new_notifer_thread(self, target, args):
        self.notifier_threads.append(
            threading.Thread(target=target, args=args, daemon=True))
        self.notifier_threads[-1].start()

    def LOGNotifiy(self, logfile):
        '''Starts notifier thread on a given caffe - logfile

        Args:
            logfile (str): The Full path to the log file
        '''
        from telenotify import Notifier
        if self.notifier is None:
            self.notifier = Notifier(configfile=notifier_config)
            self.notifier.register_re(Notifier.CAFFE_TRAIN_LOSS)
            self._new_notifer_thread(self.notifier.tail,
                                     (logfile, ))

    def notify(self, message='', matrix=None):
        '''Sends message to telegram

        Args:
            message (str, optional): The message
            matrix (smth, optional): A matrix to print
        '''
        from telenotify import Notifier
        if self.notifier is None:
            self.notifier = Notifier(configfile=notifier_config)
        if matrix is None:
            self._new_notifer_thread(self.notifier.sendMessage,
                                     (message, ))
        else:
            self._new_notifer_thread(self.notifier.sendMatrix,
                                     (matrix, message))


class Bunch(object):
    '''Serves as a dictionary in the form of an object.'''
    def __init__(self, adict):
        '''Construct a bunch

        Args:
            adict (dict): The dictionary to build the object from
        '''
        self.__dict__.update(adict)

    def __str__(self):
        '''Get the string representation for the bunch (inherit from dict..)

        Returns:
            The string representation
        '''
        return self.__dict__.__str__()


def query_boolean(question, default='yes', defaulting=False):
    '''Ask a yes/no question via input() and return their answer.

    Args:
        question (str): Is a string that is presented to the user.
        default (str, optional): Is the presumed answer if the user just
            hits <Enter>. It must be 'yes' (the default), 'no' or None
            (meaning an answer is required of the user).
        defaulting (bool, optional): If we should just do the default

    Returns:
        True for 'yes' or False for 'no'.
    '''
    valid = {'yes': True, 'y': True, 'ye': True, 'j': True, 'ja': True,
             'no': False, 'n': False, 'nein': False}
    if default is None:
        prompt = ' (y/n) '
    elif default == 'yes':
        prompt = ' ([Y]/n) '
    elif default == 'no':
        prompt = ' (y/[N]) '
    else:
        raise ValueError('invalid default answer: {}'.format(default))
    if defaulting:
        return valid[default]
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            print(default)
            return valid[default]
        elif choice in valid:
            print(choice)
            return valid[choice]
        else:
            print('Please respond with "yes" or "no" '
                  '(or "y" or "n").')


def query_overwrite(path, default='yes', defaulting=False, dotouch=True):
    '''Checks with the user if a file shall be overwritten

    Args:
        path (str): The path to the file

    Returns:
        bool: True if write over, False if not
    '''

    if not os.path.exists(path):
        if dotouch:
            touch(path)
        return True
    else:
        question = ('File {} does exist.\n'
                    'Overwrite it?'.format(path))
        if query_boolean(question, default=default, defaulting=defaulting):
            touch(path, clear=True)
            return True
        else:
            return False


def load(path):
    '''Loads the contents of a serialized file. Data format in inferred from
    file extension.

    Args:
        path (str): The path to the file

    Returns:
        the unpacked contents of that file
    '''
    extension = os.path.splitext(path)[1].lower()
    if extension == '.mp':
        with open(touch(path), 'rb') as f:
            try:
                content = msgpack.load(f)
            except Exception as e:
                print('File {} not loadable.'.format(path))
                sys.exit(1)
    elif extension == '.yaml':
        with open(touch(path), 'r') as f:
            try:
                content = yaml.load(f)
            except yaml.YAMLError as e:
                print('File {} not loadable.'.format(path))
                sys.exit(1)
    else:
        raise ValueError('Invalid extension: {}'.format(path))
    return content


def save(path, content):
    '''Saves an object to a serialized file. Data format in inferred from
    file extension.

    Args:
        path (str): The target path to the file
    '''
    bpath, extension = os.path.splitext(path)
    extension = extension.lower()
    if not extension or extension == '.':
        extension = '.mp'
    path = bpath + extension
    if extension == '.mp':
        with open(path, 'wb') as f:
            msgpack.dump(content, f)
    elif extension == '.yaml':
        with open(path, 'w') as f:
            yaml.dump(content, f)
    else:
        raise ValueError('Invalid extension: {}'.format(path))


def touch(path, clear=False):
    '''Touches a filepath (dir or file...)

    Args:
        path (str): The path to touch
        clear (bool): If the file shall be truncated

    Returns:
        The given path
    '''
    dir_ = os.path.dirname(path)
    if dir_ != '':
        os.makedirs(dir_, exist_ok=True)
    if not os.path.isdir(path):
        open(path, 'a').close()
        if clear:
            open(path, 'w').close()
    return path


def prevalent_extension(path):
    '''Looks at a directory and returns the most prevalent file extension of
    the files in this directory.

    Args:
        path (str): The path to the directory

    Returns:
        the extension without leading full stop
    '''
    path = os.path.normpath(path) + '/'
    exts = [os.path.splitext(x)[1][1:] for x in glob(path + '*')]
    exts = [x for x in exts if x]
    return max(set(exts), key=exts.count)


def sliding_slice(shape, stride, kernel_size):
    if stride is None:
        stride = kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    for x1 in range(0, shape[0], stride[0]):
        for x2 in range(0, shape[1], stride[1]):
            yield (x1, x2)


def sliding_window(image, stride=None, kernel_size=(10, 10)):
    '''Slides a quadratic window over an image.

    Args:
        image (image): The image to use, channels_last orderin!!!
        stride (int): The step size for the sliding window
        kernel_size (int): Width of the window
    '''
    if stride is None:
        stride = kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    for x1, x2 in sliding_slice(image.shape, stride, kernel_size):
        yield (x1, x2, image[x1:x1 + kernel_size[0], x2:x2 + kernel_size[1]])


def slice_overlap(x1, x2, w):
    '''Calculates the overlap between two same sized rectangles.

    Args:
        x1 (list or tuple): First 2d-Point for the first rectangle
        x2 (list or tuple): Second 2d-Point for the second rectangle
        w (list or tuple): Width and height of the rectangles

    Returns:
        The overlap in a percentage range
    '''
    x1_2 = [x1[0] + w[0], x1[1] + w[1]]
    x2_2 = [x2[0] + w[0], x2[1] + w[1]]
    SI = max(0, min(x1_2[0], x2_2[0]) -
             max(x1[0], x2[0])) * max(0, min(x1_2[1], x2_2[1]) -
                                      max(x1[1], x2[1]))
    S = 2 * w[0] * w[1] - SI
    return SI / S


def slicetupel_to_rect(slicetupel):
    xs = slicetupel[0]
    ys = slicetupel[1]
    return [xs.start, ys.start, xs.stop, ys.stop]


def rm(path):
    '''Removes a path with rm and all its content, recusively.
    With great power comes great responsibility!

    Args:
        path (str): The path to the dir, which will be deleted
    '''
    if not os.path.exists(path):
        print('Tried deleting {}, but that does not even exist'.format(path))
    else:
        os.system('rm -r {}'.format(path))


def format_meter(n, total, elapsed):
    def format_interval(t):
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        if h:
            return '%d:%02d:%02d' % (h, m, s)
        else:
            return '%02d:%02d' % (m, s)
    elapsed_str = format_interval(elapsed)
    rate = '%5.2f' % (n / elapsed) if elapsed else '?'
    frac = float(n) / total

    N_BARS = 10
    bar_length = int(frac * N_BARS)
    bar = '#' * bar_length + '-' * (N_BARS - bar_length)

    percentage = '%3d%%' % (frac * 100)

    left_str = format_interval(elapsed / n * (total - n)) if n else '?'

    return '|%s| %d/%d %s [elapsed: %s left: %s, %s iters/sec]' % (
        bar, n, total, percentage, elapsed_str, left_str, rate)


def grouper(iterable, n, fillvalue=None):
    '''Slices an iterable into groups.

    Args:
        iterable (iterable): The iterable to group
        n (int): The size of the groups
        fillvalue (value, optional): With what to fill the possible unfilled
            last group

    Returns:
        The iterator over the groups
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def bounding_box_shape(bb):
    return (bb[0].stop - bb[0].start,
            bb[1].stop - bb[1].start)


class SamplesGenerator(object):
    def __init__(self, slicefiles, imlist, images_path, negatives_path,
                 ppI=None, patch_size=(100, 100), ext='jpg', mean=0,
                 batch_size=10):
        '''
        Args:
            slicefiles (str or list of str)
            imlist (list of str)
            images_path (str)
            negatives_path (str)
            ppI (int, optionale)
            patch_size (tupel, optional)
            ext (str, optional)
            mean (ndarray, optional)
        '''
        self.patch_size = patch_size
        self.mean = mean
        if isinstance(self.mean, np.ndarray) and self.mean.ndim == 3:
            self.mean = self.mean.transpose((2, 0, 1))
        self.imlist = imlist
        self.ppI = ppI
        self.batch_size = batch_size
        self.negatives_path = negatives_path

        if not isinstance(slicefiles, list):
            slicefiles = [slicefiles]
        list_of_slice_dicts = []
        n = 0
        for slicefile in slicefiles:
            add_slices, n_slices = self.load_slice_dict(slicefile)
            n += n_slices
            if add_slices != {}:
                list_of_slice_dicts.append(add_slices)

        if ppI is None:
            if n >= 100:
                self.ppI = 2
            elif n >= 50:
                self.ppI = 4
            elif n >= 4:
                self.ppI = 16
            else:
                self.ppI = 500
        if self.ppI % 2 != 0:
            raise ValueError('Count in SingleImageLayer is not divisble by 2.')

        self.samples = np.zeros((n * self.ppI * 2, 3,
                                 patch_size[0], patch_size[1]))
        it = 0
        for slice_dict in list_of_slice_dicts:
            for path, bblist in slice_dict.items():
                im = self.imread('{}{}.{}'.format(images_path, path, ext))
                for bb in bblist:
                    subslice = slice(it, n * self.ppI, n)
                    self.samples[subslice, ...] = self.sample_image(im, bb)
                    it += 1
        self.labels = np.append(np.ones(n * self.ppI), np.zeros(n * self.ppI))
        self.gen_negs(n)
        self.gen_flow()
        self.reset_threads()

    def gen_negs(self, n):
        negs = glob(self.negatives_path + '/*png')
        negs = random.sample(negs, n * self.ppI)
        for it, neg in zip(range(1, len(negs) + 1), negs):
            im = self.imread(neg)
            im /= 255
            im = tf.resize(im, (self.patch_size[0], self.patch_size[1], 3),
                           mode='reflect')
            self.samples[-it, ...] = im.transpose((2, 0, 1))

    def gen_flow(self, batch_size=None):
        from keras.preprocessing.image import ImageDataGenerator
        if batch_size is None:
            batch_size = self.batch_size
        self.flow = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            data_format='channels_first',
            preprocessing_function=self.preprocess_image
            ).flow(self.samples, self.labels, batch_size=batch_size)

    def reset_threads(self):
        self.thread_count = 10
        self.x_res = [None] * self.thread_count
        self.y_res = [None] * self.thread_count
        self.threads = [None] * self.thread_count
        self.focus = 0
        for n in range(self.thread_count):
            self._rethread(n)

    def imread(self, path):
        im = scipy.misc.imread(path)
        if im.ndim == 2:
            w, h = im.shape
            _im = np.empty((w, h, 3), dtype=np.float32)
            _im[:, :, 0] = im
            _im[:, :, 1] = im
            _im[:, :, 2] = im
            im = _im
        else:
            im = np.array(im, dtype=np.float32)
        im = im[:, :, ::-1]
        return im

    def sample_image(self, im, bb, shiftfactor=0.25):
        bbshape = bounding_box_shape(bb)
        padded_im = np.pad(im, ((bbshape[0], bbshape[0]),
                                (bbshape[1], bbshape[1]), (0, 0),),
                           mode='reflect')

        rands = (2 * np.random.random((2, self.ppI)) - 1) * shiftfactor + 1
        xsamples = (bbshape[0] * rands[0]).astype(np.int)
        ysamples = (bbshape[1] * rands[1]).astype(np.int)

        samples = np.zeros((self.ppI, 3,
                            self.patch_size[0], self.patch_size[1]))
        for it, xsample, ysample in zip(count(), xsamples, ysamples):
            _x = [bb[0].start, bb[0].stop] + xsample
            _y = [bb[1].start, bb[1].stop] + ysample
            patch = np.copy(padded_im[_x[0]:_x[1], _y[0]:_y[1], :])
            patch /= 255
            sized_patch = tf.resize(patch, (self.patch_size[0],
                                            self.patch_size[1], 3),
                                    mode='reflect')
            # sized_patch *= 255
            samples[it, ...] = sized_patch.transpose((2, 0, 1))
        return samples

    def load_slice_dict(self, slicefile):
        slicedict = load(slicefile)
        cut_slice_dict = {i: slicedict[i] for i in self.imlist
                          if i in slicedict}
        n_slices = sum([len(sl) for sl in cut_slice_dict.values()])
        return cut_slice_dict, n_slices

    def _rethread(self, n):
        self.threads[n] = Thread(target=self._next, args=(n, ))
        self.threads[n].start()

    def _next(self, n):
        x, y = next(self.flow)
        self.x_res[n] = x
        self.y_res[n] = y

    def preprocess_image(self, im):
        hsv_im = skimage.color.rgb2hsv(im.transpose(1, 2, 0))
        power_s = random.uniform(0.25, 4)
        power_v = random.uniform(0.25, 4)
        factor_s = random.uniform(0.7, 1.4)
        factor_v = random.uniform(0.7, 1.4)
        value_s = random.uniform(-0.1, 0.1)
        value_v = random.uniform(-0.1, 0.1)
        hsv_im[:, :, 1] = np.power(hsv_im[:, :, 1], power_s)
        hsv_im[:, :, 1] = hsv_im[:, :, 1] * factor_s + value_s
        hsv_im[:, :, 2] = np.power(hsv_im[:, :, 2], power_v)
        hsv_im[:, :, 2] = hsv_im[:, :, 2] * factor_v + value_v
        im = skimage.color.hsv2rgb(hsv_im).transpose(2, 0, 1)
        im *= 255
        im -= self.mean
        return im

    def next(self):
        if self.threads[self.focus].isAlive():
            self.threads[self.focus].join()
        x = self.x_res[self.focus]
        y = self.y_res[self.focus]
        self._rethread(self.focus)
        self.focus = (self.focus + 1) % self.thread_count
        return x, y
