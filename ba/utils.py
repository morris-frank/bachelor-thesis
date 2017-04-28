from itertools import zip_longest
from glob import glob
import msgpack
import os.path
import sys
import threading
import yaml

sys.path.append('../telenotify')
notifier_config = '../telenotify/config.yaml'


def static_vars(**kwargs):
    '''A decorator with which it is possible to give functions static
    variables. Use:
        @static_vars(static_var=None)
        def function(arg1, arg2):
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


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


def query_overwrite(path, default='yes', defaulting=False):
    '''Checks with the user if a file shall be overwritten

    Args:
        path (str): The path to the file

    Returns:
        bool: True if write over, False if not
    '''

    if not os.path.exists(path):
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
        image (image): The image to use
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
