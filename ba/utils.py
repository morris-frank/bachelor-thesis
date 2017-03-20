from glob import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os.path
import sys
import threading
import yaml


sys.path.append('../telenotify')
notifier_config = '../telenotify/config.yaml'


class NotifierClass(object):
    '''A class containing an notifer'''
    def __init__(self, *args, **kwargs):
        self.notifier = None

    def LOGNotifiy(self, logfile):
        '''Starts notifier thread on a given caffe - logfile

        Args:
            logfile (str): The Full path to the log file
        '''
        from telenotify import Notifier
        notifier = Notifier(configfile=notifier_config)
        threading.Thread(target=notifier._start, args=(logfile, )).start()

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
            threading.Thread(target=self.notifier.sendMessage,
                             args=(message, )).start()
        else:
            threading.Thread(target=self.notifier.sendMatrix,
                             args=(matrix, message)).start()


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


def _prepareImagePlot(image):
    '''Prepares a PyPlot figure with an image.

    Args:
        image (image): The image

    Returns:
        the figure
    '''
    xS = 3
    yS = xS / image.shape[1] * image.shape[0]
    fig = plt.figure(frameon=False, figsize=(xS,yS), dpi=image.shape[1]/xS)
    plt.axis('off')
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image, interpolation='none')
    return fig

def apply_overlay(image, overlay, path, label='', fig=None):
    '''Overlay overlay onto image and add label as text
    and save to path (full path with extension!)

    Args:
        image (image): The image to use as 'background'.
        overlay (image): The image to overly over the image.
        path (str): The path to save the result to.
        label (str, optional): A label for the heatmap.
        fig (plt.figure, optional): An optional figure to work on
    '''
    if fig is None:
        _fig = _prepareImagePlot(image)
    else:
        _fig = fig
    plt.imshow(overlay, cmap='viridis', alpha=0.5, interpolation='none')
    if label != '':
        patch = mpatches.Patch(color='yellow', label=label)
        plt.legend(handles=[patch])
    if fig is None:
        _fig.savefig(path, pad_inches=0, dpi=_fig.dpi)
        plt.close(_fig)


def apply_rect(image, rects, path, colors='black', labels='', fig=None):
    '''Overlay rectangle onto image and save to path
    (full path with extension!)

    Args:
        image (image): The image to use as 'background'.
        rects (tuple, list[tuple]): (xmin, ymin, xmax, ymax)
        path (str): The full path to save the result to.
        color (str, list[str], optional): The color for the rectangles
        labels (str, list[str], optional): The for the rectangles
        fig (plt.figure, optional): An optional figure to work on
    '''
    if fig is None:
        _fig = _prepareImagePlot(image)
    else:
        _fig = fig
    if not isinstance(rects, list):
        rects = [rects]
    if not isinstance(colors, list):
        colors = [colors]
    if len(colors) < len(rects):
        colors = [colors[0]] * len(rects)
    if not isinstance(labels, list):
        labels = [labels]
    if len(labels) < len(rects):
        labels = [labels[0]] * len(rects)
    for rect, color, label in zip(rects, colors, labels):
        height = rect[2] - rect[0]
        width = rect[3] - rect[1]
        ca = plt.gca()
        ca.add_patch(Rectangle((rect[1], rect[0]), width, height, fill=None,
                               alpha=1, ec=color, label=label))
        if label != '':
            bbox_props = dict(boxstyle='square', fc='w', ec='w')
            ca.text(rect[3] - 3, rect[0] + 5, label, ha='right', va='top',
                    size='xx-small', bbox=bbox_props)
    if fig is None:
        _fig.savefig(path, pad_inches=0, dpi=_fig.dpi)
        plt.close(_fig)


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


def prevalentExtension(path):
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


def sliding_window(image, stride, kernel_size):
    '''Slides a quadratic window over an image.

    Args:
        image (image): The image to use
        stride (int): The step size for the sliding window
        kernel_size (int): Width of the window
    '''
    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            yield (x, y, image[y:y + kernel_size, x:x + kernel_size])


def sliceOverlap(x1, x2, w):
    '''Calculates the overlap between two same sized rectangles.

    Args:
        x1 (list or tuple): First 2d-Point for the first rectangle
        x2 (list or tuple): Second 2d-Point for the second rectangle

    Returns:
        The overlap in a percentage range
    '''
    x1_2 = [x1[0] + w[0], x1[1] + w[1]]
    x2_2 = [x2[0] + w[0], x2[1] + w[1]]
    SI = max(0, min(x1_2[0], x2_2[0]) - max(x1[0], x2[0])) * max(0, min(x1_2[1], x2_2[1]) - max(x1[1], x2[1]))
    S = 2 * w[0] * w[1] - SI
    return SI / S


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


def loadYAML(path):
    '''Loads a YAML file.

    Args:
        path (str): The path to the file

    Return:
        Returns the content of the file
    '''
    with open(touch(path), 'r') as f:
        try:
            content = yaml.load(f)
        except yaml.YAMLError as e:
            print('Config {} not loadable.'.format(path))
            sys.exit(1)
    return content
