import os.path
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def apply_overlay(image, overlay, path, label=''):
    '''Overlay overlay onto image and add label as text
    and save to path (full path with extension!)

    Args:
        image (image): The image to use as 'background'.
        overlay (image): The image to overly over the image.
        path (str): The path to save the result to.
        label (str, optional): A label for the heatmap. 
    '''
    fig = plt.figure(frameon=False)
    plt.imshow(image, interpolation='none')
    plt.imshow(overlay, cmap='plasma', alpha=0.7, interpolation='none')
    if label != '':
        red_patch = mpatches.Patch(color='yellow', label=label)
        plt.legend(handles=[red_patch])
    fig.savefig(path)
    plt.close(fig)


def query_boolean(question, default='yes'):
    '''Ask a yes/no question via input() and return their answer.

    Args:
        question (str): Is a string that is presented to the user.
        default (str, optional): Is the presumed answer if the user just
            hits <Enter>. It must be 'yes' (the default), 'no' or None
            (meaning an answer is required of the user).

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


def query_overwrite(path):
    '''Checks with the user if a file shall be overwritten

    Args:
        path (str): The path to the file

    Returns:
        bool: True if write over, False if not
    '''

    if not os.path.exists(path):
        return True
    question = ('File {} does exist.\n'
                'Overwrite it?'.format(path))
    return query_boolean(question, default='yes')


def touch(path, clear=False):
    '''Touches a filepath (dir or file...)

    Args:
        path (str): The path to touch
        clear (bool): If the file shall be truncated
    '''
    dir_ = os.path.dirname(path)
    if dir_ != '':
        os.makedirs(dir_, exist_ok=True)
    if not os.path.isdir(path):
        open(path, 'a').close()
        if clear:
            open(path, 'w').close()
