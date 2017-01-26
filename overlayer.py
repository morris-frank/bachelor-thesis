
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import warnings
import glob
from tqdm import tqdm
import os.path
from scipy.misc import imread, imsave

def apply_overlay(image, overlay, path, label=''):
    """
        Overlay overlay onto image and add label as text
        and save to path (full path with extension!)
    """
    fig = plt.figure(frameon=False)
    plt.imshow(image, interpolation='none')
    plt.imshow(overlay, cmap='plasma', alpha=0.7, interpolation='none')
    if label != '':
        red_patch = mpatches.Patch(color='yellow', label=label)
        plt.legend(handles=[red_patch])
    fig.savefig(path)
    plt.close(fig)

OVERLAY_ROOT = 'data/results/overlays/'

def apply_overlaydir(overlaydir, tag):
    """
        Run apply_overlay for all images and overlays in the corresponding
        directories.
        E.g.:
        inputdir = 'test/'
        overlaydir = '../../results/test/'
        applyoverlaydir(inputdir, overlaydir)
    """
    filelist = [os.path.basename(os.path.normpath(x))
                for x in glob.glob(overlaydir + '*' + 'png')]
    filelist = np.sort(filelist)
    if not os.path.exists(OVERLAY_ROOT + tag):
        os.makedirs(OVERLAY_ROOT + tag)
    else:
        warnings.warn(
            'The directory for the patched images already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    for imf in tqdm(filelist):
        im = imread('data/datasets/voc2010/JPEGImages/' + imf[:-3] + 'jpg')
        overlay = imread(overlaydir + imf)
        path = OVERLAY_ROOT + tag + '/' + imf[:-3] + 'png'
        apply_overlay(im, overlay, path)

#apply_overlaydir('data/results/4epcohs/', '4epochs')
#apply_overlaydir('data/results/1epoch/', '1epoch')
#apply_overlaydir('data/results/4x50/', '4x50')
#apply_overlaydir('data/results/2x10/', '2x10')
apply_overlaydir('data/results/20x2-2/', '20x2-2')
